# ruff: noqa: E402
"""Phase 2 - Pretrained-probe sweep across the full checkpoint schedule.

For every (run, step, layer, t) in ``RUN_SCHEDULES`` + ``PRETRAINED_CHECKPOINTS``
this script trains probes on hidden states extracted from a fixed train.lmdb
manifest, evaluates them on a fixed val.lmdb manifest, and appends one row
per (probe, layer) to a JSONL.

Five probes are first-class citizens, runnable in isolation or together:

    contact          long-range contact-map prediction (P@L, P@L/2, P@L/5)
    cath             CATH topology classification (accuracy, macro-F1, top-1 conf)
    inverse_folding  per-residue 20-way AA classification (top-1, macro-F1)
    dihedral         per-residue (φ, ψ) regression (mean angular error in deg)
    distance         pair Cα-Cα distance regression (Huber-MAE in Å, bucketed)

Selection is via ``--probes contact,cath`` (the legacy default). The CATH
path takes an additional ``--cath_levels`` selector. The three new probes
target residue-/pair-level structural quality and ship with task-specific
trivial-geometric baselines (knn_dist, local_frame, seqsep_pair) that
establish a hard floor — see lib/sources.py and BASELINE_PROBE_KINDS.

Conventions match ``run_sweep.py``:
  - JSONL append-resume - re-running skips rows already present
  - ``--config`` loads a profile from ``sweep_config.yaml`` (CLI flags override)
  - ``--runs`` filters to a subset of run names
  - ``consolidate()`` rebuilds the CSV/JSON from the JSONL on every run
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
import sys
import time
import traceback
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch

HERE = Path(__file__).resolve().parent
# scripts/paper/pretrain_probe_sweep.py -> parents[1] = .../representation
sys.path.insert(0, str(HERE.parents[1]))

from lib.checkpoints import (
    LMDB_PATH,
    PRETRAINED_CHECKPOINTS,
    RUN_SCHEDULES,
    find_checkpoint_path,
    load_checkpoint_by_path,
    resolve_step,
)
from lib.data import _default_device
from lib.extract import model_num_layers
from lib.feature_cache import (
    cache_is_complete,
    extract_and_cache,
    extract_reps_in_memory,
    load_cached_layer,
    purge_cache,
)
from lib.manifest import build_or_load_manifest, load_proteina_batch_from_manifest
from lib.probes.cath_pretrained import eval_cath_probe, train_cath_probe
from lib.probes.contact_pretrained import eval_contact_probe, train_contact_probe
from lib.probes.dihedral import eval_dihedral_probe, train_dihedral_probe
from lib.probes.distance import (
    eval_distance_probe,
    eval_distance_probe_seqsep,
    train_distance_probe,
    train_distance_probe_seqsep,
)
from lib.probes.inverse_folding import (
    eval_inverse_folding_probe,
    train_inverse_folding_probe,
)
from lib.sources import (
    LAYER_SEQSEP_PAIR,
    KnnDistanceRepSource,
    LocalFrameRepSource,
    NoiseInputCheckpointRepSource,
    RandomGaussianRepSource,
    SeqOnehotRepSource,
    UntrainedProteinaRepSource,
)


VALID_BASELINES = (
    "untrained_proteina",
    "random_gauss",
    "seq_onehot",
    "trained_noise",
    # Trivial-geometric baselines for the new probes (P3-P5). Each is gated
    # via BASELINE_PROBE_KINDS to its applicable probe_kind only.
    "knn_dist",  # inverse folding only
    "local_frame",  # dihedral only
    "seqsep_pair",  # distance only (pair feature, not a RepSource)
)


VALID_CATH_LEVELS = ("C", "A", "T", "H")
VALID_PROBES = ("contact", "cath", "inverse_folding", "dihedral", "distance")
EXPECTED_INHOUSE_NLAYERS = 10  # proteina 60M trunk depth


# Which probe kinds each baseline source is allowed to fill. Generic
# representation baselines apply to every per-residue / pair probe;
# trivial-geometric baselines are scoped to the single probe they bound.
# `cath` is intentionally excluded from the new probes' baselines since the
# `run_baselines_cath` path predates this dict and remains unchanged.
BASELINE_PROBE_KINDS: Dict[str, Set[str]] = {
    "random_gauss": {"contact", "cath", "inverse_folding", "dihedral", "distance"},
    "seq_onehot": {"contact", "cath", "inverse_folding", "dihedral", "distance"},
    "untrained_proteina": {
        "contact",
        "cath",
        "inverse_folding",
        "dihedral",
        "distance",
    },
    "trained_noise": {"contact", "cath", "inverse_folding", "dihedral", "distance"},
    "knn_dist": {"inverse_folding"},
    "local_frame": {"dihedral"},
    "seqsep_pair": {"distance"},
}


# --------------------------------------------------------------------------- #
# Domain model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SweepConfig:
    """Everything that's constant across all checkpoints in one sweep."""

    # Probe selection
    run_contact: bool
    cath_levels: Tuple[str, ...]  # empty tuple => no CATH
    run_inverse_folding: bool
    run_dihedral: bool
    run_distance: bool

    # Manifest / batch sizes (logged into rows for downstream filtering)
    n_train: int
    n_eval: int

    # Contact probe hyperparams
    head_type: str  # "mlp" | "linear"
    probe_epochs: int
    probe_lr: float
    probe_seed: int

    # CATH probe hyperparams
    cath_min_per_class: int

    # Side effects
    keep_cache: bool
    save_heads_dir: Optional[Path]
    in_memory: bool  # if True, skip the disk feature cache; hold reps in RAM


@dataclass(frozen=True)
class Batches:
    """Train + eval batch tensors plus the raw Data lists CATH needs.

    Cached derived tensors (``ca_*``, ``mask_*``, ``lengths_*``) live on CPU
    and are reused across every (checkpoint, layer) - building them once
    instead of per-call avoids a tiny but real bookkeeping cost.
    """

    train: Dict[str, torch.Tensor]
    eval: Dict[str, torch.Tensor]
    raw_train: List
    raw_eval: List
    train_manifest_tag: str
    eval_manifest_tag: str

    @cached_property
    def ca_train(self) -> torch.Tensor:
        return self.train["coords"][:, :, 1, :].cpu()

    @cached_property
    def ca_eval(self) -> torch.Tensor:
        return self.eval["coords"][:, :, 1, :].cpu()

    @cached_property
    def mask_train(self) -> torch.Tensor:
        return self.train["mask"].cpu()

    @cached_property
    def mask_eval(self) -> torch.Tensor:
        return self.eval["mask"].cpu()

    @cached_property
    def lengths_train(self) -> torch.Tensor:
        return self.train["lengths"].cpu()

    @cached_property
    def lengths_eval(self) -> torch.Tensor:
        return self.eval["lengths"].cpu()


@dataclass(frozen=True)
class CheckpointJob:
    """One (run, step, ckpt, t) unit of work the harness yields and consumes."""

    run: str
    step: int
    ckpt_path: str
    is_repa: bool
    layers: Tuple[int, ...]
    t_value: float

    @property
    def t_tag(self) -> str:
        return f"{float(self.t_value):.2f}"


@dataclass
class DoneSet:
    """JSONL-derived resume state — one set per probe_kind, intent-revealing accessors.

    ``contact``         keys: (run, step, layer, t_tag).
    ``cath``            keys: (run, step, layer, t_tag, level).
    ``inverse_folding`` keys: (run, step, layer, t_tag).
    ``dihedral``        keys: (run, step, layer, t_tag).
    ``distance``        keys: (run, step, layer, t_tag).

    Pre-CATH JSONL rows had no ``probe_kind`` and are all contact rows by
    construction; treat missing-kind as contact for backward compat.
    """

    contact: Set[Tuple[str, int, int, str]] = field(default_factory=set)
    cath: Set[Tuple[str, int, int, str, str]] = field(default_factory=set)
    inverse_folding: Set[Tuple[str, int, int, str]] = field(default_factory=set)
    dihedral: Set[Tuple[str, int, int, str]] = field(default_factory=set)
    distance: Set[Tuple[str, int, int, str]] = field(default_factory=set)

    @classmethod
    def from_jsonl(cls, path: Path) -> "DoneSet":
        """Load resume state from a JSONL file *plus* any sibling shards.

        Sibling shards (``pretrained_sweep_results.<tag>.jsonl``) are written
        by parallel array tasks under ``--jsonl_shard_tag``. A given task
        reads ALL shards so it skips rows another concurrent task already
        produced — and so a resumed task picks up its own prior progress.
        """
        ds = cls()
        if path.parent.exists():
            paths = sorted(path.parent.glob("pretrained_sweep_results*.jsonl"))
        else:
            paths = []
        for p in paths:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in r or not all(
                        k in r for k in ("run", "step", "layer")
                    ):
                        continue
                    key4 = (
                        r["run"],
                        int(r["step"]),
                        int(r["layer"]),
                        f"{float(r.get('t', 1.0)):.2f}",
                    )
                    kind = r.get("probe_kind", "contact")
                    if kind == "cath":
                        ds.cath.add((*key4, r.get("cath_level", "T")))
                    elif kind == "inverse_folding":
                        ds.inverse_folding.add(key4)
                    elif kind == "dihedral":
                        ds.dihedral.add(key4)
                    elif kind == "distance":
                        ds.distance.add(key4)
                    else:
                        ds.contact.add(key4)
        return ds

    def has_contact(self, job: CheckpointJob, layer: int) -> bool:
        return (job.run, job.step, layer, job.t_tag) in self.contact

    def has_cath(self, job: CheckpointJob, layer: int, level: str) -> bool:
        return (job.run, job.step, layer, job.t_tag, level) in self.cath

    def has_inverse_folding(self, job: CheckpointJob, layer: int) -> bool:
        return (job.run, job.step, layer, job.t_tag) in self.inverse_folding

    def has_dihedral(self, job: CheckpointJob, layer: int) -> bool:
        return (job.run, job.step, layer, job.t_tag) in self.dihedral

    def has_distance(self, job: CheckpointJob, layer: int) -> bool:
        return (job.run, job.step, layer, job.t_tag) in self.distance

    def add_contact(self, job: CheckpointJob, layer: int) -> None:
        self.contact.add((job.run, job.step, layer, job.t_tag))

    def add_cath(self, job: CheckpointJob, layer: int, level: str) -> None:
        self.cath.add((job.run, job.step, layer, job.t_tag, level))

    def add_inverse_folding(self, job: CheckpointJob, layer: int) -> None:
        self.inverse_folding.add((job.run, job.step, layer, job.t_tag))

    def add_dihedral(self, job: CheckpointJob, layer: int) -> None:
        self.dihedral.add((job.run, job.step, layer, job.t_tag))

    def add_distance(self, job: CheckpointJob, layer: int) -> None:
        self.distance.add((job.run, job.step, layer, job.t_tag))

    def __len__(self) -> int:
        return (
            len(self.contact)
            + len(self.cath)
            + len(self.inverse_folding)
            + len(self.dihedral)
            + len(self.distance)
        )


# --------------------------------------------------------------------------- #
# Row IO
# --------------------------------------------------------------------------- #


def _append_row(jsonl_path: Path, row: Dict, batches: Batches) -> None:
    row = {
        **row,
        "train_manifest": batches.train_manifest_tag,
        "eval_manifest": batches.eval_manifest_tag,
    }
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()


def _append_error_row(
    jsonl_path: Path,
    job: CheckpointJob,
    err: Exception,
    batches: Batches,
    *,
    layer: int = -999,
    probe_kind: Optional[str] = None,
    cath_level: Optional[str] = None,
) -> None:
    row: Dict[str, Any] = {
        "run": job.run,
        "step": job.step,
        "layer": int(layer),
        "t": float(job.t_value),
        "ckpt_path": job.ckpt_path,
        "error": f"{type(err).__name__}: {err}",
    }
    if probe_kind is not None:
        row["probe_kind"] = probe_kind
    if cath_level is not None:
        row["cath_level"] = cath_level
    _append_row(jsonl_path, row, batches)


# --------------------------------------------------------------------------- #
# Consolidation
# --------------------------------------------------------------------------- #


_CSV_FIELDS = [
    "run",
    "step",
    "layer",
    "t",
    "probe_kind",
    "head_type",
    "n_train",
    "n_eval",
    # Contact-only columns
    "n_proteins_test",
    "p_at_L_5",
    "p_at_L_2",
    "p_at_L",
    # CATH-only columns
    "cath_level",
    "cath_accuracy",
    "cath_macro_f1",
    "cath_top1_conf_mean",
    "cath_n_classes",
    "cath_n_train",
    "cath_n_eval",
    "cath_n_eval_in_vocab",
    "cath_n_eval_oov",
    # Inverse folding columns (probe_kind="inverse_folding")
    "if_top1_acc",
    "if_macro_f1",
    "if_n_residues_test",
    # Dihedral columns (probe_kind="dihedral")
    "dih_mae_phi_deg",
    "dih_mae_psi_deg",
    "dih_mae_total_deg",
    "dih_n_residues_test",
    # Distance columns (probe_kind="distance")
    "dist_mae_total",
    "dist_mae_short",
    "dist_mae_medium",
    "dist_mae_long",
    "dist_n_pairs_test",
    # Provenance
    "dim",
    "probe_seed",
    "ckpt_path",
    "train_manifest",
    "eval_manifest",
]


def consolidate(outdir: Path) -> None:
    """Read JSONL (canonical + any shard files), write JSON (full) + CSV
    (flat columns) for plotting.

    Shard files are written when ``--jsonl_shard_tag`` is set on a per-task
    basis to avoid concurrent-append contention; they're named
    ``pretrained_sweep_results.<tag>.jsonl``. Consolidate concatenates them
    with the canonical ``pretrained_sweep_results.jsonl`` (if present).

    Dedup: the per-checkpoint array launcher (run_pretrained_probe_array.sh)
    historically ran baselines per-task before --baselines "" was added, so
    baseline rows accumulated multi-fold across per-task shards (~14× for
    untrained_proteina at n=128). We dedup by the row identity tuple and
    keep the LAST-seen row, since later writes are typically the dedicated
    baselines run output and represent the canonical (re-)computation.
    """
    jsonl_paths = sorted(outdir.glob("pretrained_sweep_results*.jsonl"))
    rows: List[Dict] = []
    for jsonl in jsonl_paths:
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Dedup by (run, step, layer, t, probe_kind, cath_level, probe_seed). Errors
    # are not deduped — they're kept verbatim so a recurring failure stays visible.
    # probe_seed is part of the key so multi-seed runs (for variance bands)
    # accumulate rather than overwrite; legacy single-seed rows are all seed 42.
    seen: Set[Tuple] = set()
    deduped: List[Dict] = []
    for r in reversed(rows):  # iterate backwards so first kept = last-seen
        if "error" in r:
            deduped.append(r)
            continue
        key = (
            r.get("run"),
            r.get("step"),
            r.get("layer"),
            f"{float(r.get('t', 1.0)):.4f}" if "t" in r else None,
            r.get("probe_kind", "contact"),
            r.get("cath_level"),
            r.get("probe_seed"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    deduped.reverse()  # restore append order

    n_dropped = len(rows) - len(deduped)
    if n_dropped > 0:
        print(
            f"[consolidate] dedup: dropped {n_dropped} duplicate rows ({len(rows)} -> {len(deduped)})"
        )

    (outdir / "pretrained_sweep_results.json").write_text(
        json.dumps(deduped, indent=2, default=str)
    )
    if not deduped:
        return

    with open(outdir / "pretrained_sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in deduped:
            if "error" in r:
                continue
            w.writerow(r)


# --------------------------------------------------------------------------- #
# Per-checkpoint orchestration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JobTodo:
    """Per-job todo list across all probe kinds. None / empty means nothing to do."""

    contact: List[int]
    cath: Dict[int, List[str]]
    inverse_folding: List[int]
    dihedral: List[int]
    distance: List[int]

    def any_layer(self) -> List[int]:
        """Union of layers needed by any probe kind, sorted."""
        return sorted(
            set(self.contact)
            | {L for L, lvls in self.cath.items() if lvls}
            | set(self.inverse_folding)
            | set(self.dihedral)
            | set(self.distance)
        )

    def is_empty(self) -> bool:
        return not (
            self.contact
            or any(self.cath.values())
            or self.inverse_folding
            or self.dihedral
            or self.distance
        )


def _todo_for_job(job: CheckpointJob, cfg: SweepConfig, done: DoneSet) -> JobTodo:
    """Subtract ``done`` from this job's (layers x probes). Pure function."""
    return JobTodo(
        contact=[L for L in job.layers if not done.has_contact(job, L)]
        if cfg.run_contact
        else [],
        cath={
            L: [lvl for lvl in cfg.cath_levels if not done.has_cath(job, L, lvl)]
            for L in job.layers
        },
        inverse_folding=[L for L in job.layers if not done.has_inverse_folding(job, L)]
        if cfg.run_inverse_folding
        else [],
        dihedral=[L for L in job.layers if not done.has_dihedral(job, L)]
        if cfg.run_dihedral
        else [],
        distance=[L for L in job.layers if not done.has_distance(job, L)]
        if cfg.run_distance
        else [],
    )


def _ensure_features_cached(
    job: CheckpointJob,
    batches: Batches,
    todo_layers: List[int],
    cache_dir: Path,
    device: str,
    require_cached: bool = False,
) -> None:
    """Extract hidden states for ``todo_layers`` if not already on disk.

    When ``require_cached=True`` (probe-from-cache mode), the cache MUST already
    exist on disk — we never load the model. Missing features raise so the
    upstream extract job can be re-run, rather than silently re-extracting
    on a CPU-only node (which would be ~100x slower).
    """
    if cache_is_complete(cache_dir, job.run, job.step, job.t_value, todo_layers):
        print("    feature cache present, skipping extraction")
        return
    if require_cached:
        raise FileNotFoundError(
            f"feature cache incomplete for {job.run} step={job.step} "
            f"t={job.t_value} in {cache_dir}; rerun the extract phase first"
        )

    print(f"    loading checkpoint: {job.ckpt_path}")
    model = load_checkpoint_by_path(job.ckpt_path, is_repa=job.is_repa, device=device)
    print(
        f"    extracting features (train={batches.train['mask'].shape[0]}, "
        f"eval={batches.eval['mask'].shape[0]}), {len(todo_layers)} layers..."
    )
    t0 = time.time()
    extract_and_cache(
        model,
        train_batch=batches.train,
        eval_batch=batches.eval,
        layers=todo_layers,
        cache_dir=cache_dir,
        run=job.run,
        step=job.step,
        t_value=job.t_value,
        train_manifest_tag=batches.train_manifest_tag,
        eval_manifest_tag=batches.eval_manifest_tag,
    )
    print(f"    extraction done in {time.time() - t0:.1f}s")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_layer_reps(
    cache_dir: Path, job: CheckpointJob, layer: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    reps_train = load_cached_layer(
        cache_dir, job.run, job.step, job.t_value, "train", layer
    )
    reps_eval = load_cached_layer(
        cache_dir, job.run, job.step, job.t_value, "eval", layer
    )
    return reps_train, reps_eval


def _save_contact_head(
    head: torch.nn.Module,
    cfg: SweepConfig,
    job: CheckpointJob,
    layer: int,
    rep_dim: int,
) -> None:
    if cfg.save_heads_dir is None:
        return
    head_path = (
        cfg.save_heads_dir
        / cfg.head_type
        / f"head_{job.run}_step{job.step}_L{layer}_t{job.t_tag}.pt"
    )
    head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "head_type": cfg.head_type,
            "in_dim": rep_dim * 3,
            "rep_dim": rep_dim,
            "run": job.run,
            "step": job.step,
            "layer": int(layer),
            "t": float(job.t_value),
            "ckpt_path": job.ckpt_path,
            "n_train": cfg.n_train,
            "n_eval": cfg.n_eval,
            "probe_seed": cfg.probe_seed,
        },
        head_path,
    )


def _save_cath_head(
    head: Any,
    meta: Dict,
    cfg: SweepConfig,
    job: CheckpointJob,
    layer: int,
    level: str,
) -> None:
    if cfg.save_heads_dir is None:
        return
    head_path = (
        cfg.save_heads_dir
        / f"cath_{level}"
        / f"cath_{job.run}_step{job.step}_L{layer}_t{job.t_tag}.pkl"
    )
    head_path.parent.mkdir(parents=True, exist_ok=True)
    with open(head_path, "wb") as f:
        pickle.dump(
            {
                "head": head,
                "train_meta": meta,
                "run": job.run,
                "step": job.step,
                "layer": int(layer),
                "t": float(job.t_value),
                "ckpt_path": job.ckpt_path,
                "n_train": cfg.n_train,
                "n_eval": cfg.n_eval,
                "probe_seed": cfg.probe_seed,
            },
            f,
        )


def _run_contact_for_layer(
    job: CheckpointJob,
    batches: Batches,
    cfg: SweepConfig,
    reps_train: torch.Tensor,
    reps_eval: torch.Tensor,
    layer: int,
    jsonl_path: Path,
    done: DoneSet,
    device: str,
) -> None:
    """Train + eval the contact probe at one layer, append a JSONL row."""
    rep_dim = int(reps_train.shape[-1])
    try:
        t0 = time.time()
        head = train_contact_probe(
            reps_train,
            batches.ca_train,
            batches.mask_train,
            batches.lengths_train,
            head_type=cfg.head_type,
            epochs=cfg.probe_epochs,
            lr=cfg.probe_lr,
            seed=cfg.probe_seed,
            device=device,
        )
        result = eval_contact_probe(
            head,
            reps_eval,
            batches.ca_eval,
            batches.mask_eval,
            batches.lengths_eval,
            device=device,
        )
        elapsed = time.time() - t0

        row = {
            "run": job.run,
            "step": job.step,
            "layer": int(layer),
            "t": float(job.t_value),
            "probe_kind": "contact",
            "head_type": cfg.head_type,
            "n_train": cfg.n_train,
            "n_eval": cfg.n_eval,
            "n_proteins_test": result.n_proteins_test,
            "p_at_L": result.p_at_L,
            "p_at_L_2": result.p_at_L_2,
            "p_at_L_5": result.p_at_L_5,
            "dim": rep_dim,
            "ckpt_path": job.ckpt_path,
            "probe_epochs": cfg.probe_epochs,
            "probe_lr": cfg.probe_lr,
            "probe_seed": cfg.probe_seed,
            "probe_train_s": float(elapsed),
        }
        _append_row(jsonl_path, row, batches)
        done.add_contact(job, layer)
        _save_contact_head(head, cfg, job, layer, rep_dim)

        print(
            f"      L{layer:2d} contact: P@L/5={result.p_at_L_5:.3f}  "
            f"P@L/2={result.p_at_L_2:.3f}  n_test={result.n_proteins_test}  "
            f"({elapsed:.1f}s)"
        )
    except Exception as e:
        traceback.print_exc()
        _append_error_row(
            jsonl_path, job, e, batches, layer=layer, probe_kind="contact"
        )


def _run_cath_for_layer(
    job: CheckpointJob,
    batches: Batches,
    cfg: SweepConfig,
    reps_train: torch.Tensor,
    reps_eval: torch.Tensor,
    layer: int,
    level: str,
    jsonl_path: Path,
    done: DoneSet,
) -> None:
    """Train + eval the CATH probe at one layer/level, append a JSONL row."""
    rep_dim = int(reps_train.shape[-1])
    try:
        t0 = time.time()
        head, meta = train_cath_probe(
            reps_train,
            batches.mask_train,
            batches.raw_train,
            level=level,
            head_type="linear",
            min_per_class=cfg.cath_min_per_class,
            seed=cfg.probe_seed,
        )
        res = eval_cath_probe(
            head, meta, reps_eval, batches.mask_eval, batches.raw_eval
        )
        elapsed = time.time() - t0

        row = {
            "run": job.run,
            "step": job.step,
            "layer": int(layer),
            "t": float(job.t_value),
            "probe_kind": "cath",
            "cath_level": level,
            "cath_head_type": "linear",
            "cath_accuracy": res.accuracy,
            "cath_macro_f1": res.macro_f1,
            "cath_top1_conf_mean": res.top1_conf_mean,
            "cath_n_classes": res.n_classes,
            "cath_n_train": res.n_train,
            "cath_n_eval": res.n_eval,
            "cath_n_eval_in_vocab": res.n_eval_in_vocab,
            "cath_n_eval_oov": res.n_eval_oov,
            "n_train": cfg.n_train,
            "n_eval": cfg.n_eval,
            "dim": rep_dim,
            "ckpt_path": job.ckpt_path,
            "probe_seed": cfg.probe_seed,
            "probe_train_s": float(elapsed),
        }
        _append_row(jsonl_path, row, batches)
        done.add_cath(job, layer, level)
        if head is not None:
            _save_cath_head(head, meta, cfg, job, layer, level)

        # `accuracy` is NaN when too few classes survived training.
        acc_str = f"{res.accuracy:.3f}" if res.accuracy == res.accuracy else "  nan"
        print(
            f"      L{layer:2d} cath-{level}: acc={acc_str}  "
            f"top1_conf={res.top1_conf_mean:.3f}  classes={res.n_classes}  "
            f"oov={res.n_eval_oov}  ({elapsed:.1f}s)"
        )
    except Exception as e:
        traceback.print_exc()
        _append_error_row(
            jsonl_path,
            job,
            e,
            batches,
            layer=layer,
            probe_kind="cath",
            cath_level=level,
        )


def _run_inverse_folding_for_layer(
    job: CheckpointJob,
    batches: Batches,
    cfg: SweepConfig,
    reps_train: torch.Tensor,
    reps_eval: torch.Tensor,
    layer: int,
    jsonl_path: Path,
    done: DoneSet,
    device: str,
) -> None:
    """Train + eval the inverse-folding probe at one layer, append a JSONL row."""
    rep_dim = int(reps_train.shape[-1])
    try:
        t0 = time.time()
        head = train_inverse_folding_probe(
            reps_train,
            batches.train["residue_type"].cpu(),
            batches.mask_train,
            head_type=cfg.head_type,
            epochs=cfg.probe_epochs,
            lr=cfg.probe_lr,
            seed=cfg.probe_seed,
            device=device,
        )
        res = eval_inverse_folding_probe(
            head,
            reps_eval,
            batches.eval["residue_type"].cpu(),
            batches.mask_eval,
            device=device,
        )
        elapsed = time.time() - t0

        row = {
            "run": job.run,
            "step": job.step,
            "layer": int(layer),
            "t": float(job.t_value),
            "probe_kind": "inverse_folding",
            "head_type": cfg.head_type,
            "n_train": cfg.n_train,
            "n_eval": cfg.n_eval,
            "if_top1_acc": res.top1_acc,
            "if_macro_f1": res.macro_f1,
            "if_n_residues_test": res.n_residues_test,
            "dim": rep_dim,
            "ckpt_path": job.ckpt_path,
            "probe_epochs": cfg.probe_epochs,
            "probe_lr": cfg.probe_lr,
            "probe_seed": cfg.probe_seed,
            "probe_train_s": float(elapsed),
        }
        _append_row(jsonl_path, row, batches)
        done.add_inverse_folding(job, layer)

        print(
            f"      L{layer:2d} inv-fold: top1={res.top1_acc:.3f}  "
            f"macro_f1={res.macro_f1:.3f}  n_res={res.n_residues_test}  "
            f"({elapsed:.1f}s)"
        )
    except Exception as e:
        traceback.print_exc()
        _append_error_row(
            jsonl_path, job, e, batches, layer=layer, probe_kind="inverse_folding"
        )


def _run_dihedral_for_layer(
    job: CheckpointJob,
    batches: Batches,
    cfg: SweepConfig,
    reps_train: torch.Tensor,
    reps_eval: torch.Tensor,
    layer: int,
    jsonl_path: Path,
    done: DoneSet,
    device: str,
) -> None:
    """Train + eval the dihedral probe at one layer, append a JSONL row."""
    rep_dim = int(reps_train.shape[-1])
    try:
        t0 = time.time()
        head = train_dihedral_probe(
            reps_train,
            batches.train["coords"].cpu(),
            batches.mask_train,
            head_type=cfg.head_type,
            epochs=cfg.probe_epochs,
            lr=cfg.probe_lr,
            seed=cfg.probe_seed,
            device=device,
        )
        res = eval_dihedral_probe(
            head,
            reps_eval,
            batches.eval["coords"].cpu(),
            batches.mask_eval,
            device=device,
        )
        elapsed = time.time() - t0

        row = {
            "run": job.run,
            "step": job.step,
            "layer": int(layer),
            "t": float(job.t_value),
            "probe_kind": "dihedral",
            "head_type": cfg.head_type,
            "n_train": cfg.n_train,
            "n_eval": cfg.n_eval,
            "dih_mae_phi_deg": res.mae_phi_deg,
            "dih_mae_psi_deg": res.mae_psi_deg,
            "dih_mae_total_deg": res.mae_total_deg,
            "dih_n_residues_test": res.n_residues_test,
            "dim": rep_dim,
            "ckpt_path": job.ckpt_path,
            "probe_epochs": cfg.probe_epochs,
            "probe_lr": cfg.probe_lr,
            "probe_seed": cfg.probe_seed,
            "probe_train_s": float(elapsed),
        }
        _append_row(jsonl_path, row, batches)
        done.add_dihedral(job, layer)

        print(
            f"      L{layer:2d} dihedral: mae_phi={res.mae_phi_deg:.1f}°  "
            f"mae_psi={res.mae_psi_deg:.1f}°  total={res.mae_total_deg:.1f}°  "
            f"({elapsed:.1f}s)"
        )
    except Exception as e:
        traceback.print_exc()
        _append_error_row(
            jsonl_path, job, e, batches, layer=layer, probe_kind="dihedral"
        )


def _run_distance_for_layer(
    job: CheckpointJob,
    batches: Batches,
    cfg: SweepConfig,
    reps_train: torch.Tensor,
    reps_eval: torch.Tensor,
    layer: int,
    jsonl_path: Path,
    done: DoneSet,
    device: str,
) -> None:
    """Train + eval the pair-distance probe at one layer, append a JSONL row."""
    rep_dim = int(reps_train.shape[-1])
    try:
        t0 = time.time()
        head = train_distance_probe(
            reps_train,
            batches.ca_train,
            batches.mask_train,
            head_type=cfg.head_type,
            epochs=cfg.probe_epochs,
            lr=cfg.probe_lr,
            seed=cfg.probe_seed,
            device=device,
        )
        res = eval_distance_probe(
            head,
            reps_eval,
            batches.ca_eval,
            batches.mask_eval,
            device=device,
        )
        elapsed = time.time() - t0

        row = {
            "run": job.run,
            "step": job.step,
            "layer": int(layer),
            "t": float(job.t_value),
            "probe_kind": "distance",
            "head_type": cfg.head_type,
            "n_train": cfg.n_train,
            "n_eval": cfg.n_eval,
            "dist_mae_total": res.mae_total,
            "dist_mae_short": res.mae_short,
            "dist_mae_medium": res.mae_medium,
            "dist_mae_long": res.mae_long,
            "dist_n_pairs_test": res.n_pairs_test,
            "dim": rep_dim,
            "ckpt_path": job.ckpt_path,
            "probe_epochs": cfg.probe_epochs,
            "probe_lr": cfg.probe_lr,
            "probe_seed": cfg.probe_seed,
            "probe_train_s": float(elapsed),
        }
        _append_row(jsonl_path, row, batches)
        done.add_distance(job, layer)

        print(
            f"      L{layer:2d} distance: total={res.mae_total:.2f}Å  "
            f"short={res.mae_short:.2f}  med={res.mae_medium:.2f}  "
            f"long={res.mae_long:.2f}  ({elapsed:.1f}s)"
        )
    except Exception as e:
        traceback.print_exc()
        _append_error_row(
            jsonl_path, job, e, batches, layer=layer, probe_kind="distance"
        )


def _run_baseline_cath_at_layer(
    run_name: str,
    layer_sentinel: int,
    reps_train: torch.Tensor,
    reps_eval: torch.Tensor,
    batches: Batches,
    cfg: SweepConfig,
    level: str,
    jsonl_path: Path,
    done: DoneSet,
) -> None:
    """Fit + score one (baseline, layer, level) row. Mirrors _run_cath_for_layer
    but keyed off a synthetic ``CheckpointJob`` (step=0, t=1.0) so DoneSet
    resume logic still works without forking a parallel schema.
    """
    job = CheckpointJob(
        run=run_name,
        step=0,
        ckpt_path="<baseline>",
        is_repa=False,
        layers=(layer_sentinel,),
        t_value=1.0,
    )
    if done.has_cath(job, layer_sentinel, level):
        return
    _run_cath_for_layer(
        job,
        batches,
        cfg,
        reps_train,
        reps_eval,
        layer_sentinel,
        level,
        jsonl_path,
        done,
    )


def run_baselines_cath(
    names: List[str],
    batches: Batches,
    cfg: SweepConfig,
    done: DoneSet,
    jsonl_path: Path,
    device: str,
    random_gauss_dim: int = 512,
    random_gauss_seed: int = 42,
    untrained_config_name: str = "training_baseline_256",
    trained_noise_run: Optional[str] = None,
    trained_noise_step: Optional[int] = None,
) -> None:
    """Run the CATH probe over the three rep-source baselines.

    Each baseline emits per-residue features for ``batches.train`` and
    ``batches.eval``; we then call the shared ``train_cath_probe`` /
    ``eval_cath_probe`` pair once per CATH level. Sentinel ``layer`` codes
    (negative; see ``lib/sources.py``) keep these rows from colliding with
    real trunk-layer rows in the JSONL.

    Baselines are checkpoint-agnostic and t-agnostic, so we tag every row
    step=0, t=1.0. Resume logic via DoneSet is reused as-is.
    """
    if not cfg.cath_levels:
        return
    if not names:
        return
    print(f"\n=== Baselines (CATH): {names} ===")

    def _get(src, batch):
        # All three sources return {layer_sentinel: [B, N, D]} on CPU; the
        # untrained one emits 10 sentinel layers, the others a single layer.
        return src.get_reps(batch, list(src.emits_layers))

    for name in names:
        if name not in VALID_BASELINES:
            print(f"  [skip] unknown baseline: {name}")
            continue
        try:
            if name == "random_gauss":
                src = RandomGaussianRepSource(
                    dim=random_gauss_dim, seed=random_gauss_seed
                )
            elif name == "seq_onehot":
                src = SeqOnehotRepSource()
            elif name == "trained_noise":
                if not trained_noise_run:
                    print(
                        "  [skip] trained_noise baseline requires "
                        "--baseline_trained_noise_run (or yaml `baseline_trained_noise_run:`)"
                    )
                    continue
                if trained_noise_run not in RUN_SCHEDULES:
                    print(
                        f"  [skip] trained_noise baseline: '{trained_noise_run}' not in RUN_SCHEDULES"
                    )
                    continue
                run_dir, is_repa, _bs, steps = RUN_SCHEDULES[trained_noise_run]
                step = (
                    trained_noise_step if trained_noise_step is not None else steps[-1]
                )
                from lib.checkpoints import find_checkpoint_path

                ckpt_path = find_checkpoint_path(run_dir, step)
                if ckpt_path is None:
                    print(
                        f"  [skip] trained_noise: ckpt not found for {trained_noise_run} step={step}"
                    )
                    continue
                print(
                    f"  [trained_noise] using {trained_noise_run} step={step} ({ckpt_path})"
                )
                src = NoiseInputCheckpointRepSource(
                    ckpt_path=str(ckpt_path),
                    is_repa=is_repa,
                    run_name="trained_noise",
                )
            else:  # untrained_proteina
                src = UntrainedProteinaRepSource(config_name=untrained_config_name)

            print(f"  [{name}] extracting train reps...")
            reps_train_by_L = _get(src, batches.train)
            print(f"  [{name}] extracting eval reps...")
            reps_eval_by_L = _get(src, batches.eval)

            for L in src.emits_layers:
                reps_train = reps_train_by_L[L]
                reps_eval = reps_eval_by_L[L]
                for level in cfg.cath_levels:
                    _run_baseline_cath_at_layer(
                        name,
                        L,
                        reps_train,
                        reps_eval,
                        batches,
                        cfg,
                        level,
                        jsonl_path,
                        done,
                    )

            src.unload()
        except Exception as e:
            traceback.print_exc()
            print(f"  [X] baseline {name} failed: {e}")
            # Write a single error row so the failure is visible in the JSONL.
            err_job = CheckpointJob(
                run=name,
                step=0,
                ckpt_path="<baseline>",
                is_repa=False,
                layers=(-999,),
                t_value=1.0,
            )
            _append_error_row(
                jsonl_path,
                err_job,
                e,
                batches,
                layer=-999,
                probe_kind="cath",
            )


def _baseline_synth_job(name: str, layer_sentinel: int) -> CheckpointJob:
    """Build a synthetic CheckpointJob for a baseline row. step=0, t=1.0 keep
    DoneSet's (run, step, layer, t_tag) keying unique without touching schema."""
    return CheckpointJob(
        run=name,
        step=0,
        ckpt_path="<baseline>",
        is_repa=False,
        layers=(layer_sentinel,),
        t_value=1.0,
    )


def _run_baseline_seqsep_distance(
    batches: Batches,
    cfg: SweepConfig,
    jsonl_path: Path,
    done: DoneSet,
    device: str,
) -> None:
    """Distance-probe head fed only |i-j| (no representation at all).

    Sets the chain-prior floor: any rep that doesn't beat this MAE has
    learned nothing about geometry beyond the trivial chain ordering.
    """
    job = _baseline_synth_job("seqsep_pair", LAYER_SEQSEP_PAIR)
    if done.has_distance(job, LAYER_SEQSEP_PAIR):
        return
    try:
        t0 = time.time()
        head = train_distance_probe_seqsep(
            batches.ca_train,
            batches.mask_train,
            head_type=cfg.head_type,
            epochs=cfg.probe_epochs,
            lr=cfg.probe_lr,
            seed=cfg.probe_seed,
            device=device,
        )
        res = eval_distance_probe_seqsep(
            head,
            batches.ca_eval,
            batches.mask_eval,
            device=device,
        )
        elapsed = time.time() - t0
        row = {
            "run": "seqsep_pair",
            "step": 0,
            "layer": int(LAYER_SEQSEP_PAIR),
            "t": 1.0,
            "probe_kind": "distance",
            "head_type": cfg.head_type,
            "n_train": cfg.n_train,
            "n_eval": cfg.n_eval,
            "dist_mae_total": res.mae_total,
            "dist_mae_short": res.mae_short,
            "dist_mae_medium": res.mae_medium,
            "dist_mae_long": res.mae_long,
            "dist_n_pairs_test": res.n_pairs_test,
            "dim": 1,  # head input is 1-d (|i-j|)
            "ckpt_path": "<baseline>",
            "probe_seed": cfg.probe_seed,
            "probe_train_s": float(elapsed),
        }
        _append_row(jsonl_path, row, batches)
        done.add_distance(job, LAYER_SEQSEP_PAIR)
        print(
            f"  [seqsep_pair] distance: total={res.mae_total:.2f}Å  "
            f"short={res.mae_short:.2f}  med={res.mae_medium:.2f}  "
            f"long={res.mae_long:.2f}  ({elapsed:.1f}s)"
        )
    except Exception as e:
        traceback.print_exc()
        _append_error_row(
            jsonl_path,
            job,
            e,
            batches,
            layer=LAYER_SEQSEP_PAIR,
            probe_kind="distance",
        )


def run_baselines_new_probes(
    names: List[str],
    chosen_kinds: Set[str],
    batches: Batches,
    cfg: SweepConfig,
    done: DoneSet,
    jsonl_path: Path,
    device: str,
    random_gauss_dim: int = 512,
    random_gauss_seed: int = 42,
    untrained_config_name: str = "training_baseline_256",
    trained_noise_run: Optional[str] = None,
    trained_noise_step: Optional[int] = None,
) -> None:
    """Run the inverse-folding / dihedral / distance probes over baseline sources.

    For each ``(baseline, kind)`` cell allowed by ``BASELINE_PROBE_KINDS``,
    extracts per-residue features once from the source and dispatches to the
    matching ``_run_<kind>_for_layer``. Synthetic CheckpointJobs (step=0,
    t=1.0) keep the DoneSet keying disjoint from real-checkpoint rows.

    The ``seqsep_pair`` baseline is special-cased because its "feature" is
    pair-only (|i-j|) and bypasses the per-residue rep contract entirely.
    """
    chosen_new = chosen_kinds & {"inverse_folding", "dihedral", "distance"}
    if not chosen_new or not names:
        return
    print(f"\n=== Baselines (new probes): {names} kinds={sorted(chosen_new)} ===")

    def _dispatch(
        kind: str,
        src_job: CheckpointJob,
        layer: int,
        reps_train: torch.Tensor,
        reps_eval: torch.Tensor,
    ) -> None:
        if kind == "inverse_folding":
            if done.has_inverse_folding(src_job, layer):
                return
            _run_inverse_folding_for_layer(
                src_job,
                batches,
                cfg,
                reps_train,
                reps_eval,
                layer,
                jsonl_path,
                done,
                device,
            )
        elif kind == "dihedral":
            if done.has_dihedral(src_job, layer):
                return
            _run_dihedral_for_layer(
                src_job,
                batches,
                cfg,
                reps_train,
                reps_eval,
                layer,
                jsonl_path,
                done,
                device,
            )
        elif kind == "distance":
            if done.has_distance(src_job, layer):
                return
            _run_distance_for_layer(
                src_job,
                batches,
                cfg,
                reps_train,
                reps_eval,
                layer,
                jsonl_path,
                done,
                device,
            )

    for name in names:
        if name not in VALID_BASELINES:
            print(f"  [skip] unknown baseline: {name}")
            continue
        kinds_here = BASELINE_PROBE_KINDS.get(name, set()) & chosen_new
        if not kinds_here:
            continue

        # Special-case the pair-only seqsep baseline — no RepSource involved.
        if name == "seqsep_pair":
            _run_baseline_seqsep_distance(batches, cfg, jsonl_path, done, device)
            continue

        try:
            if name == "random_gauss":
                src = RandomGaussianRepSource(
                    dim=random_gauss_dim, seed=random_gauss_seed
                )
            elif name == "seq_onehot":
                src = SeqOnehotRepSource()
            elif name == "knn_dist":
                src = KnnDistanceRepSource()
            elif name == "local_frame":
                src = LocalFrameRepSource()
            elif name == "trained_noise":
                if not trained_noise_run:
                    print(
                        "  [skip] trained_noise baseline requires "
                        "--baseline_trained_noise_run"
                    )
                    continue
                if trained_noise_run not in RUN_SCHEDULES:
                    print(
                        f"  [skip] trained_noise: '{trained_noise_run}' not in RUN_SCHEDULES"
                    )
                    continue
                run_dir, is_repa, _bs, steps = RUN_SCHEDULES[trained_noise_run]
                step = (
                    trained_noise_step if trained_noise_step is not None else steps[-1]
                )
                ckpt_path = find_checkpoint_path(run_dir, step)
                if ckpt_path is None:
                    print(
                        f"  [skip] trained_noise: ckpt not found for {trained_noise_run} step={step}"
                    )
                    continue
                print(f"  [trained_noise] using {trained_noise_run} step={step}")
                src = NoiseInputCheckpointRepSource(
                    ckpt_path=str(ckpt_path),
                    is_repa=is_repa,
                    run_name="trained_noise",
                )
            else:  # untrained_proteina
                src = UntrainedProteinaRepSource(config_name=untrained_config_name)

            print(f"  [{name}] extracting train reps...")
            reps_train_by_L = src.get_reps(batches.train, list(src.emits_layers))
            print(f"  [{name}] extracting eval reps...")
            reps_eval_by_L = src.get_reps(batches.eval, list(src.emits_layers))

            for L in src.emits_layers:
                rt = reps_train_by_L[L]
                re_ = reps_eval_by_L[L]
                for kind in sorted(kinds_here):
                    job = _baseline_synth_job(name, L)
                    _dispatch(kind, job, L, rt, re_)

            src.unload()
        except Exception as e:
            traceback.print_exc()
            print(f"  [X] baseline {name} failed: {e}")
            err_job = _baseline_synth_job(name, -999)
            _append_error_row(jsonl_path, err_job, e, batches, layer=-999)


def run_one_checkpoint(
    job: CheckpointJob,
    batches: Batches,
    cfg: SweepConfig,
    done: DoneSet,
    cache_dir: Path,
    jsonl_path: Path,
    device: str,
    extract_only: bool = False,
    probe_from_cache: bool = False,
) -> None:
    """Top-level per-checkpoint flow: extract -> per-layer probes -> maybe purge.

    Two split-mode flags:
      extract_only      — run only the GPU feature-extraction phase, keep the
                          cache on disk for a later probe-from-cache job.
      probe_from_cache  — skip extraction (no GPU needed), require features
                          already on disk, run probes only.

    Use these to split the GPU and CPU phases across different SLURM jobs:
    extract under intr (≤1h GPU), then probe-fit on a CPU-only node.
    """
    todo = _todo_for_job(job, cfg, done)
    if todo.is_empty():
        print(f"    t={job.t_tag}: all layers cached, skip")
        return

    todo_layers = todo.any_layer()

    # In-memory mode: load model, forward once for all layers, hold reps in RAM,
    # run all probes, drop. No disk cache, no purge. Trades disk for memory
    # (~12-24 GB / ckpt at n=256, fp16). Suited for SLURM array jobs at one
    # task per checkpoint; mutually exclusive with extract_only / probe_from_cache.
    if cfg.in_memory:
        if extract_only or probe_from_cache:
            raise SystemExit(
                "--in_memory is mutually exclusive with --extract_only / --probe_from_cache"
            )
        print(f"    [in-memory] loading checkpoint: {job.ckpt_path}")
        t0 = time.time()
        model = load_checkpoint_by_path(
            job.ckpt_path, is_repa=job.is_repa, device=device
        )
        print(f"    [in-memory] model loaded in {time.time() - t0:.1f}s")
        try:
            reps_by_L = extract_reps_in_memory(
                model,
                train_batch=batches.train,
                eval_batch=batches.eval,
                layers=todo_layers,
                t_value=job.t_value,
            )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for L in todo_layers:
            reps_train = reps_by_L[L]["train"].float()
            reps_eval = reps_by_L[L]["eval"].float()
            if L in todo.contact:
                _run_contact_for_layer(
                    job,
                    batches,
                    cfg,
                    reps_train,
                    reps_eval,
                    L,
                    jsonl_path,
                    done,
                    device,
                )
            for level in todo.cath.get(L, []):
                _run_cath_for_layer(
                    job, batches, cfg, reps_train, reps_eval, L, level, jsonl_path, done
                )
            if L in todo.inverse_folding:
                _run_inverse_folding_for_layer(
                    job,
                    batches,
                    cfg,
                    reps_train,
                    reps_eval,
                    L,
                    jsonl_path,
                    done,
                    device,
                )
            if L in todo.dihedral:
                _run_dihedral_for_layer(
                    job,
                    batches,
                    cfg,
                    reps_train,
                    reps_eval,
                    L,
                    jsonl_path,
                    done,
                    device,
                )
            if L in todo.distance:
                _run_distance_for_layer(
                    job,
                    batches,
                    cfg,
                    reps_train,
                    reps_eval,
                    L,
                    jsonl_path,
                    done,
                    device,
                )
            # Free this layer's reps as soon as its probes are done so peak
            # RAM stays bounded at ~2 layers' worth (current + next).
            del reps_by_L[L], reps_train, reps_eval
            gc.collect()
        return

    # Default: disk-cache mode (original behaviour).
    _ensure_features_cached(
        job, batches, todo_layers, cache_dir, device, require_cached=probe_from_cache
    )

    if extract_only:
        # Stop after extraction so a downstream CPU job picks up probe-fits.
        # Keep the cache (don't purge) — caller is responsible for cleanup
        # once the probe-fit phase has consumed the features.
        return

    for L in todo_layers:
        reps_train, reps_eval = _load_layer_reps(cache_dir, job, L)
        if L in todo.contact:
            _run_contact_for_layer(
                job, batches, cfg, reps_train, reps_eval, L, jsonl_path, done, device
            )
        for level in todo.cath.get(L, []):
            _run_cath_for_layer(
                job, batches, cfg, reps_train, reps_eval, L, level, jsonl_path, done
            )
        if L in todo.inverse_folding:
            _run_inverse_folding_for_layer(
                job, batches, cfg, reps_train, reps_eval, L, jsonl_path, done, device
            )
        if L in todo.dihedral:
            _run_dihedral_for_layer(
                job, batches, cfg, reps_train, reps_eval, L, jsonl_path, done, device
            )
        if L in todo.distance:
            _run_distance_for_layer(
                job, batches, cfg, reps_train, reps_eval, L, jsonl_path, done, device
            )

    if not cfg.keep_cache:
        purge_cache(cache_dir, job.run, job.step, job.t_value)


# --------------------------------------------------------------------------- #
# Job iteration (in-house schedule + pretrained anchors)
# --------------------------------------------------------------------------- #


def _peek_pretrained_n_layers(
    ckpt_path: str, is_repa: bool, device: str
) -> Optional[int]:
    """Briefly load a pretrained ckpt to count its trunk layers (10 vs 12).

    In-house runs are always 10 layers so we skip this load for them.
    """
    try:
        model = load_checkpoint_by_path(ckpt_path, is_repa=is_repa, device=device)
        n = model_num_layers(model)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return n
    except Exception as e:
        print(f"  [X] could not determine n_layers for {ckpt_path}: {e}")
        return None


def iter_jobs(
    runs_filter: Optional[Set[str]],
    steps_override: Optional[List[int]],
    timesteps: List[float],
    skip_pretrained_refs: bool,
    device: str,
    probe_from_cache: bool = False,
    cache_dir: Optional[Path] = None,
) -> Iterator[CheckpointJob]:
    """Yield every (run, step, t) we need to probe, in schedule order.

    In-house runs (RUN_SCHEDULES): nlayers=10, t from --timesteps.
    Pretrained anchors (PRETRAINED_CHECKPOINTS): forced t=1.0 (clean input),
    nlayers detected by briefly loading the checkpoint.
    """
    inhouse_layers = tuple(range(EXPECTED_INHOUSE_NLAYERS))

    for run_name, (run_dir, is_repa, _bs, steps) in RUN_SCHEDULES.items():
        if runs_filter is not None and run_name not in runs_filter:
            continue
        steps_here = steps_override if steps_override is not None else steps
        print(f"\n=== {run_name} ({run_dir}, {len(steps_here)} steps) ===")
        for step_spec in steps_here:
            try:
                ckpt = find_checkpoint_path(run_dir, step_spec, prefer_ema=True)
            except (FileNotFoundError, OSError) as e:
                # In probe_from_cache mode we don't need the ckpt — features
                # are already on disk. Look up the matching cache dir to get
                # the resolved step. (Important for run_dirs that have been
                # cleaned up after extraction, like deleted ESM training runs.)
                ckpt = None
                if probe_from_cache and cache_dir is not None:
                    matches = sorted(
                        Path(cache_dir).glob(f"features_{run_name}_step*_t*")
                    )
                    if matches:
                        # Parse step from cache dir name: features_<run>_step<N>_t<t>
                        for m in matches:
                            try:
                                step_int = int(m.name.split("_step")[-1].split("_t")[0])
                            except ValueError:
                                continue
                            print(
                                f"\n  --- step {step_int} @ <from-cache> (ckpt deleted) ---"
                            )
                            for t_val in timesteps:
                                yield CheckpointJob(
                                    run=run_name,
                                    step=step_int,
                                    ckpt_path="<from-cache>",
                                    is_repa=is_repa,
                                    layers=inhouse_layers,
                                    t_value=float(t_val),
                                )
                        continue
                print(f"  step {step_spec}: ckpt lookup failed ({e}) - skip")
                continue
            if ckpt is None:
                print(f"  step {step_spec}: NO EMA CKPT - skip")
                continue
            step_int = resolve_step(ckpt, step_spec)
            print(f"\n  --- step {step_int} @ {ckpt.name} ---")
            for t_val in timesteps:
                yield CheckpointJob(
                    run=run_name,
                    step=int(step_int),
                    ckpt_path=str(ckpt),
                    is_repa=is_repa,
                    layers=inhouse_layers,
                    t_value=float(t_val),
                )

    if skip_pretrained_refs or not PRETRAINED_CHECKPOINTS:
        return

    chosen_pre = runs_filter if runs_filter is not None else set(PRETRAINED_CHECKPOINTS)
    for pre_name, (ckpt_path, is_repa, _exp_nlayers) in PRETRAINED_CHECKPOINTS.items():
        if pre_name not in chosen_pre:
            continue
        if not Path(ckpt_path).exists():
            print(f"  [X] {pre_name}: ckpt not found: {ckpt_path}")
            continue
        print(f"\n=== {pre_name} ({ckpt_path}) ===")
        if probe_from_cache and cache_dir is not None:
            # No model load on CPU node — infer layer count from cached files.
            # PRETRAINED_CHECKPOINTS[name][2] holds the expected nlayers (used
            # as the upper bound; the actual extracted set is whatever's on
            # disk under the matching features_<run>_step0_t1.00 dir).
            n = _exp_nlayers
        else:
            n = _peek_pretrained_n_layers(ckpt_path, is_repa, device)
            if n is None:
                continue
        yield CheckpointJob(
            run=pre_name,
            step=0,
            ckpt_path=ckpt_path,
            is_repa=is_repa,
            layers=tuple(range(n)),
            t_value=1.0,
        )


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #


def _default_train_lmdb() -> str:
    p = Path(os.environ.get("PROBES_LMDB_PATH", LMDB_PATH))
    return str(p.parent / "train.lmdb")


def _parse_probes(spec: str) -> Set[str]:
    chosen = {s.strip() for s in spec.split(",") if s.strip()}
    unknown = chosen - set(VALID_PROBES)
    if unknown:
        raise SystemExit(
            f"--probes: unknown {sorted(unknown)}; valid: {sorted(VALID_PROBES)}"
        )
    if not chosen:
        raise SystemExit(
            f"--probes must include at least one of {sorted(VALID_PROBES)}"
        )
    return chosen


def _parse_cath_levels(spec: str) -> Tuple[str, ...]:
    levels = tuple(s.strip() for s in spec.split(",") if s.strip())
    bad = [lvl for lvl in levels if lvl not in VALID_CATH_LEVELS]
    if bad:
        raise SystemExit(f"--cath_levels: {bad} not in {list(VALID_CATH_LEVELS)}")
    return levels


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--max_size", type=int, default=256)
    ap.add_argument("--head_type", type=str, default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--probe_epochs", type=int, default=15)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--probe_seed", type=int, default=42)
    ap.add_argument("--manifest_seed", type=int, default=42)
    ap.add_argument("--train_manifest_version", type=str, default="train_v1")
    ap.add_argument("--eval_manifest_version", type=str, default="eval_v1")
    ap.add_argument("--train_lmdb_path", type=str, default=None)
    ap.add_argument("--eval_lmdb_path", type=str, default=None)
    ap.add_argument(
        "--runs",
        type=str,
        default=None,
        help="Comma-separated subset of RUN_SCHEDULES / PRETRAINED_CHECKPOINTS keys.",
    )
    ap.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated explicit step list (overrides schedule per-run).",
    )
    ap.add_argument(
        "--timesteps",
        type=str,
        default="1.0",
        help="Comma-separated flow-matching timesteps. Default 1.0 (clean).",
    )
    ap.add_argument("--skip_pretrained_refs", action="store_true")
    ap.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Per-checkpoint feature cache directory. Default: output_dir/cache.",
    )
    ap.add_argument(
        "--keep_cache",
        action="store_true",
        help="Don't purge per-checkpoint cache after probing.",
    )
    ap.add_argument(
        "--save_heads",
        action="store_true",
        help="Save trained probe heads to <output_dir>/probe_heads/.",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=str(HERE.parents[1] / "results" / "paper" / "n256_paper_contact"),
    )
    ap.add_argument(
        "--consolidate_only",
        action="store_true",
        help="Rebuild CSV/JSON from existing JSONL, no probing.",
    )
    ap.add_argument(
        "--probes",
        type=str,
        default="contact,cath",
        help=(
            "Comma-separated probe selection: any subset of "
            f"{{{', '.join(VALID_PROBES)}}}. Default 'contact,cath' (legacy). "
            "New residue-level probes: 'inverse_folding' (per-residue AA "
            "classification), 'dihedral' (per-residue (φ, ψ) regression), "
            "'distance' (pair Cα-Cα regression)."
        ),
    )
    ap.add_argument(
        "--cath_levels",
        type=str,
        default="T",
        help=(
            "Comma-separated CATH levels to probe per layer (only used when "
            "'cath' is in --probes). Any subset of "
            f"{{{', '.join(VALID_CATH_LEVELS)}}}. Default 'T'."
        ),
    )
    ap.add_argument("--cath_min_per_class", type=int, default=3)
    ap.add_argument(
        "--baselines",
        type=str,
        default="",
        help=(
            "Comma-separated baselines to add. Any subset of "
            f"{{{', '.join(VALID_BASELINES)}}}. Generic baselines "
            "(random_gauss, seq_onehot, untrained_proteina, trained_noise) "
            "apply to every chosen probe; trivial-geometric baselines are "
            "scoped via BASELINE_PROBE_KINDS (knn_dist→inverse_folding, "
            "local_frame→dihedral, seqsep_pair→distance). Each row uses a "
            "sentinel negative layer code (see lib/sources.py). Empty = skip."
        ),
    )
    ap.add_argument(
        "--baseline_random_gauss_dim",
        type=int,
        default=512,
        help="Hidden dim D for random_gauss baseline reps (default 512).",
    )
    ap.add_argument(
        "--baseline_random_gauss_seed",
        type=int,
        default=42,
        help="Seed for the single random_gauss draw.",
    )
    ap.add_argument(
        "--baseline_untrained_config",
        type=str,
        default="training_baseline_256",
        help=(
            "Hydra config name for the freshly-initialised Proteina used as the "
            "untrained_proteina baseline. The transformer arch is shared across "
            "sequence lengths, so the default works for any sweep."
        ),
    )
    ap.add_argument(
        "--baseline_trained_noise_run",
        type=str,
        default="",
        help=(
            "RUN_SCHEDULES key for the canonical trained checkpoint used by the "
            "trained_noise baseline (good model + reference-distribution input "
            "at t=0). Required if 'trained_noise' is in --baselines."
        ),
    )
    ap.add_argument(
        "--baseline_trained_noise_step",
        type=int,
        default=None,
        help=(
            "Step for --baseline_trained_noise_run. Defaults to the last step "
            "listed for that run in RUN_SCHEDULES."
        ),
    )
    ap.add_argument(
        "--extract_only",
        action="store_true",
        help=(
            "Run only the GPU feature-extraction phase. Skips probe fitting, "
            "keeps the cache on disk. Pair with --cache_dir pointing at a "
            "persistent location (e.g. /rds/.../features/<sweep>/), then run "
            "a follow-up --probe_from_cache job on a CPU-only node."
        ),
    )
    ap.add_argument(
        "--probe_from_cache",
        action="store_true",
        help=(
            "Skip the GPU feature-extraction phase. Requires features to "
            "already exist in --cache_dir (typically populated by a prior "
            "--extract_only run). Errors loudly rather than re-extracting "
            "if features are missing."
        ),
    )
    ap.add_argument(
        "--baselines_only",
        action="store_true",
        help=(
            "Skip every (run, step) in RUN_SCHEDULES + PRETRAINED_CHECKPOINTS "
            "and run only the rep-source baselines (see --baselines). Use "
            "this to backfill baseline rows into an existing JSONL without "
            "re-touching checkpoint rows. Requires --baselines to be set."
        ),
    )
    ap.add_argument(
        "--in_memory",
        action="store_true",
        help=(
            "Skip the per-checkpoint disk feature cache; hold all-layer reps "
            "in RAM (~12-24 GB at n=256, fp16) for the duration of probe-fit "
            "for one checkpoint, then drop. Each layer's reps are released "
            "as soon as that layer's probes finish. Suited for SLURM array "
            "jobs running one task per checkpoint. Mutually exclusive with "
            "--extract_only / --probe_from_cache."
        ),
    )
    ap.add_argument(
        "--jsonl_shard_tag",
        type=str,
        default="",
        help=(
            "Per-task JSONL shard suffix for parallel-safe writes. When set, "
            "results write to pretrained_sweep_results.<tag>.jsonl instead "
            "of the shared file. Use with --runs <single> in array jobs to "
            "avoid concurrent-append contention. consolidate() globs all "
            "shards plus the canonical file when rebuilding CSV/JSON."
        ),
    )
    return ap


def _apply_config_profile(
    ap: argparse.ArgumentParser, config_name: Optional[str]
) -> None:
    """If --config <name> was passed, fold its yaml fields into the parser
    defaults so subsequent CLI args can still override individually."""
    if config_name is None:
        return
    import yaml

    with open(HERE.parents[1] / "sweep_config.yaml") as f:
        all_profiles = yaml.safe_load(f)
    if config_name not in all_profiles:
        ap.error(
            f"Unknown --config '{config_name}'. "
            f"Available: {[k for k in all_profiles if not k.startswith('_')]}"
        )
    profile = {k: v for k, v in all_profiles[config_name].items() if v is not None}
    ap.set_defaults(**profile)
    print(f"[config] Loaded profile '{config_name}': {profile}")


def _load_batches(
    train_lmdb: str,
    eval_lmdb: str,
    args: argparse.Namespace,
    outdir: Path,
    device: str,
) -> Batches:
    """Build/load both manifests, materialise both batches + raw lists."""
    print(f"\n[manifest] train: v={args.train_manifest_version}, n={args.n_train}")
    train_manifest = build_or_load_manifest(
        outdir=outdir,
        version=args.train_manifest_version,
        lmdb_path=train_lmdb,
        n=args.n_train,
        max_size=args.max_size,
        seed=args.manifest_seed,
    )
    print(f"[manifest] eval: v={args.eval_manifest_version}, n={args.n_eval}")
    eval_manifest = build_or_load_manifest(
        outdir=outdir,
        version=args.eval_manifest_version,
        lmdb_path=eval_lmdb,
        n=args.n_eval,
        max_size=args.max_size,
        seed=args.manifest_seed,
    )
    # The cached manifest's lmdb_path may point at a previous job's /dev/shm
    # directory that's gone. Keys are stable (same source LMDB on Lustre),
    # so we override the path with the current run's value.
    train_manifest["lmdb_path"] = train_lmdb
    eval_manifest["lmdb_path"] = eval_lmdb

    print("\n[load] eval batch...")
    eval_batch, raw_eval = load_proteina_batch_from_manifest(
        eval_manifest, device=device
    )
    print(f"  {int(eval_batch['mask'].sum().item())} real residues")
    print("[load] train batch...")
    train_batch, raw_train = load_proteina_batch_from_manifest(
        train_manifest, device=device
    )
    print(f"  {int(train_batch['mask'].sum().item())} real residues")

    return Batches(
        train=train_batch,
        eval=eval_batch,
        raw_train=raw_train,
        raw_eval=raw_eval,
        train_manifest_tag=args.train_manifest_version,
        eval_manifest_tag=args.eval_manifest_version,
    )


def _log_cath_label_coverage(batches: Batches) -> None:
    def _has(g) -> bool:
        cc = getattr(g, "cath_code", None)
        if cc is None:
            return False
        return (not hasattr(cc, "__len__")) or len(cc) > 0

    n_train_lab = sum(1 for g in batches.raw_train if _has(g))
    n_eval_lab = sum(1 for g in batches.raw_eval if _has(g))
    print(
        f"[cath] train labelled: {n_train_lab}/{len(batches.raw_train)} "
        f"({100 * n_train_lab / max(1, len(batches.raw_train)):.1f}%), "
        f"eval labelled: {n_eval_lab}/{len(batches.raw_eval)} "
        f"({100 * n_eval_lab / max(1, len(batches.raw_eval)):.1f}%)"
    )


def main() -> None:
    ap = _build_arg_parser()
    pre, _ = ap.parse_known_args()
    _apply_config_profile(ap, pre.config)
    args = ap.parse_args()

    if args.extract_only and args.probe_from_cache:
        ap.error("--extract_only and --probe_from_cache are mutually exclusive")
    if args.baselines_only and not args.baselines:
        ap.error("--baselines_only requires --baselines (or a config with baselines:)")

    train_lmdb = args.train_lmdb_path or _default_train_lmdb()
    eval_lmdb = args.eval_lmdb_path or LMDB_PATH

    outdir = Path(args.output_dir)
    if not outdir.is_absolute():
        outdir = HERE.parents[1] / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    if args.jsonl_shard_tag:
        # Per-task shard so concurrent array tasks don't append-contend on the
        # same file. consolidate() globs all shards + canonical when rebuilding.
        safe_tag = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in args.jsonl_shard_tag
        )
        jsonl_path = outdir / f"pretrained_sweep_results.{safe_tag}.jsonl"
        print(f"[shard] writing rows to {jsonl_path.name}")
    else:
        jsonl_path = outdir / "pretrained_sweep_results.jsonl"

    cache_dir = Path(args.cache_dir) if args.cache_dir else outdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.consolidate_only:
        consolidate(outdir)
        print(f"Consolidated to {outdir}/pretrained_sweep_results.{{csv,json}}")
        return

    # --- Probe selection ---
    chosen_probes = _parse_probes(args.probes)
    cath_levels = (
        _parse_cath_levels(args.cath_levels) if "cath" in chosen_probes else ()
    )
    if "cath" in chosen_probes and not cath_levels:
        ap.error("--probes includes 'cath' but --cath_levels is empty")

    save_heads_dir = (outdir / "probe_heads") if args.save_heads else None
    if save_heads_dir is not None:
        save_heads_dir.mkdir(parents=True, exist_ok=True)
        print(f"[save_heads] writing probe heads to {save_heads_dir}")

    cfg = SweepConfig(
        run_contact=("contact" in chosen_probes),
        cath_levels=cath_levels,
        run_inverse_folding=("inverse_folding" in chosen_probes),
        run_dihedral=("dihedral" in chosen_probes),
        run_distance=("distance" in chosen_probes),
        n_train=args.n_train,
        n_eval=args.n_eval,
        head_type=args.head_type,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        probe_seed=args.probe_seed,
        cath_min_per_class=args.cath_min_per_class,
        # Force keep_cache=True under extract_only so the downstream probe-fit
        # job can pick up the features. Honour the user's flag otherwise.
        keep_cache=args.keep_cache or args.extract_only,
        save_heads_dir=save_heads_dir,
        in_memory=args.in_memory,
    )

    timesteps = [float(t.strip()) for t in args.timesteps.split(",") if t.strip()] or [
        1.0
    ]

    print(f"Probes: {sorted(chosen_probes)}")
    if cfg.run_contact:
        print(
            f"  contact: head={cfg.head_type}, epochs={cfg.probe_epochs}, lr={cfg.probe_lr}"
        )
    if cfg.cath_levels:
        print(
            f"  cath: levels={list(cfg.cath_levels)}, min_per_class={cfg.cath_min_per_class}"
        )
    if cfg.run_inverse_folding:
        print(
            f"  inverse_folding: head={cfg.head_type}, epochs={cfg.probe_epochs}, lr={cfg.probe_lr}"
        )
    if cfg.run_dihedral:
        print(
            f"  dihedral: head={cfg.head_type}, epochs={cfg.probe_epochs}, lr={cfg.probe_lr}"
        )
    if cfg.run_distance:
        print(
            f"  distance: head={cfg.head_type}, epochs={cfg.probe_epochs}, lr={cfg.probe_lr}"
        )
    print(f"Train LMDB: {train_lmdb}")
    print(f"Eval  LMDB: {eval_lmdb}")
    print(f"N_train={cfg.n_train}, N_eval={cfg.n_eval}, max_size={args.max_size}")
    print(f"Cache dir: {cache_dir}")
    print(f"Output dir: {outdir}")

    done = DoneSet.from_jsonl(jsonl_path)
    if len(done):
        print(
            f"Resuming: contact={len(done.contact)} cath={len(done.cath)} "
            f"if={len(done.inverse_folding)} dih={len(done.dihedral)} "
            f"dist={len(done.distance)} rows in JSONL"
        )

    device = _default_device()

    # --- Build batches once, reused across every job ---
    batches = _load_batches(train_lmdb, eval_lmdb, args, outdir, device)
    if cfg.cath_levels:
        _log_cath_label_coverage(batches)

    # --- Iterate the full schedule ---
    runs_filter = set(args.runs.split(",")) if args.runs else None
    steps_override = (
        [int(s.strip()) for s in args.steps.split(",") if s.strip()]
        if args.steps
        else None
    )

    if args.extract_only:
        print("[mode] EXTRACT-ONLY: writing features to cache, skipping probe fits")
    elif args.probe_from_cache:
        print("[mode] PROBE-FROM-CACHE: reading features from cache, no GPU forward")

    if args.baselines_only:
        print("[mode] BASELINES-ONLY: skipping all checkpoint jobs")
    else:
        for job in iter_jobs(
            runs_filter=runs_filter,
            steps_override=steps_override,
            timesteps=timesteps,
            skip_pretrained_refs=args.skip_pretrained_refs,
            device=device,
            probe_from_cache=args.probe_from_cache,
            cache_dir=cache_dir,
        ):
            try:
                run_one_checkpoint(
                    job,
                    batches,
                    cfg,
                    done,
                    cache_dir,
                    jsonl_path,
                    device,
                    extract_only=args.extract_only,
                    probe_from_cache=args.probe_from_cache,
                )
            except Exception as e:
                traceback.print_exc()
                print(f"    [X] {job.run}@{job.step} t={job.t_tag} failed: {e}")
                _append_error_row(jsonl_path, job, e, batches)

    if args.extract_only:
        print(
            f"\n[out] EXTRACT-ONLY done. Cache at {cache_dir}. "
            f"Run --probe_from_cache --cache_dir {cache_dir} on a CPU node."
        )
        return

    # --- Baseline rows (checkpoint-agnostic, run after main schedule) ---
    baseline_names = [b.strip() for b in args.baselines.split(",") if b.strip()]
    if baseline_names:
        # CATH baselines (legacy path; only applies when CATH is enabled).
        if cfg.cath_levels:
            cath_names = [
                n
                for n in baseline_names
                if "cath" in BASELINE_PROBE_KINDS.get(n, set())
            ]
            if cath_names:
                run_baselines_cath(
                    cath_names,
                    batches,
                    cfg,
                    done,
                    jsonl_path,
                    device,
                    random_gauss_dim=args.baseline_random_gauss_dim,
                    random_gauss_seed=args.baseline_random_gauss_seed,
                    untrained_config_name=args.baseline_untrained_config,
                    trained_noise_run=args.baseline_trained_noise_run,
                    trained_noise_step=args.baseline_trained_noise_step,
                )

        # New-probe baselines (inverse_folding / dihedral / distance).
        if chosen_probes & {"inverse_folding", "dihedral", "distance"}:
            run_baselines_new_probes(
                baseline_names,
                chosen_probes,
                batches,
                cfg,
                done,
                jsonl_path,
                device,
                random_gauss_dim=args.baseline_random_gauss_dim,
                random_gauss_seed=args.baseline_random_gauss_seed,
                untrained_config_name=args.baseline_untrained_config,
                trained_noise_run=args.baseline_trained_noise_run,
                trained_noise_step=args.baseline_trained_noise_step,
            )

    consolidate(outdir)
    print(f"\n[out] Consolidated to {outdir}/pretrained_sweep_results.{{csv,json}}")


if __name__ == "__main__":
    main()
