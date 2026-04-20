# ruff: noqa: E402, E702
"""Sweep representation-quality probes across the FID-style step schedule,
at ALL transformer layers per checkpoint, with JSONL append-resume.

What this produces (for each of ~48 checkpoints × ~10 layers, plus encoders):
  P1 — long-range contact P@L/5 (headline)
  P2 — CATH fold classification (accuracy + macro-F1)

Columns in the resulting CSV: `run, step, layer, dim, p_at_L, p_at_L_2, p_at_L_5, ...`
— directly sliceable into:

  • Training-progression curve (y = metric, x = step, one curve per run/layer)
  • Layer-wise curve (y = metric, x = layer, one curve per run × step) —
    the REPA-paper Fig. 3a/b analogue

Resume behaviour:
  Results are appended to `sweep_results.jsonl` (one probe per line) as they
  complete. On rerun we read this file first and skip any (run, step, layer)
  tuple already present, so SLURM preemption / OOM / Lustre hiccups never
  waste more than one in-flight probe. Removing the JSONL forces a clean run.

Usage:
  sbatch hpc-scripts/proteina/evaluation/run_probes.sh --sweep
  python evaluation/proteina/representation/scripts/run_sweep.py --n_proteins 200

Sample selection:
  Default — first-N proteins ≤ max_size in LMDB cursor order. Deterministic
  but biased by insertion order. Every historical row in sweep_results.jsonl
  (the 231 present before 2026-04-19) was collected this way; they carry
  ``manifest: "v1"`` implicitly when re-consolidated.

  --manifest_version v2 opts into uniform-random reservoir sampling. The
  sampled LMDB keys are cached to ``batch_manifest_v2.json`` alongside the
  JSONL so all reruns of the same version see identical proteins. New rows
  are tagged ``manifest: "v2"`` so ``plot.py`` / ``pareto.py`` can filter
  apples-to-apples.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch

HERE = Path(__file__).resolve().parent
# HERE        = .../representation/scripts (for the utils/contact/cath shims)
# HERE.parent = .../representation         (so `from lib import ...` works)
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from utils import (
    LMDB_PATH,
    RUN_SCHEDULES,
    _default_device,
    extract_gearnet_embeddings,
    extract_model_hidden_states_multilayer,
    find_checkpoint_path,
    load_checkpoint_by_path,
    load_proteina_batch,
    model_num_layers,
)
from lib.manifest import (
    build_or_load_manifest,
    load_proteina_batch_from_manifest,
)
from lib.probes.contact import (
    evaluate_contact_from_scores,
    run_contact_probe_full,
)
from lib.sources import (
    LAYER_DISTANCE_ONLY,
    LAYER_RANDOM_GAUSS,
    LAYER_RANDOM_RANK,
    LAYER_SEQ_ONEHOT,
    DistanceOnlyScorer,
    RandomGaussianRepSource,
    RandomRankScorer,
    SeqOnehotRepSource,
    UntrainedProteinaRepSource,
)
from contact import run_contact_probe
from cath import run_cath_probe


GEARNET_CKPT = os.environ.get(
    "GEARNET_CKPT_PATH",
    "/rds/user/sr2173/hpc-work/proteina/data/metric_factory/model_weights/gearnet_ca.pth",
)

# Special sentinel layer used for frozen encoder rows; distinguishes them from
# real layer indices (0..nlayers-1) in the results table.
ENCODER_LAYER_SENTINEL = -1


# --------------------------------------------------------------------------- #
# Resume logic
# --------------------------------------------------------------------------- #


def _load_done_set(jsonl_path: Path) -> Tuple[Set[Tuple[str, int, int]], List[Dict]]:
    """Read existing JSONL; return set of completed (run, step, layer) keys + all rows."""
    if not jsonl_path.exists():
        return set(), []
    done: Set[Tuple[str, int, int]] = set()
    rows: List[Dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r and all(k in r for k in ("run", "step", "layer")):
                done.add((r["run"], int(r["step"]), int(r["layer"])))
            rows.append(r)
    return done, rows


def _append_row(jsonl_path: Path, row: Dict, manifest_tag: str = None) -> None:
    """Append a result row to the JSONL.

    If ``manifest_tag`` is given AND the row has no existing ``manifest`` key,
    inject it so downstream plots can filter apples-to-apples across sample
    selections. Rows written before this flag existed implicitly are ``v1``.
    """
    if manifest_tag is not None and "manifest" not in row:
        row = {**row, "manifest": manifest_tag}
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()


# --------------------------------------------------------------------------- #
# Probe helper
# --------------------------------------------------------------------------- #


def _probe_one(
    reps: torch.Tensor,
    batch,
    raw,
    cath_level: str,
    heads: List[str] = None,
    seeds: List[int] = None,
) -> Dict:
    """Run P1 + P2 on the given reps.

    heads=None, seeds=None (default) → legacy flat schema matching the 231
    historical rows: ``contact: {p_at_L, p_at_L_2, p_at_L_5, n_proteins_test}``.
    heads/seeds set explicitly → rich schema with per-head / per-seed sub-dicts
    plus back-compat flat fields (so consolidate() + historical readers still
    work).
    """
    if heads is None and seeds is None:
        contact = asdict(run_contact_probe(reps, batch))
    else:
        heads = heads or ["mlp"]
        seeds = seeds or [42]
        contact = run_contact_probe_full(reps, batch, heads=heads, seeds=seeds)
    cath = run_cath_probe(reps, batch["mask"], raw, preferred_level=cath_level)
    return {
        "dim": int(reps.shape[-1]),
        "contact": contact,
        "cath": asdict(cath),
    }


def _probe_analytic(
    source,
    batch,
    raw,
    cath_level: str,
) -> Dict:
    """Run P1 for an AnalyticScorer (random_rank / distance_only).

    Analytic sources don't emit per-residue reps, so CATH is skipped (no
    mean-pooling target). Contact path uses ``evaluate_contact_from_scores``
    over whatever scores the source produces.

    For RandomRankScorer: averages P@L/k over all seeds.
    For DistanceOnlyScorer: single deterministic scoring.
    """
    if hasattr(source, "score_pairs_per_seed"):
        per_seed = []
        for seed, scores in source.score_pairs_per_seed(batch):
            per_seed.append(evaluate_contact_from_scores(scores, batch))
        import numpy as _np

        p_L = _np.array([r.p_at_L for r in per_seed])
        p_L2 = _np.array([r.p_at_L_2 for r in per_seed])
        p_L5 = _np.array([r.p_at_L_5 for r in per_seed])
        contact = {
            "p_at_L": float(p_L.mean()),
            "p_at_L_2": float(p_L2.mean()),
            "p_at_L_5": float(p_L5.mean()),
            "p_at_L_std": float(p_L.std(ddof=0)),
            "p_at_L_2_std": float(p_L2.std(ddof=0)),
            "p_at_L_5_std": float(p_L5.std(ddof=0)),
            "n_proteins_test": per_seed[0].n_proteins_test if per_seed else 0,
            "seeds": list(source.seeds),
        }
    else:
        scores = source.score_pairs(batch)
        r = evaluate_contact_from_scores(scores, batch)
        contact = asdict(r)

    # CATH is undefined for analytic scorers; keep the sentinel shape so
    # consolidate() still reads ``r["cath"]["accuracy"]`` without KeyError.
    cath = {
        "level": "none",
        "accuracy": float("nan"),
        "macro_f1": float("nan"),
        "n_train": 0,
        "n_test": 0,
        "n_classes": 0,
    }
    return {
        "dim": 0,  # no rep dim — analytic path
        "contact": contact,
        "cath": cath,
    }


class _LoadTimeout(Exception):
    pass


def _load_ckpt_with_retry(
    ckpt_path, is_repa, device, retries=3, backoff=10, timeout_s=600
):
    """Load with retries AND a wall-clock timeout per attempt.

    SIGALRM-based timeout so one stuck checkpoint (network hang, corrupted
    state, GearNet re-init deadlock…) doesn't kill the whole sweep.
    """
    import signal

    def _handler(signum, frame):
        raise _LoadTimeout(f"checkpoint load exceeded {timeout_s}s")

    last_err = None
    for attempt in range(retries):
        try:
            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(timeout_s)
            try:
                return load_checkpoint_by_path(
                    ckpt_path, is_repa=is_repa, device=device
                )
            finally:
                signal.alarm(0)
        except Exception as e:
            last_err = e
            print(f"    load attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(backoff)
    raise last_err


def _load_gearnet():
    from proteinfoundation.repa.gearnet_encoder import GearNetPerResidueEncoder

    enc = GearNetPerResidueEncoder(ckpt_path=GEARNET_CKPT)
    enc.eval()
    return enc.to(_default_device())


# --------------------------------------------------------------------------- #
# Consolidation (CSV + MD from JSONL)
# --------------------------------------------------------------------------- #


def consolidate(outdir: Path) -> None:
    """Rebuild sweep_results.{csv,md,json} from the append-only JSONL."""
    jsonl = outdir / "sweep_results.jsonl"
    rows: List[Dict] = []
    if jsonl.exists():
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # JSON (consolidated list)
    (outdir / "sweep_results.json").write_text(json.dumps(rows, indent=2, default=str))

    # CSV
    with open(outdir / "sweep_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run",
                "step",
                "layer",
                "dim",
                "p_at_L",
                "p_at_L_2",
                "p_at_L_5",
                "contact_n_test",
                "cath_level",
                "cath_acc",
                "cath_f1",
                "cath_classes",
                "manifest",
                "ckpt_path",
                "error",
            ]
        )
        for r in rows:
            # Rows written before the manifest flag existed implicitly belong
            # to the cursor-first sample, recorded here as "v1" for clarity.
            manifest = r.get("manifest", "v1")
            if "error" in r:
                w.writerow(
                    [
                        r.get("run", ""),
                        r.get("step", ""),
                        r.get("layer", ""),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        manifest,
                        r.get("ckpt_path", ""),
                        r["error"],
                    ]
                )
                continue
            c, ca = r["contact"], r["cath"]
            w.writerow(
                [
                    r["run"],
                    r["step"],
                    r["layer"],
                    r["dim"],
                    f"{c['p_at_L']:.4f}",
                    f"{c['p_at_L_2']:.4f}",
                    f"{c['p_at_L_5']:.4f}",
                    c["n_proteins_test"],
                    ca["level"],
                    f"{ca['accuracy']:.4f}",
                    f"{ca['macro_f1']:.4f}",
                    ca["n_classes"],
                    manifest,
                    r.get("ckpt_path", ""),
                    "",
                ]
            )

    # MD summary: one table grouped by (run, step), showing the best-layer peak.
    lines = ["# Proteina Probe Sweep — peak-layer summary\n"]
    lines.append("| run | step | best_layer | P@L/5 | CATH-acc | CATH-classes |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    # Group by (run, step), take max P@L/5 across layers
    buckets: Dict[Tuple[str, int], List[Dict]] = {}
    for r in rows:
        if "error" in r:
            continue
        key = (r["run"], int(r["step"]))
        buckets.setdefault(key, []).append(r)
    for (run, step), group in sorted(buckets.items()):
        best = max(group, key=lambda r: r["contact"]["p_at_L_5"])
        lines.append(
            f"| {run} | {step} | {best['layer']} | "
            f"{best['contact']['p_at_L_5']:.3f} | {best['cath']['accuracy']:.3f} | "
            f"{best['cath']['n_classes']} |"
        )
    (outdir / "sweep_results.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_proteins", type=int, default=200)
    ap.add_argument("--max_size", type=int, default=256)
    ap.add_argument("--cath_level", type=str, default="T", choices=["C", "A", "T"])
    ap.add_argument(
        "--manifest_version",
        type=str,
        default=None,
        help=(
            "Opt into reservoir-sampled protein selection. When set (e.g. 'v2'), "
            "the sweep loads / builds `batch_manifest_<version>.json` and uses "
            "those proteins. When unset, falls back to the legacy cursor-first "
            "load_proteina_batch behavior (manifest='v1' implicit, matches "
            "historical records in sweep_results.jsonl)."
        ),
    )
    ap.add_argument(
        "--manifest_seed",
        type=int,
        default=42,
        help="Seed for reservoir sampling when building a new manifest.",
    )
    ap.add_argument(
        "--heads",
        type=str,
        default="mlp",
        help=(
            "Comma-separated probe heads to train on trained/untrained/encoder "
            "sources. Choices: mlp, linear, both. 'both' is sugar for 'mlp,linear'. "
            "Default 'mlp' preserves the legacy flat schema."
        ),
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="42",
        help=(
            "Comma-separated seeds for the contact probe train/test split and "
            "head init. Multi-seed runs produce mean + std per head. "
            "Default '42' is single-seed (legacy)."
        ),
    )
    ap.add_argument(
        "--baselines",
        type=str,
        default=None,
        help=(
            "Comma-separated list of baseline sources to include in the sweep. "
            "Choices: random_gauss, seq_onehot, untrained_proteina, "
            "random_rank, distance_only. None = skip all (legacy behavior)."
        ),
    )
    ap.add_argument(
        "--rep_dim_random",
        type=int,
        default=512,
        help="D for random_gauss reps (match the backbone's hidden dim — default 512).",
    )
    ap.add_argument(
        "--runs",
        type=str,
        default=None,
        help="Comma-separated subset of runs to probe (e.g. baseline,repa_l4).",
    )
    ap.add_argument("--skip_gearnet", action="store_true")
    # Default writes to .../representation/results/ (sibling of scripts/).
    ap.add_argument("--output_dir", type=str, default=str(HERE.parent / "results"))
    ap.add_argument(
        "--consolidate_only",
        action="store_true",
        help="Just rebuild CSV/MD from existing JSONL without probing anything.",
    )
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "sweep_results.jsonl"

    if args.consolidate_only:
        consolidate(outdir)
        print(f"Consolidated to {outdir}/sweep_results.{{csv,md,json}}")
        return

    # Normalize --heads / --seeds. Default "mlp" + "42" preserves the legacy
    # flat schema; anything else flips _probe_one to rich sub-dict output.
    heads_arg = args.heads.strip().lower()
    if heads_arg == "both":
        heads_list = ["mlp", "linear"]
    else:
        heads_list = [h.strip() for h in heads_arg.split(",") if h.strip()]
    seeds_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    rich_schema = heads_list != ["mlp"] or seeds_list != [42]
    if rich_schema:
        print(f"Rich schema: heads={heads_list}, seeds={seeds_list}")

    done, _ = _load_done_set(jsonl_path)
    if done:
        print(f"Resuming: {len(done)} probes already cached in {jsonl_path.name}")

    device = _default_device()

    # --- Shared batch: loaded once, reused everywhere ---
    # Two paths:
    #   1. --manifest_version v2+  → uniform-random sample, cached to JSON, tagged
    #                                 in each record as ``manifest=v2``.
    #   2. legacy (no flag)        → cursor-first, implicit ``manifest=v1``, matches
    #                                 the 231 historical rows in sweep_results.jsonl.
    manifest_tag = "v1"
    if args.manifest_version:
        print(
            f"Building/loading manifest '{args.manifest_version}' "
            f"({args.n_proteins} proteins, ≤ {args.max_size} residues, seed={args.manifest_seed})..."
        )
        manifest = build_or_load_manifest(
            outdir=outdir,
            version=args.manifest_version,
            lmdb_path=LMDB_PATH,
            n=args.n_proteins,
            max_size=args.max_size,
            seed=args.manifest_seed,
        )
        batch, raw = load_proteina_batch_from_manifest(manifest, device=device)
        manifest_tag = args.manifest_version
    else:
        print(
            f"Loading {args.n_proteins} proteins (≤ {args.max_size} residues) via cursor-first..."
        )
        batch, raw = load_proteina_batch(
            n=args.n_proteins, max_size=args.max_size, device=device
        )
    print(f"  Loaded {len(raw)} proteins, mask sum = {int(batch['mask'].sum().item())}")
    print(f"  Manifest tag for new rows: '{manifest_tag}'")

    # One-shot diagnostic: what does cath_code look like? CATH probe needs ≥2 classes.
    from collections import Counter

    cath_types: Counter = Counter()
    cath_examples = []
    for g in raw:
        cc = getattr(g, "cath_code", None)
        if cc is None:
            cath_types["None"] += 1
        else:
            cath_types[type(cc).__name__] += 1
            if len(cath_examples) < 3:
                cath_examples.append(cc)
    print(f"  cath_code type distribution: {dict(cath_types)}")
    print(f"  cath_code sample values: {cath_examples}")

    # --- Gearnet (flat reference, single "layer" = -1 sentinel) ---
    gearnet_key = ("gearnet", 0, ENCODER_LAYER_SENTINEL)
    if not args.skip_gearnet and gearnet_key not in done:
        print("\n=== gearnet (frozen reference) ===")
        try:
            enc = _load_gearnet()
            reps = extract_gearnet_embeddings(enc, batch)
            out = _probe_one(
                reps,
                batch,
                raw,
                args.cath_level,
                heads=heads_list if rich_schema else None,
                seeds=seeds_list if rich_schema else None,
            )
            out.update(
                {
                    "run": "gearnet",
                    "step": 0,
                    "layer": ENCODER_LAYER_SENTINEL,
                    "ckpt_path": None,
                }
            )
            _append_row(jsonl_path, out, manifest_tag=manifest_tag)
            done.add(gearnet_key)
            print(
                f"  gearnet: P@L/5={out['contact']['p_at_L_5']:.3f}  "
                f"CATH-acc={out['cath']['accuracy']:.3f}"
            )
            del enc
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  ❌ gearnet failed: {e}")
            _append_row(
                jsonl_path,
                {
                    "run": "gearnet",
                    "step": 0,
                    "layer": ENCODER_LAYER_SENTINEL,
                    "error": f"{type(e).__name__}: {e}",
                },
                manifest_tag=manifest_tag,
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    elif gearnet_key in done:
        print("gearnet already cached, skip")

    # --- Iterate (run, step) grid — probe all layers per checkpoint ---
    chosen_runs = set(args.runs.split(",")) if args.runs else set(RUN_SCHEDULES)
    for run_name in RUN_SCHEDULES:
        if run_name not in chosen_runs:
            continue
        run_dir, is_repa, _, steps = RUN_SCHEDULES[run_name]
        print(f"\n=== {run_name} ({run_dir}, {len(steps)} steps) ===")

        # Proteina 60M uses 10 trunk layers; fixed across _v2 runs. We use
        # this to skip fully-cached (run, step) pairs BEFORE paying a load.
        EXPECTED_NLAYERS = 10

        for step in steps:
            # Pre-load skip: if all expected layers are cached, don't load.
            if all((run_name, int(step), L) in done for L in range(EXPECTED_NLAYERS)):
                print(
                    f"  step {step}: all {EXPECTED_NLAYERS} layers cached, skip (no load)"
                )
                continue

            ckpt = find_checkpoint_path(run_dir, step, prefer_ema=True)
            if ckpt is None:
                print(f"  step {step}: NO EMA CKPT — skip")
                continue

            print(f"\n  --- step {step} @ {ckpt.name} ---")

            try:
                # Load model once; probe all layers from one forward pass.
                model = _load_ckpt_with_retry(ckpt, is_repa=is_repa, device=device)
                n_layers = model_num_layers(model)
                all_layers = list(range(n_layers))

                # Skip layers already done; run the remaining set.
                todo_layers = [
                    L for L in all_layers if (run_name, int(step), L) not in done
                ]
                if not todo_layers:
                    print(f"    all {n_layers} layers already cached, skip")
                    del model
                    continue
                if len(todo_layers) < n_layers:
                    print(
                        f"    resuming: {len(all_layers) - len(todo_layers)}/{n_layers} "
                        f"layers already cached, probing {len(todo_layers)} more"
                    )

                t0 = time.time()
                reps_by_layer = extract_model_hidden_states_multilayer(
                    model, batch, todo_layers
                )
                t_extract = time.time() - t0
                print(f"    extracted {len(todo_layers)} layers in {t_extract:.1f}s")

                for L in todo_layers:
                    try:
                        out = _probe_one(
                            reps_by_layer[L],
                            batch,
                            raw,
                            args.cath_level,
                            heads=heads_list if rich_schema else None,
                            seeds=seeds_list if rich_schema else None,
                        )
                        out.update(
                            {
                                "run": run_name,
                                "step": int(step),
                                "layer": int(L),
                                "ckpt_path": str(ckpt),
                            }
                        )
                        _append_row(jsonl_path, out, manifest_tag=manifest_tag)
                        done.add((run_name, int(step), L))
                        print(
                            f"      L{L:2d}: P@L/5={out['contact']['p_at_L_5']:.3f}  "
                            f"CATH-{out['cath']['level']}-acc={out['cath']['accuracy']:.3f}"
                        )
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        _append_row(
                            jsonl_path,
                            {
                                "run": run_name,
                                "step": int(step),
                                "layer": int(L),
                                "ckpt_path": str(ckpt),
                                "error": f"{type(e).__name__}: {e}",
                            },
                            manifest_tag=manifest_tag,
                        )
                del model, reps_by_layer

            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"    ❌ {run_name}@{step} failed at ckpt load / extract: {e}")
                # Record a single failure row so we don't retry this checkpoint forever.
                _append_row(
                    jsonl_path,
                    {
                        "run": run_name,
                        "step": int(step),
                        "layer": -999,
                        "ckpt_path": str(ckpt),
                        "error": f"{type(e).__name__}: {e}",
                    },
                    manifest_tag=manifest_tag,
                )
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # --- Baseline sources: random_gauss, seq_onehot, untrained_proteina,
    #      random_rank, distance_only. Each produces additive rows with
    #      negative-layer sentinels so they never collide with trunk layers 0..9.
    if args.baselines:
        requested = [b.strip() for b in args.baselines.split(",") if b.strip()]
        print(f"\n=== Baseline sources: {requested} ===")
        for name in requested:
            try:
                if name == "random_gauss":
                    src = RandomGaussianRepSource(dim=args.rep_dim_random, seed=42)
                    key = (name, 0, LAYER_RANDOM_GAUSS)
                    if key in done:
                        print(f"  {name}: cached, skip")
                        continue
                    reps = src.get_reps(batch, [LAYER_RANDOM_GAUSS])[LAYER_RANDOM_GAUSS]
                    row = _probe_one(
                        reps,
                        batch,
                        raw,
                        args.cath_level,
                        heads=heads_list if rich_schema else None,
                        seeds=seeds_list if rich_schema else None,
                    )
                    row.update(
                        {
                            "run": name,
                            "step": 0,
                            "layer": LAYER_RANDOM_GAUSS,
                            "ckpt_path": None,
                        }
                    )
                    _append_row(jsonl_path, row, manifest_tag=manifest_tag)
                    done.add(key)
                    print(f"  {name}: P@L/5={row['contact']['p_at_L_5']:.3f}")

                elif name == "seq_onehot":
                    src = SeqOnehotRepSource()
                    key = (name, 0, LAYER_SEQ_ONEHOT)
                    if key in done:
                        print(f"  {name}: cached, skip")
                        continue
                    reps = src.get_reps(batch, [LAYER_SEQ_ONEHOT])[LAYER_SEQ_ONEHOT]
                    row = _probe_one(
                        reps,
                        batch,
                        raw,
                        args.cath_level,
                        heads=heads_list if rich_schema else None,
                        seeds=seeds_list if rich_schema else None,
                    )
                    row.update(
                        {
                            "run": name,
                            "step": 0,
                            "layer": LAYER_SEQ_ONEHOT,
                            "ckpt_path": None,
                        }
                    )
                    _append_row(jsonl_path, row, manifest_tag=manifest_tag)
                    done.add(key)
                    print(f"  {name}: P@L/5={row['contact']['p_at_L_5']:.3f}")

                elif name == "untrained_proteina":
                    src = UntrainedProteinaRepSource()
                    todo = [L for L in src.emits_layers if (name, 0, L) not in done]
                    if not todo:
                        print(
                            f"  {name}: all {len(src.emits_layers)} layers cached, skip"
                        )
                        continue
                    reps_by_layer = src.get_reps(batch, todo)
                    for L, reps in reps_by_layer.items():
                        try:
                            row = _probe_one(
                                reps,
                                batch,
                                raw,
                                args.cath_level,
                                heads=heads_list if rich_schema else None,
                                seeds=seeds_list if rich_schema else None,
                            )
                            row.update(
                                {
                                    "run": name,
                                    "step": 0,
                                    "layer": int(L),
                                    "ckpt_path": None,
                                }
                            )
                            _append_row(jsonl_path, row, manifest_tag=manifest_tag)
                            done.add((name, 0, L))
                            print(f"    L{L}: P@L/5={row['contact']['p_at_L_5']:.3f}")
                        except Exception as e:
                            import traceback

                            traceback.print_exc()
                            _append_row(
                                jsonl_path,
                                {
                                    "run": name,
                                    "step": 0,
                                    "layer": int(L),
                                    "error": f"{type(e).__name__}: {e}",
                                },
                                manifest_tag=manifest_tag,
                            )
                    src.unload()

                elif name == "random_rank":
                    src = RandomRankScorer()
                    key = (name, 0, LAYER_RANDOM_RANK)
                    if key in done:
                        print(f"  {name}: cached, skip")
                        continue
                    row = _probe_analytic(src, batch, raw, args.cath_level)
                    row.update(
                        {
                            "run": name,
                            "step": 0,
                            "layer": LAYER_RANDOM_RANK,
                            "ckpt_path": None,
                        }
                    )
                    _append_row(jsonl_path, row, manifest_tag=manifest_tag)
                    done.add(key)
                    print(
                        f"  {name}: P@L/5={row['contact']['p_at_L_5']:.3f}  (mean over {len(src.seeds)} seeds)"
                    )

                elif name == "distance_only":
                    src = DistanceOnlyScorer()
                    key = (name, 0, LAYER_DISTANCE_ONLY)
                    if key in done:
                        print(f"  {name}: cached, skip")
                        continue
                    row = _probe_analytic(src, batch, raw, args.cath_level)
                    row.update(
                        {
                            "run": name,
                            "step": 0,
                            "layer": LAYER_DISTANCE_ONLY,
                            "ckpt_path": None,
                        }
                    )
                    _append_row(jsonl_path, row, manifest_tag=manifest_tag)
                    done.add(key)
                    print(f"  {name}: P@L/5={row['contact']['p_at_L_5']:.3f}")

                else:
                    print(f"  [skip] unknown baseline: {name}")

            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"  ❌ baseline {name} failed: {e}")
                _append_row(
                    jsonl_path,
                    {
                        "run": name,
                        "step": 0,
                        "layer": -999,
                        "error": f"{type(e).__name__}: {e}",
                    },
                    manifest_tag=manifest_tag,
                )
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # --- Consolidate into CSV / MD ---
    consolidate(outdir)
    print(f"\nFinal outputs in {outdir}:")
    for p in (
        "sweep_results.jsonl",
        "sweep_results.csv",
        "sweep_results.json",
        "sweep_results.md",
    ):
        print(f"  {outdir / p}")


if __name__ == "__main__":
    main()
