"""Multi-checkpoint generation sweep for Proteina evaluation.

Orchestrates the generation + FID/designability/diversity metric pipeline
across a schedule of checkpoints, mirroring how
``representation/scripts/run_sweep.py`` orchestrates contact/CATH probes.

Each task runs one (run_name, step) pair and appends a result row to a shared
``sweep_results.jsonl``.  On re-run the done-set is read first and already-
completed tasks are skipped, so SLURM preemption never wastes more than one
in-flight checkpoint.

Under the hood each task delegates to ``evaluate.py``'s ``main()`` by
patching ``sys.argv``, which preserves the per-checkpoint tensor checkpoint
resume and Hydra config loading logic unchanged.

Usage:
    # Dry run - print task index table and exit
    python run_sweep.py --config n128 --dry_run

    # SLURM array (one task per checkpoint)
    sbatch --array=0-3 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n128

    # Backfill a single run within a profile
    sbatch --array=0-11 hpc-scripts/proteina/evaluation/generation/run_sweep.sh \\
        --config n512_convergence --runs repa_l4

    # Ad-hoc single checkpoint (bypasses RUN_SCHEDULES)
    sbatch hpc-scripts/proteina/evaluation/generation/run_sweep.sh \\
        --ckpt_path /rds/...ckpt --ckpt_label myrun_100k \\
        --config_name inference/inference_fid_60m_baseline_lite

    # Rebuild CSV/MD from existing JSONL without re-running generation
    python run_sweep.py --config n128 --consolidate_only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
# HERE        = .../generation/scripts
# HERE.parent = .../generation
# HERE.parent.parent = .../proteina  (evaluation/proteina)
sys.path.insert(0, str(HERE))  # -> generation/scripts/ for evaluate module
sys.path.insert(
    0, str(HERE.parent.parent)
)  # -> evaluation/proteina/ for lib.checkpoints

from lib.checkpoints import (  # noqa: E402
    GEN_RUN_CONFIGS,
    RUN_SCHEDULES,
    find_checkpoint_path,
    resolve_step,
)

# ``evaluate`` is imported lazily inside ``run_one_task`` because it pulls in
# proteinfoundation modules that aren't on sys.path during --dry_run /
# --consolidate_only invocations from a vanilla shell. METRIC_PREFIX and the
# shared CLI helper come from ``_metric_args`` directly so this module stays
# importable without the proteina src tree.
from utils._metric_args import METRIC_PREFIX, add_metric_args  # noqa: E402

SWEEP_CONFIG_PATH = HERE.parent / "sweep_config.yaml"


# -- Sweep config loading ------------------------------------------------------


def _load_sweep_config(profile: str) -> Dict:
    """Load a named profile from sweep_config.yaml."""
    import yaml

    with open(SWEEP_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    if profile not in raw:
        available = [k for k in raw if not k.startswith("_")]
        raise ValueError(f"Unknown profile '{profile}'. Available: {available}")
    defaults = raw.get("_defaults", {})
    merged = {**defaults, **raw[profile]}
    merged.pop("<<", None)  # YAML merge key artefact
    return merged


# -- Task list construction ----------------------------------------------------


def load_ckpts_file(path: Path) -> List[Tuple[str, Optional[int], str, Path]]:
    """Parse a TSV of (label, ckpt_path, config_name) into a task list.

    Each non-comment line must have exactly three TAB-separated fields. Step
    is left None so resolve_step picks the actual integer from the checkpoint
    filename (or last-EMA fallback) at execution time.
    """
    if not path.exists():
        raise FileNotFoundError(f"--ckpts_file not found: {path}")
    tasks: List[Tuple[str, Optional[int], str, Path]] = []
    seen_labels: Set[str] = set()
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{lineno}: expected 3 TAB-separated fields "
                    f"(label, ckpt_path, config_name), got {len(parts)}: {line!r}"
                )
            label, ckpt_path, config_name = (p.strip() for p in parts)
            if not label or not ckpt_path or not config_name:
                raise ValueError(f"{path}:{lineno}: empty field in {line!r}")
            if label in seen_labels:
                raise ValueError(
                    f"{path}:{lineno}: duplicate label {label!r}; labels are "
                    f"used as the JSONL key and must be unique within a file."
                )
            seen_labels.add(label)
            tasks.append((label, None, config_name, Path(ckpt_path)))
    if not tasks:
        raise ValueError(f"{path}: no checkpoint rows found")
    return tasks


_PROTOCOL_SUFFIXES = ("_fid", "_des")


def _strip_protocol_suffix(run_name: str) -> str:
    """Return the RUN_SCHEDULES key for a run name with an optional `_fid` /
    `_des` protocol suffix.

    Paper-protocol sweeps use one RUN_SCHEDULES entry per checkpoint and let
    the run-name suffix select the protocol (FID vs des/div). We map e.g.
    ``baseline_256_ep21_fid`` → ``baseline_256_ep21`` for the schedule lookup.
    Names that aren't in RUN_SCHEDULES even after stripping fall through
    unchanged so the caller's unknown-run error fires with the original name.
    """
    if run_name in RUN_SCHEDULES:
        return run_name
    for suf in _PROTOCOL_SUFFIXES:
        if run_name.endswith(suf):
            stripped = run_name[: -len(suf)]
            if stripped in RUN_SCHEDULES:
                return stripped
    return run_name


def build_task_list(
    runs: List[str],
) -> List[Tuple[str, Optional[int], str, Path]]:
    """Return a flat list of (run_name, step, config_name, ckpt_path) tuples.

    Expands each run's step_list from RUN_SCHEDULES.  Tasks are ordered
    run-first then step-ascending so a partial --array covers early steps of
    all runs before late steps of any run.

    Run names ending in ``_fid`` / ``_des`` are protocol-suffixed paper-style
    keys; their RUN_SCHEDULES entry lives under the suffix-stripped name
    while their GEN_RUN_CONFIGS entry uses the full suffixed name.
    """
    tasks = []
    for run_name in runs:
        schedule_key = _strip_protocol_suffix(run_name)
        if schedule_key not in RUN_SCHEDULES:
            raise ValueError(
                f"Unknown run '{run_name}'. Available: {sorted(RUN_SCHEDULES)}"
            )
        if run_name not in GEN_RUN_CONFIGS:
            raise ValueError(
                f"Run '{run_name}' has no GEN_RUN_CONFIGS entry. "
                f"Add it to evaluation/proteina/lib/checkpoints.py."
            )
        run_dir, _is_repa, _layer, step_list = RUN_SCHEDULES[schedule_key]
        config_name = GEN_RUN_CONFIGS[run_name]
        for step in step_list:
            ckpt_path = find_checkpoint_path(run_dir, step)
            tasks.append((run_name, step, config_name, ckpt_path))
    return tasks


# -- JSONL resume helpers ------------------------------------------------------


def _load_done_set(jsonl_path: Path) -> Tuple[Set[Tuple[str, int]], List[Dict]]:
    """Read existing JSONL; return set of completed (run, step) keys + all rows."""
    if not jsonl_path.exists():
        return set(), []
    done: Set[Tuple[str, int]] = set()
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
            if "error" not in r and "run" in r and "step" in r:
                done.add((r["run"], int(r["step"])))
            rows.append(r)
    return done, rows


def _append_row(jsonl_path: Path, row: Dict) -> None:
    """Append one result row to the JSONL (OS-level append atomicity)."""
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()


# -- Result consolidation ------------------------------------------------------

# Metric columns written by evaluate.py that we surface in the sweep CSV.
_METRIC_COLS = [
    "_res_PDB_FID",
    "_res_PDB_fJSD_C",
    "_res_PDB_fJSD_A",
    "_res_PDB_fJSD_T",
    "_res_AFDB_FID",
    "_res_AFDB_fJSD_C",
    "_res_AFDB_fJSD_A",
    "_res_AFDB_fJSD_T",
    "_res_fS_C",
    "_res_fS_A",
    "_res_fS_T",
    "_res_designability_rate",
    "_res_scRMSD_mean",
    "_res_scRMSD_median",
    "_res_designability_n",
    "_res_diversity_clusters_mean",
    "_res_diversity_pairwise_tm_mean",
    "_res_diversity_n",
    "_res_diversity_designable_filtered",
    "_res_novelty_rate",
    "_res_novelty_max_tm_mean",
    "_res_novelty_designable_filtered",
    "_res_novelty_foldseek_pdb_rate",
    "_res_novelty_foldseek_pdb_max_tm_mean",
    "_res_novelty_foldseek_pdb_n",
    "_res_novelty_foldseek_afdb_swissprot_rate",
    "_res_novelty_foldseek_afdb_swissprot_max_tm_mean",
    "_res_novelty_foldseek_afdb_swissprot_n",
]


def _backfill_from_csv(rows: List[Dict], repo_root: Path) -> int:
    """For each result row missing _res_ columns, try to read them from the
    per-checkpoint CSV written by evaluate.py.  Returns number of rows updated."""
    try:
        import pandas as pd
    except ImportError:
        return 0
    updated = 0
    for r in rows:
        if "error" in r or "run" not in r or "step" not in r or "config_name" not in r:
            continue
        config_slug = r["config_name"].replace("/", "_")
        output_suffix = f"sweep_{r['run']}_step_{r['step']}"
        csv_path = (
            repo_root
            / f"eval_output/{config_slug}_{output_suffix}"
            / f"results_{config_slug}_{output_suffix}_fid.csv"
        )
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if len(df) == 0:
            continue
        new_vals = {
            k: v
            for k, v in df.iloc[0].items()
            if str(k).startswith(METRIC_PREFIX) and k not in r
        }
        if new_vals:
            r.update(new_vals)
            updated += 1
    return updated


def consolidate(jsonl_path: Path, output_dir: Path) -> None:
    """Rebuild the sweep_results.md human summary from the JSONL.

    JSONL is the source of truth (append-only, drives the resume-skip set and
    every plot loader). MD is a glanceable side-effect for IDE / PR review;
    everything plot-facing reads JSONL directly via load_sweep_rows().
    """
    _, rows = _load_done_set(jsonl_path)
    if not rows:
        print("No rows in JSONL yet, skipping consolidation.")
        return

    # Deduplicate: for each (run, step) pair keep only the last non-error entry.
    seen_keys: Dict[Tuple, int] = {}
    for i, r in enumerate(rows):
        if "error" not in r and "run" in r and "step" in r:
            seen_keys[(r["run"], int(r["step"]))] = i
    rows = [r for i, r in enumerate(rows) if i in seen_keys.values()]

    # Backfill _res_ columns from per-checkpoint CSVs for rows that were written
    # before diversity/novelty/cath metrics existed.
    repo_root = HERE.parent.parent.parent.parent  # repo root
    n_updated = _backfill_from_csv(rows, repo_root)
    if n_updated:
        print(
            f"  Backfilled _res_ columns from per-checkpoint CSVs for {n_updated} rows."
        )

    # MD summary - one row per (run, step), key metrics only
    md_path = output_dir / "sweep_results.md"
    present_metrics = [c for c in _METRIC_COLS if any(c in r for r in rows)]
    header = ["run", "step"] + present_metrics
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    result_rows = [r for r in rows if "error" not in r]
    result_rows.sort(key=lambda r: (r.get("run", ""), int(r.get("step", 0))))
    for r in result_rows:
        vals = [str(r.get("run", "")), str(r.get("step", ""))]
        for m in present_metrics:
            v = r.get(m, "")
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Consolidated {len(result_rows)} rows -> {md_path}")


# -- Single-task execution -----------------------------------------------------


def run_one_task(
    run_name: str,
    step: Optional[int],
    config_name: str,
    ckpt_path: Optional[Path],
    output_dir: Path,
    seed: int,
    designability_subset_per_length: int,
    designability_lengths: Optional[str],
    skip_fid: bool,
    fast_inference: bool,
    jsonl_path: Path,
    cath_subset: int = 0,
    cath_head_path: Optional[str] = None,
    centroid_path: Optional[str] = None,
    centroid_filter_designable: bool = True,
    foldseek_target_dbs: Optional[str] = None,
    foldseek_alignment_type: int = 2,
    foldseek_max_seqs: int = 1000,
    foldseek_sensitivity: float = 9.5,
    foldseek_threads: int = 0,
    foldseek_filter_designable: bool = True,
    metrics: Optional[str] = None,
    skip_generation: bool = False,
    ss_reference_pdb_path: Optional[str] = None,
    ss_reference_afdb_path: Optional[str] = None,
) -> None:
    """Run evaluate.main() for one (run_name, step) and append to JSONL."""
    # Lazy import: evaluate pulls in proteinfoundation, which isn't always on
    # sys.path (e.g. during --dry_run from a vanilla shell). Pulling it in
    # only when we're about to actually generate keeps the orchestrator
    # importable in lighter modes.
    import evaluate  # noqa: PLC0415

    if ckpt_path is None or not ckpt_path.exists():
        msg = f"Checkpoint not found for run={run_name} step={step}: {ckpt_path}"
        print(f"ERROR: {msg}")
        _append_row(jsonl_path, {"run": run_name, "step": step, "error": msg})
        return

    # Resolve actual step integer (handles step=None -> last-EMA)
    actual_step = resolve_step(ckpt_path, step)
    ckpt_name = ckpt_path.name

    print(f"=== run={run_name} step={actual_step} config={config_name} ===")
    print(f"    ckpt: {ckpt_path}")

    # Auto-derive --metrics from the run-name protocol suffix (paper-protocol
    # sweeps). The `_fid` suffix selects fid metrics only; `_des` selects
    # designability + diversity. Unsuffixed run names fall through to the
    # caller's value (legacy sweeps run everything by default).
    if run_name.endswith("_fid"):
        metrics_eff = "fid"
    elif run_name.endswith("_des"):
        metrics_eff = "designability,diversity"
    else:
        metrics_eff = metrics

    # Build the Namespace evaluate.main() consumes. Mirrors evaluate.parse_args
    # plus the shared metric flags from add_metric_args. None values trigger
    # evaluate's resolve_evaluate_defaults so the standalone defaults stay
    # authoritative for evaluate.py.
    eval_args = argparse.Namespace(
        config_name=config_name,
        config_subdir=None,
        ckpt_name_override=ckpt_name,
        ckpt_path_override=str(ckpt_path.parent),
        output_suffix=f"sweep_{run_name}_step_{actual_step}",
        skip_generation=skip_generation,
        novelty_tm_threshold=0.5,
        seed=seed,
        designability_subset_per_length=designability_subset_per_length,
        designability_lengths=designability_lengths,
        cath_subset=cath_subset,
        cath_head_path=cath_head_path,
        centroid_path=centroid_path,
        centroid_filter_designable=centroid_filter_designable,
        foldseek_target_dbs=foldseek_target_dbs,
        foldseek_alignment_type=foldseek_alignment_type,
        foldseek_max_seqs=foldseek_max_seqs,
        foldseek_sensitivity=foldseek_sensitivity,
        foldseek_threads=foldseek_threads,
        foldseek_filter_designable=foldseek_filter_designable,
        skip_fid=skip_fid,
        fast_inference=fast_inference,
        metrics=metrics_eff,
        ss_reference_pdb_path=ss_reference_pdb_path,
        ss_reference_afdb_path=ss_reference_afdb_path,
    )

    try:
        result_row = evaluate.main(args=eval_args)
        row: Dict = {
            "run": run_name,
            "step": actual_step,
            "config_name": config_name,
            "ckpt_path": str(ckpt_path),
            "seed": seed,
        }
        if isinstance(result_row, dict):
            row.update(
                {
                    k: v
                    for k, v in result_row.items()
                    if str(k).startswith(METRIC_PREFIX)
                }
            )
        _append_row(jsonl_path, row)
        print(f"    -> appended to {jsonl_path}")
        return True
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        # Loud, greppable marker so log scans / sacct sweepers find failures.
        print(f"!!! TASK_FAILED run={run_name} step={actual_step}: {msg}", flush=True)
        import traceback

        traceback.print_exc()
        _append_row(
            jsonl_path,
            {
                "run": run_name,
                "step": actual_step,
                "config_name": config_name,
                "error": msg,
            },
        )
        return False


# -- CLI -----------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-checkpoint generation sweep (mirrors representation/scripts/run_sweep.py)."
    )
    # Profile / run selection
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Named profile from sweep_config.yaml (e.g. 'n128', 'n512_convergence'). "
        "Loads canonical defaults; any other flag overrides individual fields.",
    )
    p.add_argument(
        "--runs",
        type=str,
        default=None,
        help="Comma-separated subset of run names to evaluate "
        "(e.g. 'baseline,repa_l4'). Overrides profile's runs field.",
    )

    # SLURM array task selection
    p.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="SLURM_ARRAY_TASK_ID. If set, runs only that index in the "
        "flattened task list. Use --dry_run to print the full index table.",
    )

    # Ad-hoc / backfill override (bypasses RUN_SCHEDULES)
    p.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Explicit checkpoint path for ad-hoc evaluation (bypasses RUN_SCHEDULES).",
    )
    p.add_argument(
        "--ckpt_label",
        type=str,
        default=None,
        help="Label for ad-hoc checkpoint (used as run name in JSONL). "
        "Required when --ckpt_path is set.",
    )
    p.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Hydra config name for ad-hoc checkpoint "
        "(e.g. 'inference/inference_fid_60m_baseline_lite'). "
        "Required when --ckpt_path is set.",
    )

    # Output
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for sweep_results.jsonl/csv/md. "
        "Defaults to profile's output_dir field (relative to generation/).",
    )

    # Metric settings (all overridable; defaults come from sweep profile).
    # The exact set of flags is shared with evaluate.py via add_metric_args
    # so the two scripts stay in sync.
    add_metric_args(p)

    # Arbitrary checkpoint list (TSV: label\tckpt_path\tconfig_name per row).
    # Bypasses RUN_SCHEDULES; one task per row. Mutually exclusive with
    # --config / --runs / --ckpt_path.
    p.add_argument(
        "--ckpts_file",
        type=str,
        default=None,
        help="Path to TSV file with one row per checkpoint to evaluate. "
        "Each non-comment line: 'label<TAB>ckpt_path<TAB>config_name'. Lines "
        "starting with # are ignored. Use with --array=0-N-1 to fan out one "
        "task per row.",
    )

    # Metric-only resume: reuse cached PDBs in eval_output, skip the
    # generation loop. Maps directly to evaluate.py's --skip_generation.
    p.add_argument(
        "--skip_generation",
        action="store_true",
        help="Reuse cached PDBs under eval_output/<output_suffix>/ instead of "
        "regenerating. Use to retry metric-only after an earlier task crashed "
        "during designability/diversity. Requires the cache to already hold "
        "the full PDB pool (1125 for paper protocols).",
    )

    # Utility modes
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the full task index table and exit without running.",
    )
    p.add_argument(
        "--consolidate_only",
        action="store_true",
        help="Rebuild CSV/MD from existing JSONL without generating.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # -- Load profile defaults, apply CLI overrides -------------------------- #
    profile_cfg: Dict = {}
    if args.config:
        profile_cfg = _load_sweep_config(args.config)

    seed = args.seed if args.seed is not None else int(profile_cfg.get("seed", 42))
    designability_subset_per_length = (
        args.designability_subset_per_length
        if args.designability_subset_per_length is not None
        else int(profile_cfg.get("designability_subset_per_length", 0))
    )
    designability_lengths = (
        args.designability_lengths
        if args.designability_lengths is not None
        else profile_cfg.get("designability_lengths")
    )
    skip_fid = (
        args.skip_fid
        if args.skip_fid is not None
        else bool(profile_cfg.get("skip_fid", False))
    )
    fast_inference = args.fast_inference if args.fast_inference is not None else True
    cath_subset = (
        args.cath_subset
        if args.cath_subset is not None
        else int(profile_cfg.get("cath_subset", 0))
    )
    cath_head_path = args.cath_head_path or profile_cfg.get("cath_head_path")
    centroid_path = args.centroid_path or profile_cfg.get("centroid_path")
    centroid_filter_designable = (
        args.centroid_filter_designable
        if args.centroid_filter_designable is not None
        else bool(profile_cfg.get("centroid_filter_designable", True))
    )
    metrics = args.metrics if args.metrics is not None else profile_cfg.get("metrics")
    ss_reference_pdb_path = args.ss_reference_pdb_path or profile_cfg.get(
        "ss_reference_pdb_path"
    )
    ss_reference_afdb_path = args.ss_reference_afdb_path or profile_cfg.get(
        "ss_reference_afdb_path"
    )

    # Foldseek knobs: CLI > profile > script default. Empty target_dbs -> skip.
    foldseek_target_dbs = (
        args.foldseek_target_dbs
        if args.foldseek_target_dbs is not None
        else profile_cfg.get("foldseek_target_dbs")
    )
    foldseek_alignment_type = (
        args.foldseek_alignment_type
        if args.foldseek_alignment_type is not None
        else int(profile_cfg.get("foldseek_alignment_type", 2))
    )
    foldseek_max_seqs = (
        args.foldseek_max_seqs
        if args.foldseek_max_seqs is not None
        else int(profile_cfg.get("foldseek_max_seqs", 1000))
    )
    foldseek_sensitivity = (
        args.foldseek_sensitivity
        if args.foldseek_sensitivity is not None
        else float(profile_cfg.get("foldseek_sensitivity", 9.5))
    )
    foldseek_threads = (
        args.foldseek_threads
        if args.foldseek_threads is not None
        else int(profile_cfg.get("foldseek_threads", 0))
    )
    foldseek_filter_designable = (
        args.foldseek_filter_designable
        if args.foldseek_filter_designable is not None
        else bool(profile_cfg.get("foldseek_filter_designable", True))
    )

    # -- Resolve output dir -------------------------------------------------- #
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif "output_dir" in profile_cfg:
        output_dir = HERE.parent / profile_cfg["output_dir"]
    else:
        output_dir = HERE.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "sweep_results.jsonl"

    # -- Consolidate-only mode ----------------------------------------------- #
    if args.consolidate_only:
        consolidate(jsonl_path, output_dir)
        return

    # -- Mutually exclusive task sources ------------------------------------- #
    sources = sum(
        bool(x)
        for x in (args.ckpt_path, args.ckpts_file, args.runs or profile_cfg.get("runs"))
    )
    if sources > 1 and not args.ckpt_path:  # ckpt_path handled below standalone
        # Allow --runs override of profile, but disallow combining with --ckpts_file
        if args.ckpts_file and (args.runs or profile_cfg.get("runs")):
            raise ValueError(
                "--ckpts_file is mutually exclusive with --runs / profile runs."
            )

    # -- Ad-hoc single checkpoint -------------------------------------------- #
    if args.ckpt_path:
        if args.ckpts_file:
            raise ValueError("--ckpt_path and --ckpts_file are mutually exclusive.")
        if not args.ckpt_label or not args.config_name:
            raise ValueError("--ckpt_path requires --ckpt_label and --config_name")
        ckpt_path = Path(args.ckpt_path)
        actual_step = resolve_step(ckpt_path, None)
        done, _ = _load_done_set(jsonl_path)
        if (args.ckpt_label, actual_step) in done:
            print(f"Already done: run={args.ckpt_label} step={actual_step}, skipping.")
        else:
            run_one_task(
                run_name=args.ckpt_label,
                step=None,
                config_name=args.config_name,
                ckpt_path=ckpt_path,
                output_dir=output_dir,
                seed=seed,
                designability_subset_per_length=designability_subset_per_length,
                designability_lengths=designability_lengths,
                skip_fid=skip_fid,
                fast_inference=fast_inference,
                jsonl_path=jsonl_path,
                cath_subset=cath_subset,
                cath_head_path=cath_head_path,
                centroid_path=centroid_path,
                centroid_filter_designable=centroid_filter_designable,
                foldseek_target_dbs=foldseek_target_dbs,
                foldseek_alignment_type=foldseek_alignment_type,
                foldseek_max_seqs=foldseek_max_seqs,
                foldseek_sensitivity=foldseek_sensitivity,
                foldseek_threads=foldseek_threads,
                foldseek_filter_designable=foldseek_filter_designable,
                metrics=metrics,
                skip_generation=args.skip_generation,
                ss_reference_pdb_path=ss_reference_pdb_path,
                ss_reference_afdb_path=ss_reference_afdb_path,
            )
        consolidate(jsonl_path, output_dir)
        return

    # -- Build task list (file-driven OR RUN_SCHEDULES-driven) -------------- #
    if args.ckpts_file:
        tasks = load_ckpts_file(Path(args.ckpts_file))
    else:
        runs_str = args.runs or profile_cfg.get("runs", "")
        if not runs_str:
            raise ValueError(
                "Specify one of --ckpts_file, --ckpt_path, --runs, or --config "
                "with a runs field."
            )
        runs = [r.strip() for r in runs_str.split(",") if r.strip()]
        tasks = build_task_list(runs)

    # -- Dry run ------------------------------------------------------------- #
    if args.dry_run:
        print(f"Task list ({len(tasks)} tasks):")
        print(f"  {'idx':>4}  {'run':<20}  {'step':>8}  {'config_name'}")
        for i, (run_name, step, config_name, ckpt_path) in enumerate(tasks):
            ckpt_exists = "OK" if (ckpt_path and ckpt_path.exists()) else "MISSING"
            print(
                f"  {i:>4}  {run_name:<20}  {str(step):>8}  {config_name}  [{ckpt_exists}]"
            )
        if args.ckpts_file:
            hint = f"--ckpts_file {args.ckpts_file}"
        else:
            hint = f"--config {args.config or '<profile>'}"
        print(f"\nSubmit with: sbatch --array=0-{len(tasks)-1} run_sweep.sh {hint}")
        return

    # -- Select task(s) to run ----------------------------------------------- #
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))

    if task_id >= 0:
        if task_id >= len(tasks):
            print(f"ERROR: task_id={task_id} out of range (total tasks: {len(tasks)})")
            sys.exit(1)
        tasks_to_run = [tasks[task_id]]
    else:
        # No task_id - run all tasks sequentially (useful for local testing)
        tasks_to_run = tasks

    # -- Load done set ------------------------------------------------------- #
    done, _ = _load_done_set(jsonl_path)

    # -- Execute ------------------------------------------------------------- #
    failures = []
    for run_name, step, config_name, ckpt_path in tasks_to_run:
        # Resolve actual step for done-set check (handles step=None)
        if ckpt_path and ckpt_path.exists():
            actual_step = resolve_step(ckpt_path, step)
        else:
            actual_step = step or -1

        if (run_name, actual_step) in done:
            print(f"Already done: run={run_name} step={actual_step}, skipping.")
            continue

        ok = run_one_task(
            run_name=run_name,
            step=step,
            config_name=config_name,
            ckpt_path=ckpt_path,
            output_dir=output_dir,
            seed=seed,
            designability_subset_per_length=designability_subset_per_length,
            designability_lengths=designability_lengths,
            skip_fid=skip_fid,
            fast_inference=fast_inference,
            jsonl_path=jsonl_path,
            cath_subset=cath_subset,
            cath_head_path=cath_head_path,
            centroid_path=centroid_path,
            centroid_filter_designable=centroid_filter_designable,
            foldseek_target_dbs=foldseek_target_dbs,
            foldseek_alignment_type=foldseek_alignment_type,
            foldseek_max_seqs=foldseek_max_seqs,
            foldseek_sensitivity=foldseek_sensitivity,
            foldseek_threads=foldseek_threads,
            foldseek_filter_designable=foldseek_filter_designable,
            metrics=metrics,
            skip_generation=args.skip_generation,
            ss_reference_pdb_path=ss_reference_pdb_path,
            ss_reference_afdb_path=ss_reference_afdb_path,
        )
        if not ok:
            failures.append((run_name, actual_step))

    consolidate(jsonl_path, output_dir)

    if failures:
        print(
            f"\n!!! SWEEP_FAILED: {len(failures)}/{len(tasks_to_run)} task(s) errored:",
            flush=True,
        )
        for r, s in failures:
            print(f"    - run={r} step={s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
