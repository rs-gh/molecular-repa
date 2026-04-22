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
    # Dry run — print task index table and exit
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
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
# HERE        = .../generation/scripts
# HERE.parent = .../generation
# HERE.parent.parent = .../proteina  (evaluation/proteina)
sys.path.insert(
    0, str(HERE.parent.parent)
)  # -> evaluation/proteina/ for lib.checkpoints

from lib.checkpoints import (  # noqa: E402
    GEN_RUN_CONFIGS,
    RUN_SCHEDULES,
    find_checkpoint_path,
    resolve_step,
)

SWEEP_CONFIG_PATH = HERE.parent / "sweep_config.yaml"


# ── Sweep config loading ──────────────────────────────────────────────────────


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


# ── Task list construction ────────────────────────────────────────────────────


def build_task_list(
    runs: List[str],
) -> List[Tuple[str, Optional[int], str, Path]]:
    """Return a flat list of (run_name, step, config_name, ckpt_path) tuples.

    Expands each run's step_list from RUN_SCHEDULES.  Tasks are ordered
    run-first then step-ascending so a partial --array covers early steps of
    all runs before late steps of any run.
    """
    tasks = []
    for run_name in runs:
        if run_name not in RUN_SCHEDULES:
            raise ValueError(
                f"Unknown run '{run_name}'. Available: {sorted(RUN_SCHEDULES)}"
            )
        if run_name not in GEN_RUN_CONFIGS:
            raise ValueError(
                f"Run '{run_name}' has no GEN_RUN_CONFIGS entry. "
                f"Add it to evaluation/proteina/lib/checkpoints.py."
            )
        run_dir, _is_repa, _layer, step_list = RUN_SCHEDULES[run_name]
        config_name = GEN_RUN_CONFIGS[run_name]
        for step in step_list:
            ckpt_path = find_checkpoint_path(run_dir, step)
            tasks.append((run_name, step, config_name, ckpt_path))
    return tasks


# ── JSONL resume helpers ──────────────────────────────────────────────────────


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


# ── Result consolidation ──────────────────────────────────────────────────────

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
    "_res_novelty_rate",
]


def consolidate(jsonl_path: Path, output_dir: Path) -> None:
    """Rebuild sweep_results.csv and sweep_results.md from the JSONL."""
    _, rows = _load_done_set(jsonl_path)
    if not rows:
        print("No rows in JSONL yet, skipping consolidation.")
        return

    # Write full JSON list
    json_path = output_dir / "sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    # Write CSV
    all_keys = []
    seen = set()
    base_cols = ["run", "step", "config_name", "ckpt_path", "seed", "error"]
    for k in base_cols:
        if k not in seen:
            all_keys.append(k)
            seen.add(k)
    for r in rows:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    csv_path = output_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write MD summary — one row per (run, step), key metrics only
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

    print(f"Consolidated {len(result_rows)} rows -> {csv_path}, {md_path}")


# ── Single-task execution ─────────────────────────────────────────────────────


def run_one_task(
    run_name: str,
    step: Optional[int],
    config_name: str,
    ckpt_path: Optional[Path],
    output_dir: Path,
    seed: int,
    designability_subset: int,
    diversity_subset_per_bin: int,
    skip_fid: bool,
    fast_inference: bool,
    jsonl_path: Path,
) -> None:
    """Run evaluate.py for one (run_name, step) and append to JSONL."""
    import importlib

    if ckpt_path is None or not ckpt_path.exists():
        msg = f"Checkpoint not found for run={run_name} step={step}: {ckpt_path}"
        print(f"ERROR: {msg}")
        _append_row(jsonl_path, {"run": run_name, "step": step, "error": msg})
        return

    # Resolve actual step integer (handles step=None -> last-EMA)
    from lib.checkpoints import resolve_step as _resolve_step

    actual_step = _resolve_step(ckpt_path, step)
    ckpt_name = ckpt_path.name
    ckpt_dir = str(ckpt_path.parent)

    print(f"=== run={run_name} step={actual_step} config={config_name} ===")
    print(f"    ckpt: {ckpt_path}")

    # Build sys.argv for evaluate.py's parse_args()
    argv = [
        "evaluate.py",
        "--config_name",
        config_name,
        "--ckpt_name_override",
        ckpt_name,
        "--output_suffix",
        f"sweep_{run_name}_step_{actual_step}",
        "--seed",
        str(seed),
        "--designability_subset",
        str(designability_subset),
        "--diversity_subset_per_bin",
        str(diversity_subset_per_bin),
    ]
    if skip_fid:
        argv.append("--skip_fid")
    if not fast_inference:
        argv.append("--no-fast_inference")

    # Patch cfg.ckpt_path via env so evaluate.py uses our resolved path
    os.environ["_GEN_SWEEP_CKPT_DIR_OVERRIDE"] = ckpt_dir

    saved_argv = sys.argv[:]
    sys.argv = argv
    try:
        evaluate = importlib.import_module("evaluate")
        importlib.reload(evaluate)  # ensure fresh state if called multiple times
        evaluate.main()
        # Read back the per-checkpoint CSV to get metric values
        output_suffix = f"sweep_{run_name}_step_{actual_step}"
        results_csv = (
            HERE.parent.parent.parent.parent  # repo root
            / f"eval_output/{config_name}_{output_suffix}"
            / f"results_{config_name}_{output_suffix}_fid.csv"
        )
        row: Dict = {
            "run": run_name,
            "step": actual_step,
            "config_name": config_name,
            "ckpt_path": str(ckpt_path),
            "seed": seed,
        }
        if results_csv.exists():
            import pandas as pd

            df = pd.read_csv(results_csv)
            if len(df) > 0:
                metric_vals = {
                    k: v for k, v in df.iloc[0].items() if str(k).startswith("_res_")
                }
                row.update(metric_vals)
        _append_row(jsonl_path, row)
        print(f"    -> appended to {jsonl_path}")
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        print(f"ERROR in run={run_name} step={actual_step}: {msg}")
        _append_row(
            jsonl_path,
            {
                "run": run_name,
                "step": actual_step,
                "config_name": config_name,
                "error": msg,
            },
        )
    finally:
        sys.argv = saved_argv
        os.environ.pop("_GEN_SWEEP_CKPT_DIR_OVERRIDE", None)


# ── CLI ───────────────────────────────────────────────────────────────────────


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

    # Metric settings (all overridable; defaults come from sweep profile)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--designability_subset",
        type=int,
        default=None,
        help="PDBs to eval for designability (0=skip).",
    )
    p.add_argument("--diversity_subset_per_bin", type=int, default=None)
    p.add_argument(
        "--skip_fid",
        action="store_true",
        default=None,
        help="Skip GearNet FID/fJSD/fS metrics.",
    )
    p.add_argument(
        "--fast_inference", action=argparse.BooleanOptionalAction, default=True
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

    # ── Load profile defaults, apply CLI overrides ────────────────────────── #
    profile_cfg: Dict = {}
    if args.config:
        profile_cfg = _load_sweep_config(args.config)

    seed = args.seed if args.seed is not None else int(profile_cfg.get("seed", 42))
    designability_subset = (
        args.designability_subset
        if args.designability_subset is not None
        else int(profile_cfg.get("designability_subset", 0))
    )
    diversity_subset_per_bin = (
        args.diversity_subset_per_bin
        if args.diversity_subset_per_bin is not None
        else int(profile_cfg.get("diversity_subset_per_bin", 0))
    )
    skip_fid = (
        args.skip_fid
        if args.skip_fid is not None
        else bool(profile_cfg.get("skip_fid", False))
    )
    fast_inference = args.fast_inference

    # ── Resolve output dir ────────────────────────────────────────────────── #
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif "output_dir" in profile_cfg:
        output_dir = HERE.parent / profile_cfg["output_dir"]
    else:
        output_dir = HERE.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "sweep_results.jsonl"

    # ── Consolidate-only mode ─────────────────────────────────────────────── #
    if args.consolidate_only:
        consolidate(jsonl_path, output_dir)
        return

    # ── Ad-hoc single checkpoint ──────────────────────────────────────────── #
    if args.ckpt_path:
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
                designability_subset=designability_subset,
                diversity_subset_per_bin=diversity_subset_per_bin,
                skip_fid=skip_fid,
                fast_inference=fast_inference,
                jsonl_path=jsonl_path,
            )
        consolidate(jsonl_path, output_dir)
        return

    # ── Build task list from RUN_SCHEDULES ───────────────────────────────── #
    runs_str = args.runs or profile_cfg.get("runs", "")
    if not runs_str:
        raise ValueError("Specify --runs or --config with a runs field.")
    runs = [r.strip() for r in runs_str.split(",") if r.strip()]
    tasks = build_task_list(runs)

    # ── Dry run ───────────────────────────────────────────────────────────── #
    if args.dry_run:
        print(f"Task list ({len(tasks)} tasks):")
        print(f"  {'idx':>4}  {'run':<20}  {'step':>8}  {'config_name'}")
        for i, (run_name, step, config_name, ckpt_path) in enumerate(tasks):
            ckpt_exists = "OK" if (ckpt_path and ckpt_path.exists()) else "MISSING"
            print(
                f"  {i:>4}  {run_name:<20}  {str(step):>8}  {config_name}  [{ckpt_exists}]"
            )
        print(
            f"\nSubmit with: sbatch --array=0-{len(tasks)-1} run_sweep.sh --config {args.config or '<profile>'}"
        )
        return

    # ── Select task(s) to run ─────────────────────────────────────────────── #
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))

    if task_id >= 0:
        if task_id >= len(tasks):
            print(f"ERROR: task_id={task_id} out of range (total tasks: {len(tasks)})")
            sys.exit(1)
        tasks_to_run = [tasks[task_id]]
    else:
        # No task_id — run all tasks sequentially (useful for local testing)
        tasks_to_run = tasks

    # ── Load done set ─────────────────────────────────────────────────────── #
    done, _ = _load_done_set(jsonl_path)

    # ── Execute ───────────────────────────────────────────────────────────── #
    for run_name, step, config_name, ckpt_path in tasks_to_run:
        # Resolve actual step for done-set check (handles step=None)
        if ckpt_path and ckpt_path.exists():
            actual_step = resolve_step(ckpt_path, step)
        else:
            actual_step = step or -1

        if (run_name, actual_step) in done:
            print(f"Already done: run={run_name} step={actual_step}, skipping.")
            continue

        run_one_task(
            run_name=run_name,
            step=step,
            config_name=config_name,
            ckpt_path=ckpt_path,
            output_dir=output_dir,
            seed=seed,
            designability_subset=designability_subset,
            diversity_subset_per_bin=diversity_subset_per_bin,
            skip_fid=skip_fid,
            fast_inference=fast_inference,
            jsonl_path=jsonl_path,
        )

    consolidate(jsonl_path, output_dir)


if __name__ == "__main__":
    main()
