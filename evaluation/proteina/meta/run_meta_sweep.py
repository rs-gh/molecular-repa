"""Meta-evaluation sweep: variance + FID-scaling experiments on the gen pipeline.

Thin wrapper over ``generation/scripts/run_sweep.run_one_task`` that fans out
``(target_run, seed)`` pairs as independent tasks. Each task gets a synthesized
``ckpt_label = "<target>__seed<S>"`` so the ``output_suffix`` differs per rep
and the PDB pool is regenerated fresh (the standard sweep reuses cached PDBs
whenever output_suffix matches, which kills variance estimation).

Profiles live in ``evaluation/proteina/meta/sweep_config.yaml``.

Usage:
    # Dry run — print the (target, seed) task index table and exit
    python evaluation/proteina/meta/run_meta_sweep.py --config sanity_seed_n128 --dry_run

    # SLURM array, one task per (target, seed)
    sbatch --array=0-0  hpc-scripts/proteina/evaluation/meta/run_meta_sweep.sh --config sanity_seed_n128
    sbatch --array=0-0  hpc-scripts/proteina/evaluation/meta/run_meta_sweep.sh --config fid_scaling_n256
    sbatch --array=0-9  hpc-scripts/proteina/evaluation/meta/run_meta_sweep.sh --config variance_n128_layer
    sbatch --array=0-4  hpc-scripts/proteina/evaluation/meta/run_meta_sweep.sh --config variance_n256_layer

    # Rebuild MD/CSV from existing JSONL without rerunning
    python evaluation/proteina/meta/run_meta_sweep.py --config variance_n128_layer --consolidate_only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
EVAL_PROTEINA = HERE.parent
sys.path.insert(0, str(EVAL_PROTEINA / "generation" / "scripts"))
sys.path.insert(0, str(EVAL_PROTEINA))

from lib.checkpoints import (  # noqa: E402
    GEN_RUN_CONFIGS,
    RUN_SCHEDULES,
    find_checkpoint_path,
    resolve_step,
)
from run_sweep import (  # noqa: E402
    _load_done_set,
    consolidate,
)

SWEEP_CONFIG_PATH = HERE / "sweep_config.yaml"


# -- Profile loading -----------------------------------------------------------


def _load_profile(name: str) -> Dict:
    import yaml

    with open(SWEEP_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    if name not in raw:
        keys = [k for k in raw if not k.startswith("_")]
        raise KeyError(
            f"Profile '{name}' not in {SWEEP_CONFIG_PATH}. Available: {keys}"
        )
    return raw[name]


# -- Task expansion ------------------------------------------------------------


def _resolve_ckpt(target: str) -> Tuple[Path, int, str]:
    """Return (ckpt_path, step, config_name) for a RUN_SCHEDULES target.

    Uses the schedule's last step entry (most schedules pin a single step for
    paper protocols; multi-step schedules are not meaningful here since we
    iterate over seeds, not over training time).
    """
    if target not in RUN_SCHEDULES:
        raise KeyError(f"target '{target}' not in RUN_SCHEDULES")
    if target not in GEN_RUN_CONFIGS:
        raise KeyError(f"target '{target}' has no GEN_RUN_CONFIGS entry")
    run_dir, _is_repa, _layer, step_list = RUN_SCHEDULES[target]
    step = step_list[-1] if step_list else None
    ckpt_path = find_checkpoint_path(run_dir, step)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found for target={target} step={step} run_dir={run_dir}"
        )
    return ckpt_path, resolve_step(ckpt_path, step), GEN_RUN_CONFIGS[target]


def _build_tasks(profile: Dict) -> List[Dict]:
    """Flatten profile into one dict-per-task. Order: target-first, then seed."""
    targets = [t.strip() for t in str(profile["targets"]).split(",") if t.strip()]
    seeds = [int(s) for s in str(profile["seeds"]).split(",") if s.strip()]
    cfg_override = profile.get("config_name_override")
    tasks: List[Dict] = []
    for target in targets:
        ckpt_path, step, default_cfg = _resolve_ckpt(target)
        config_name = cfg_override or default_cfg
        for seed in seeds:
            tasks.append(
                {
                    "target": target,
                    "seed": seed,
                    "ckpt_label": f"{target}__seed{seed}",
                    "ckpt_path": ckpt_path,
                    "step": step,
                    "config_name": config_name,
                }
            )
    return tasks


# -- Execution -----------------------------------------------------------------


def _run_task(task: Dict, profile: Dict, jsonl_path: Path, output_dir: Path) -> None:
    """Invoke run_sweep.run_one_task with the per-rep ckpt_label + seed."""
    # Lazy import: run_one_task pulls proteinfoundation, only needed for actual runs.
    from run_sweep import run_one_task  # noqa: PLC0415

    run_one_task(
        run_name=task["ckpt_label"],
        step=None,  # forces resolve_step to read global_step from ckpt
        config_name=task["config_name"],
        ckpt_path=task["ckpt_path"],
        output_dir=output_dir,
        seed=task["seed"],
        designability_subset_per_length=int(
            profile.get("designability_subset_per_length", 0)
        ),
        designability_lengths=profile.get("designability_lengths"),
        skip_fid=bool(profile.get("skip_fid", False)),
        fast_inference=bool(profile.get("fast_inference", True)),
        jsonl_path=jsonl_path,
        cath_subset=int(profile.get("cath_subset", 0)),
        cath_head_path=profile.get("cath_head_path"),
        centroid_path=profile.get("centroid_path"),
        centroid_filter_designable=bool(
            profile.get("centroid_filter_designable", True)
        ),
        foldseek_target_dbs=profile.get("foldseek_target_dbs"),
        foldseek_alignment_type=int(profile.get("foldseek_alignment_type", 2)),
        foldseek_max_seqs=int(profile.get("foldseek_max_seqs", 1000)),
        foldseek_sensitivity=float(profile.get("foldseek_sensitivity", 9.5)),
        foldseek_threads=int(profile.get("foldseek_threads", 0)),
        foldseek_filter_designable=bool(
            profile.get("foldseek_filter_designable", True)
        ),
        metrics=profile.get("metrics"),
        skip_generation=False,
    )


# -- CLI -----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--config", required=True, help="Profile name from sweep_config.yaml"
    )
    p.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Override SLURM_ARRAY_TASK_ID. Runs one (target, seed) pair from the "
        "flattened task list. Omit to run all tasks sequentially.",
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--consolidate_only", action="store_true")
    p.add_argument(
        "--output_dir",
        default=None,
        help="Override profile's output_dir (relative to evaluation/proteina/generation/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    profile = _load_profile(args.config)

    # Resolve output dir relative to evaluation/proteina/generation/ to keep
    # results next to the standard sweep's results/ tree.
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif "output_dir" in profile:
        output_dir = EVAL_PROTEINA / "generation" / profile["output_dir"]
    else:
        output_dir = EVAL_PROTEINA / "generation" / "results" / "meta" / args.config
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "sweep_results.jsonl"

    if args.consolidate_only:
        consolidate(jsonl_path, output_dir)
        return

    tasks = _build_tasks(profile)

    if args.dry_run:
        print(f"Profile '{args.config}': {len(tasks)} tasks")
        print(f"  output_dir: {output_dir}")
        print(f"  {'idx':>4}  {'target':<40}  {'seed':>4}  {'step':>8}  config_name")
        for i, t in enumerate(tasks):
            print(
                f"  {i:>4}  {t['target']:<40}  {t['seed']:>4}  "
                f"{t['step']:>8}  {t['config_name']}"
            )
        print(
            f"\nSubmit with: sbatch --array=0-{len(tasks) - 1} "
            f"hpc-scripts/proteina/evaluation/meta/run_meta_sweep.sh --config {args.config}"
        )
        return

    # Select task subset
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
    if task_id >= 0:
        if task_id >= len(tasks):
            print(f"ERROR: task_id={task_id} >= {len(tasks)}")
            sys.exit(1)
        tasks_to_run = [tasks[task_id]]
    else:
        tasks_to_run = tasks

    # Resume: skip (ckpt_label, step) pairs already in JSONL
    done, _ = _load_done_set(jsonl_path)
    for t in tasks_to_run:
        if (t["ckpt_label"], t["step"]) in done:
            print(f"Already done: {t['ckpt_label']} step={t['step']}, skipping.")
            continue
        print(
            f"=== target={t['target']} seed={t['seed']} step={t['step']} "
            f"config={t['config_name']} ==="
        )
        _run_task(t, profile, jsonl_path, output_dir)

    consolidate(jsonl_path, output_dir)


if __name__ == "__main__":
    main()
