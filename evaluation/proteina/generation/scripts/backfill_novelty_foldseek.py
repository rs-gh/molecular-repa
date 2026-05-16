"""Backfill `_res_novelty_foldseek_*` columns into the paper tables.

What this does
==============

1. Loads every row from `results/paper/n*_paper_*/sweep_results.jsonl`.
2. Resolves the eval_output dir for each row using the same slug pattern
   `run_sweep.py` uses: `{config_name.replace('/','_')}_sweep_{run}_step_{step}`.
3. Buckets each row by what artifacts are still on disk:
     - usable    : samples_fid populated + designability_index.csv present
     - cached    : eval_output CSV already has both DB foldseek columns
     - pruned    : samples_fid empty (PDBs were cleaned up)
4. For `cached` rows: copy the columns from the per-ckpt CSV into the jsonl row
   (idempotent sync, no foldseek invocation).
5. For `usable` rows: read the designable subset from `designability_index.csv`,
   call `compute_novelty_foldseek` against the pdb + afdb_swissprot DBs, merge
   the resulting columns into both the per-ckpt `results_*_fid.csv` and the
   jsonl row.
6. For `pruned` rows: search for a sibling eval_output dir with the same
   (run, step) but a different config slug, and if exactly one match has
   samples populated, copy the foldseek columns across. (Same checkpoint,
   same generation seed -> same PDBs; safe to share novelty columns.)
7. Writes a JSON report summarising what got filled, what didn't, and why.

Run with `--dry-run` first to see the bucket counts without touching anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from utils.novelty_foldseek import compute_novelty_foldseek  # noqa: E402
from utils._metric_args import METRIC_PREFIX  # noqa: E402

REPO_DIR = HERE.parent.parent.parent.parent
PAPER_RESULTS_ROOT = REPO_DIR / "evaluation/proteina/generation/results/paper"
EVAL_OUTPUT_ROOT = REPO_DIR / "eval_output"
# /rds counterpart with the canonical PDBs for the runs that were pruned
# from /home (Phase A, 2026-04-21). The local /home dir often has an empty
# samples_fid for these runs; the /rds version still has the artefacts.
EVAL_OUTPUT_ROOT_FALLBACK = Path("/rds/user/sr2173/hpc-work/proteina/eval_output")

DBS = [
    ("pdb", "/rds/user/sr2173/hpc-work/proteina/foldseek_dbs/pdb/db"),
    (
        "afdb_swissprot",
        "/rds/user/sr2173/hpc-work/proteina/foldseek_dbs/afdb_swissprot/db",
    ),
]

# Column names we expect to find after a successful foldseek run; presence of
# both `_max_tm_mean` columns is the idempotency check.
FOLDSEEK_SENTINELS = [f"_res_novelty_foldseek_{label}_max_tm_mean" for label, _ in DBS]


def eval_dir_for_row(row: dict) -> Path:
    """Resolve eval dir, preferring /home but falling back to /rds when /home's
    `samples_fid` is empty (Phase A pruning left empty placeholder dirs).
    """
    slug = row["config_name"].replace("/", "_")
    suff = f"sweep_{row['run']}_step_{row['step']}"
    primary = EVAL_OUTPUT_ROOT / f"{slug}_{suff}"
    fallback = EVAL_OUTPUT_ROOT_FALLBACK / f"{slug}_{suff}"
    samples_primary = primary / "samples_fid"
    primary_has_pdbs = samples_primary.is_dir() and any(
        p.suffix == ".pdb" for p in samples_primary.iterdir()
    )
    if primary_has_pdbs:
        return primary
    if fallback.is_dir():
        samples_fallback = fallback / "samples_fid"
        if samples_fallback.is_dir() and any(
            p.suffix == ".pdb" for p in samples_fallback.iterdir()
        ):
            return fallback
    return primary  # let classify() bucket it as 'pruned' if neither has PDBs


def find_results_csv(eval_dir: Path) -> Optional[Path]:
    matches = sorted(eval_dir.glob("results_*_fid.csv"))
    if len(matches) != 1:
        return None
    return matches[0]


def read_designable_queries(eval_dir: Path) -> list[str]:
    idx_csv = eval_dir / "designability_index.csv"
    df = pd.read_csv(idx_csv)
    sub = df[df["designable"].astype(str).str.lower() == "true"]
    return [str(eval_dir / rel) for rel in sub["pdb_path"].tolist()]


def _to_jsonable(v):
    """Coerce numpy scalars to native Python so json.dumps accepts them.
    Pandas hands back int64/float64/bool_/etc. when round-tripping through CSV.
    """
    import numpy as np

    if isinstance(v, np.generic):
        return v.item()
    return v


def has_all_foldseek_cols(d: dict) -> bool:
    return all(k in d and pd.notna(d[k]) for k in FOLDSEEK_SENTINELS)


@dataclass
class RowTask:
    profile: str
    jsonl_path: Path
    row_index: int
    row: dict
    eval_dir: Path
    bucket: str  # 'cached' | 'usable' | 'pruned' | 'no_dir'
    n_designable: int = 0
    fill_source: Optional[str] = None  # 'foldseek_run' | 'csv_sync' | 'sibling_copy'
    wall_seconds: float = 0.0
    columns_written: dict = field(default_factory=dict)
    error: Optional[str] = None


def classify(row: dict, profile: str, jsonl_path: Path, idx: int) -> RowTask:
    eval_dir = eval_dir_for_row(row)
    task = RowTask(
        profile=profile,
        jsonl_path=jsonl_path,
        row_index=idx,
        row=row,
        eval_dir=eval_dir,
        bucket="no_dir",
    )
    if not eval_dir.is_dir():
        return task
    csv_path = find_results_csv(eval_dir)
    if csv_path is not None:
        existing = pd.read_csv(csv_path)
        if (
            len(existing) == 1
            and all(c in existing.columns for c in FOLDSEEK_SENTINELS)
            and all(pd.notna(existing[c].iloc[0]) for c in FOLDSEEK_SENTINELS)
        ):
            task.bucket = "cached"
            return task
    samples_dir = eval_dir / "samples_fid"
    n_pdbs = (
        sum(1 for p in samples_dir.iterdir() if p.suffix == ".pdb")
        if samples_dir.is_dir()
        else 0
    )
    if n_pdbs == 0:
        task.bucket = "pruned"
        return task
    if not (eval_dir / "designability_index.csv").is_file():
        task.bucket = "pruned"  # treat like pruned — we can't run paper-protocol filter
        return task
    task.bucket = "usable"
    return task


def run_foldseek_on_task(task: RowTask, threads: int) -> None:
    queries = read_designable_queries(task.eval_dir)
    task.n_designable = len(queries)
    if not queries:
        task.error = "no designable PDBs"
        return
    t0 = time.time()
    cols: dict = {}
    for label, db_path in DBS:
        res = compute_novelty_foldseek(
            queries,
            target_db=db_path,
            db_label=label,
            alignment_type=1,
            max_seqs=1000,
            sensitivity=9.5,
            threads=threads,
        )
        cols.update(res)
    task.wall_seconds = time.time() - t0
    task.columns_written = cols
    task.fill_source = "foldseek_run"


def sync_from_csv(task: RowTask) -> None:
    csv_path = find_results_csv(task.eval_dir)
    if csv_path is None:
        task.error = "no results CSV to sync from"
        return
    df = pd.read_csv(csv_path)
    if len(df) != 1:
        task.error = f"unexpected CSV row count: {len(df)}"
        return
    cols = {
        c: _to_jsonable(df[c].iloc[0])
        for c in df.columns
        if c.startswith(f"{METRIC_PREFIX}novelty_foldseek_")
    }
    if not cols:
        task.error = "CSV has no foldseek columns"
        return
    task.columns_written = cols
    task.fill_source = "csv_sync"


def write_back(task: RowTask) -> None:
    """Persist task.columns_written into per-ckpt CSV and the jsonl row."""
    if not task.columns_written:
        return
    # Per-ckpt CSV merge (only for foldseek-run case; csv_sync already has them).
    if task.fill_source == "foldseek_run":
        csv_path = find_results_csv(task.eval_dir)
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            for k, v in task.columns_written.items():
                df[k] = v
            df.to_csv(csv_path, index=False)
    # jsonl row update — rewrite the whole file each time we touch it; cheap.
    lines = task.jsonl_path.read_text().splitlines()
    row = json.loads(lines[task.row_index])
    for k, v in task.columns_written.items():
        row[k] = v
    lines[task.row_index] = json.dumps(row)
    task.jsonl_path.write_text("\n".join(lines) + "\n")


def audit_pruned(pruned: list[RowTask], cached_or_done: list[RowTask]) -> dict:
    """For each pruned row, look for a sibling task with the same (run, step)
    whose foldseek columns are now known. Returns mapping pruned-key -> sibling key."""
    known: dict[tuple, dict] = {}
    for t in cached_or_done:
        if t.columns_written:
            known[(t.row["run"], t.row["step"])] = t.columns_written
        else:
            # `cached` rows don't fill task.columns_written; pull them now
            csv_path = find_results_csv(t.eval_dir)
            if csv_path is None:
                continue
            df = pd.read_csv(csv_path)
            cols = {
                c: _to_jsonable(df[c].iloc[0])
                for c in df.columns
                if c.startswith(f"{METRIC_PREFIX}novelty_foldseek_")
            }
            if cols:
                known[(t.row["run"], t.row["step"])] = cols
    matches = {}
    for p in pruned:
        key = (p.row["run"], p.row["step"])
        if key in known:
            p.columns_written = dict(known[key])
            p.fill_source = "sibling_copy"
            matches[f"{p.profile} :: {p.row['run']} step={p.row['step']}"] = key
    return matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and report; don't run foldseek or mutate files.",
    )
    ap.add_argument(
        "--shard",
        type=int,
        default=None,
        help="0-indexed shard. With --num-shards K, this task only "
        "processes usable[shard::K] foldseek runs and skips the "
        "cached/pruned finalize step.",
    )
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument(
        "--finalize",
        action="store_true",
        help="Skip foldseek runs; only sync cached rows + copy "
        "foldseek columns into pruned rows from filled siblings. "
        "Run after all foldseek shards have completed.",
    )
    ap.add_argument(
        "--report-path",
        type=str,
        default=str(
            REPO_DIR
            / "evaluation/proteina/generation/results/paper/_foldseek_backfill_report.json"
        ),
    )
    args = ap.parse_args()
    if (args.shard is None) != (args.num_shards is None):
        ap.error("--shard and --num-shards must be passed together")
    if args.shard is not None and args.finalize:
        ap.error("--shard and --finalize are mutually exclusive")

    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}")

    # Discover and classify every row.
    tasks: list[RowTask] = []
    for jsonl_path in sorted(PAPER_RESULTS_ROOT.glob("n*_paper_*/sweep_results.jsonl")):
        profile = jsonl_path.parent.name
        lines = jsonl_path.read_text().splitlines()
        for idx, ln in enumerate(lines):
            if not ln.strip():
                continue
            tasks.append(classify(json.loads(ln), profile, jsonl_path, idx))

    by_bucket: dict[str, list[RowTask]] = {
        "cached": [],
        "usable": [],
        "pruned": [],
        "no_dir": [],
    }
    for t in tasks:
        by_bucket[t.bucket].append(t)
    logger.info(f"Total rows: {len(tasks)}")
    for k, v in by_bucket.items():
        logger.info(f"  {k}: {len(v)}")

    if args.dry_run:
        logger.info("--dry-run set; exiting before foldseek invocations")
        return

    # Sharded mode: only run foldseek on this shard's slice; skip cached/pruned.
    # Pre-classification keeps the slice deterministic across shards (sorted
    # by (profile, row_index) — same ordering both shards see).
    usable_slice = by_bucket["usable"]
    if args.shard is not None:
        usable_slice = [
            t
            for i, t in enumerate(by_bucket["usable"])
            if i % args.num_shards == args.shard
        ]
        logger.info(
            f"Shard {args.shard}/{args.num_shards}: processing "
            f"{len(usable_slice)}/{len(by_bucket['usable'])} usable rows"
        )

    if not args.finalize:
        # 1) Process usable rows (run foldseek).
        for i, t in enumerate(usable_slice, 1):
            logger.info(
                f"[{i}/{len(usable_slice)}] foldseek: {t.profile} :: "
                f"{t.row['run']} step={t.row['step']}"
            )
            try:
                run_foldseek_on_task(t, threads=args.threads)
            except Exception as exc:
                t.error = f"{type(exc).__name__}: {exc}"
                logger.exception("foldseek failed")
                continue
            if t.columns_written:
                write_back(t)
                logger.info(
                    f"  done in {t.wall_seconds:.1f}s, n_designable={t.n_designable}"
                )

    # Cached sync + pruned sibling-copy run only in non-shard mode or --finalize.
    if args.shard is None:
        # 2) Sync cached rows into jsonl.
        for t in by_bucket["cached"]:
            try:
                sync_from_csv(t)
            except Exception as exc:
                t.error = f"{type(exc).__name__}: {exc}"
                continue
            if t.columns_written:
                write_back(t)
        logger.info(
            f"Synced {sum(1 for t in by_bucket['cached'] if t.fill_source == 'csv_sync')} cached rows"
        )

        # 3) Audit pruned rows for sibling matches.
        matches = audit_pruned(
            by_bucket["pruned"], by_bucket["usable"] + by_bucket["cached"]
        )
        for t in by_bucket["pruned"]:
            if t.fill_source == "sibling_copy":
                write_back(t)
        logger.info(
            f"Copied foldseek columns from siblings into "
            f"{len(matches)}/{len(by_bucket['pruned'])} pruned rows"
        )

    # Write report.
    report = {
        "summary": {k: len(v) for k, v in by_bucket.items()},
        "filled": {
            "foldseek_run": sum(1 for t in tasks if t.fill_source == "foldseek_run"),
            "csv_sync": sum(1 for t in tasks if t.fill_source == "csv_sync"),
            "sibling_copy": sum(1 for t in tasks if t.fill_source == "sibling_copy"),
        },
        "still_pruned_no_sibling": [
            {"profile": t.profile, "run": t.row["run"], "step": t.row["step"]}
            for t in by_bucket["pruned"]
            if t.fill_source is None
        ],
        "errors": [
            {
                "profile": t.profile,
                "run": t.row["run"],
                "step": t.row["step"],
                "error": t.error,
            }
            for t in tasks
            if t.error
        ],
        "per_task_wall": [
            {
                "profile": t.profile,
                "run": t.row["run"],
                "step": t.row["step"],
                "n_designable": t.n_designable,
                "wall_s": round(t.wall_seconds, 1),
            }
            for t in tasks
            if t.fill_source == "foldseek_run"
        ],
    }
    if args.shard is not None:
        # Per-shard report so concurrent shards don't clobber each other.
        report_path = args.report_path.replace(".json", f".shard{args.shard}.json")
    else:
        report_path = args.report_path
    Path(report_path).write_text(json.dumps(report, indent=2))
    logger.info(f"Report written to {report_path}")
    logger.info(f"Filled: {report['filled']}")
    logger.info(
        f"Still pruned without sibling: {len(report['still_pruned_no_sibling'])}"
    )


if __name__ == "__main__":
    main()
