"""One-off backfill: add SS-metric columns to existing sweep_results.jsonl rows.

Reads each row of `<sweep_dir>/sweep_results.jsonl`, locates the matching
`eval_output/<config_slug>_<suffix>/samples_fid/` dir, runs
`compute_ss_metrics` on the existing PDBs (no generation, no GPU), and
merges the resulting `_res_ss_*` columns into the row in place. JSONL is
rewritten atomically (write-tmp + rename).

Usage:
    python evaluation/proteina/generation/scripts/backfill_ss.py \
        --sweep_dir evaluation/proteina/generation/results/paper/n128_paper_layer \
        --ss_reference_pdb_path  /rds/.../ss_reference_pdb.pt \
        --ss_reference_afdb_path /rds/.../ss_reference_afdb.pt

Idempotent: rows that already have an `_res_ss_n` column are skipped unless
`--force` is passed. Errors per row are logged but don't abort the run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `utils.ss_metrics` resolves

REPO_ROOT = (
    HERE.parent.parent.parent.parent
)  # evaluation/proteina/generation/scripts/ -> repo

# eval_output dirs sometimes live on /rds (jobs that ran from a /rds CWD) instead
# of /home (the repo CWD). Try both; first hit wins.
EVAL_OUTPUT_ROOTS = [
    REPO_ROOT / "eval_output",
    Path("/rds/user/sr2173/hpc-work/proteina/eval_output"),
]


def _samples_dir_for_row(row: dict) -> Path | None:
    """Mirror run_sweep's eval_output dir layout for a JSONL row."""
    if "config_name" not in row or "run" not in row or "step" not in row:
        return None
    config_slug = str(row["config_name"]).replace("/", "_")
    output_suffix = f"sweep_{row['run']}_step_{row['step']}"
    rel = f"{config_slug}_{output_suffix}/samples_fid"
    for root in EVAL_OUTPUT_ROOTS:
        cand = root / rel
        if cand.exists() and any(cand.glob("*_fid.pdb")):
            return cand
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", type=str, required=True)
    p.add_argument("--ss_reference_pdb_path", type=str, default=None)
    p.add_argument("--ss_reference_afdb_path", type=str, default=None)
    p.add_argument(
        "--force", action="store_true", help="Re-run even if _res_ss_n is present"
    )
    return p.parse_args()


def main():
    args = parse_args()
    from utils.ss_metrics import compute_ss_metrics  # noqa: PLC0415

    sweep_dir = Path(args.sweep_dir).resolve()
    jsonl_path = sweep_dir / "sweep_results.jsonl"
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)

    rows = [
        json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()
    ]
    print(f"Loaded {len(rows)} rows from {jsonl_path}")

    n_done = n_skipped = n_errored = 0
    for row in rows:
        if "error" in row:
            continue
        if not args.force and "_res_ss_n" in row:
            n_skipped += 1
            continue
        samples_dir = _samples_dir_for_row(row)
        if samples_dir is None or not samples_dir.exists():
            print(
                f"  skip: samples_fid not found for run={row.get('run')} step={row.get('step')} ({samples_dir})"
            )
            n_errored += 1
            continue

        list_of_pdbs = sorted(
            samples_dir.glob("*_fid.pdb"),
            key=lambda p: int(p.stem.split("_")[0]),
        )
        if not list_of_pdbs:
            print(f"  skip: no PDBs in {samples_dir}")
            n_errored += 1
            continue

        # Designable subset: not preserved in JSONL. Run all-samples only;
        # designable variants stay missing for backfilled rows. Trade-off
        # accepted to keep this a pure offline backfill.
        try:
            ss = compute_ss_metrics(
                list_of_pdbs=[str(p) for p in list_of_pdbs],
                designable_pdbs=None,
                ss_reference_pdb_path=args.ss_reference_pdb_path,
                ss_reference_afdb_path=args.ss_reference_afdb_path,
                cache_dir=samples_dir.parent / "ss_cache",
            )
        except Exception as exc:
            print(
                f"  error on run={row.get('run')} step={row.get('step')}: {type(exc).__name__}: {exc}"
            )
            n_errored += 1
            continue

        row.update(ss)
        n_done += 1
        ssH = ss.get("_res_ss_frac_H", float("nan"))
        ssE = ss.get("_res_ss_frac_E", float("nan"))
        jpdb = ss.get("_res_ss_jsd_pdb", float("nan"))
        jafdb = ss.get("_res_ss_jsd_afdb", float("nan"))
        print(
            f"  {row.get('run'):<40} step={row.get('step'):<8} "
            f"N={ss.get('_res_ss_n')} H={ssH:.3f} E={ssE:.3f} "
            f"jsd_pdb={jpdb:.4f} jsd_afdb={jafdb:.4f}"
        )

    # Atomic rewrite
    tmp = jsonl_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, jsonl_path)
    print(
        f"\nUpdated {n_done} rows, skipped {n_skipped} (already had SS), {n_errored} errored."
    )
    print(f"Rewrote {jsonl_path}")


if __name__ == "__main__":
    main()
