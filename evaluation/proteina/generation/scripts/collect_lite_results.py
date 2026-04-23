"""Collect lite FID sweep results into a single CSV for convergence plotting.

Scans eval_output/inference_fid_60m_*_lite_step_*/ directories for result CSVs,
extracts run type and step, computes samples_seen, and saves a unified CSV.

Usage:
    python evaluation/proteina/generation/scripts/collect_lite_results.py
"""

import glob
import os
import re

import pandas as pd

# __file__ lives at evaluation/proteina/generation/scripts/ — 4 levels to repo root.
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
EVAL_OUTPUT = os.path.join(REPO_ROOT, "eval_output")
# ../results/pdb/fid/ → evaluation/proteina/generation/results/pdb/fid/
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "pdb", "fid", "lite_convergence_all.csv"
)

# Map directory name fragment → display name + batch size
RUN_MAP = {
    "baseline": ("Baseline (60M)", 6),
    "repa_layer0": ("REPA (L0)", 4),
    "repa_layer9": ("REPA (L9)", 4),
    "repa": ("REPA (L4)", 4),  # Must come after layer0/layer9 to avoid false match
}

METRIC_COLS = [
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
]


def identify_run(dirname: str) -> tuple:
    """Parse directory name to extract run type and step.

    Example: "inference_fid_60m_repa_layer0_lite_step_250000"
             → ("repa_layer0", 250000)
    """
    step_match = re.search(r"step_(\d+)$", dirname)
    if not step_match:
        return None, None
    step = int(step_match.group(1))

    # Strip prefix and suffix to get run fragment
    fragment = dirname.replace("inference_fid_60m_", "").replace(
        f"_lite_step_{step}", ""
    )

    # Match against known runs (order matters: longer matches first)
    for key in RUN_MAP:
        if fragment == key:
            return key, step

    return None, None


def main():
    pattern = os.path.join(
        EVAL_OUTPUT, "inference_fid_60m_*_lite_step_*", "results_*_fid.csv"
    )
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"No lite result CSVs found matching: {pattern}")
        print(
            "Have you run the hpc-scripts/proteina/evaluation/generation/eval_fid_lite_sweep.sh jobs?"
        )
        return

    records = []
    for csv_path in csv_files:
        dirname = os.path.basename(os.path.dirname(csv_path))
        run_key, step = identify_run(dirname)
        if run_key is None:
            print(f"  WARNING: Could not parse directory: {dirname}, skipping")
            continue

        display_name, batch_size = RUN_MAP[run_key]

        df = pd.read_csv(csv_path)
        row = df.iloc[0]

        record = {
            "run": display_name,
            "global_step": step,
            "batch_size": batch_size,
            "samples_seen": step * batch_size,
        }
        for col in METRIC_COLS:
            record[col] = row.get(col, float("nan"))

        records.append(record)
        print(
            f"  {display_name:16s} step={step:>7d}  PDB_FID={record['_res_PDB_FID']:.1f}"
        )

    result_df = (
        pd.DataFrame(records).sort_values(["run", "global_step"]).reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result_df)} rows to {OUTPUT_PATH}")

    # Print summary
    print(f"\nRuns found: {sorted(result_df['run'].unique())}")
    for run_name, group in result_df.groupby("run"):
        print(
            f"  {run_name}: {len(group)} checkpoints, "
            f"steps {group['global_step'].min()}-{group['global_step'].max()}"
        )


if __name__ == "__main__":
    main()
