"""Denoised cos_sim trajectory pull for the H1 "does cos_sim plateau
differently across encoders?" question.

The earlier H1 CSVs sampled a SINGLE step per target (the noisy `*_step`
metric). Within-run single-step scatter (~±0.03) is as large as the apparent
cross-run trend, so a plateau-vs-rising read is unreliable.

This script pulls a WIDE window (±`HALF_WIN` steps) around each target step
and reports mean ± std ± n, so we can tell whether a trajectory difference
exceeds within-window noise. Each run's aligned layer differs, so we read the
matching cos_sim_layer_{L} key per run.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from statistics import mean, pstdev

from wandb import Api

ENTITY = "sr2173-university-of-cambridge"
PROJECT = "proteina-repa"
HALF_WIN = 10_000  # ±10k steps per window

# (label, run_id, aligned_layer)
RUNS = [
    ("PDB_L9_GN", "proteina_60m_repa_l9_256_per_residue_bs24_2gpu", 9),
    ("AFDB_L4_GN", "proteina_60m_repa_l4_256_afdb_per_residue", 4),
    ("AFDB_L9_GN", "proteina_60m_repa_l9_256_afdb_per_residue", 9),
    ("AFDB_MPNN_L9", "proteina_60m_repa_mpnn_l9_256_afdb_per_residue", 9),
]

TARGETS = [
    100_000,
    200_000,
    300_000,
    400_000,
    500_000,
    600_000,
    700_000,
    800_000,
    900_000,
    1_000_000,
    1_100_000,
    1_200_000,
]

OUT = Path("evaluation/proteina/generation/results/variance/h1_cossim_denoised.csv")


def pull_window(run, key, lo, hi, max_retries=4):
    for attempt in range(max_retries):
        try:
            return list(
                run.scan_history(
                    keys=["trainer/global_step", key],
                    min_step=lo,
                    max_step=hi,
                    page_size=2000,
                )
            )
        except Exception as e:
            wait = 2**attempt * 5
            print(
                f"    retry {attempt+1} [{lo},{hi}]: {e!r}; sleep {wait}s", flush=True
            )
            time.sleep(wait)
    return None


def main():
    api = Api()
    out_rows = []
    for label, run_id, layer in RUNS:
        key = f"train/repa/cos_sim_layer_{layer}_step"
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        last_step = run.summary.get("trainer/global_step", 0)
        print(f"\n=== {label}  (layer {layer}, last_step={last_step}) ===", flush=True)
        for tgt in TARGETS:
            if tgt > (last_step or 0) + HALF_WIN:
                continue
            rows = pull_window(run, key, tgt - HALF_WIN, tgt + HALF_WIN)
            if not rows:
                print(f"  {tgt:>9}: no data", flush=True)
                continue
            vals = [r[key] for r in rows if r.get(key) is not None]
            if not vals:
                continue
            m = mean(vals)
            sd = pstdev(vals) if len(vals) > 1 else 0.0
            print(
                f"  {tgt:>9}: mean={m:.4f}  std={sd:.4f}  n={len(vals)}  range=[{min(vals):.3f},{max(vals):.3f}]",
                flush=True,
            )
            out_rows.append(
                {
                    "run": label,
                    "layer": layer,
                    "target": tgt,
                    "cos_sim_mean": round(m, 5),
                    "cos_sim_std": round(sd, 5),
                    "n": len(vals),
                    "cos_sim_min": round(min(vals), 4),
                    "cos_sim_max": round(max(vals), 4),
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "layer",
                "target",
                "cos_sim_mean",
                "cos_sim_std",
                "n",
                "cos_sim_min",
                "cos_sim_max",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
