"""Pull FM / REPA loss + cos_sim trajectory for MPNN-L9-AFDB from wandb.

Targets specific training steps to avoid timeouts on the full scan_history
(which has been flaking with HTTP 500s on this run).
"""

import csv
import sys
import time
from pathlib import Path

from wandb import Api

RUN_NAME = "proteina_60m_repa_mpnn_l9_256_afdb_per_residue"
RUN_ID = "proteina_60m_repa_mpnn_l9_256_afdb_per_residue"
ENTITY = "sr2173-university-of-cambridge"
PROJECT = "proteina-repa"

TARGET_STEPS = [100_000, 200_000, 400_000, 700_000, 1_000_000, 1_200_000, 1_500_000]
KEYS = [
    "trainer/global_step",
    "train/loss_step",
    "train/repa/loss_step",
    "train/repa/cos_sim_layer_9_step",
]
OUT = Path("evaluation/proteina/generation/results/variance/h1_wandb_mpnn_afdb.csv")


def pull_window(api, run, lo, hi, max_retries=4):
    """scan_history with min_step/max_step + retry on 500s."""
    for attempt in range(max_retries):
        try:
            rows = list(
                run.scan_history(keys=KEYS, min_step=lo, max_step=hi, page_size=1000)
            )
            return rows
        except Exception as e:
            wait = 2**attempt * 5
            print(
                f"  attempt {attempt+1} failed for [{lo}, {hi}]: {e!r}; sleeping {wait}s",
                flush=True,
            )
            time.sleep(wait)
    return None


def main():
    api = Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
    print(f"Run: {run.name} state={run.state}", flush=True)

    found = {}
    for target in TARGET_STEPS:
        lo = target - 500
        hi = target + 500
        print(f"Pulling window [{lo}, {hi}] for target {target}", flush=True)
        rows = pull_window(api, run, lo, hi)
        if rows is None:
            print(f"  GAVE UP on {target}", flush=True)
            continue
        if not rows:
            print("  no rows in window", flush=True)
            continue
        # pick the row closest to target
        best = min(
            rows, key=lambda r: abs(r.get("trainer/global_step", -1e18) - target)
        )
        found[target] = best
        print(
            f"  got step={best.get('trainer/global_step')} cos_sim={best.get('train/repa/cos_sim_layer_9_step')}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target"] + KEYS)
        for target in TARGET_STEPS:
            if target not in found:
                continue
            row = found[target]
            w.writerow([target] + [row.get(k) for k in KEYS])
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
