"""Fetch wandb training-loss histories for every run in `runs.ABLATIONS`.

Pulls these keys (those that exist for the run):
  - train/trans_loss_epoch      — FM translation loss (comparable across all runs)
  - train/loss_epoch            — total training loss (= trans + aux + λ·repa for REPA runs)
  - train/repa/loss_epoch       — REPA-only loss (REPA runs only)
  - trainer/global_step         — optimizer step
  - scaling/nsamples_processed  — total samples (bs-fair x-axis)
  - _runtime, _timestamp        — wall-clock metadata for stall stripping

Each run is cached as a pickled DataFrame at
  evaluation/proteina/training_dynamics/cache/<run_id>.pkl

`--refresh` re-downloads. `--only <id> [<id> …]` restricts the set.
Runs that don't exist on wandb yet (configs-only rows in the ablation doc) are
silently skipped after one wandb 4xx.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import wandb

sys.path.insert(0, str(Path(__file__).parent))
from runs import ENTITY_PROJECT, all_run_ids  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

# Use run.history(samples=N) for server-side downsampling — orders of magnitude
# faster than scan_history for long runs (full scan was >30 min on a 1M-step
# baseline). N=4000 samples gives ~250-step resolution on a 1M-step run, which
# is plenty for the smoothed convergence curves we want.
ALL_KEYS = [
    "train/trans_loss_epoch",
    "train/loss_epoch",
    "train/repa/loss_epoch",
    "trainer/global_step",
    "scaling/nsamples_processed",
]
SAMPLES = 4000


def fetch_one(api: wandb.Api, run_id: str) -> pd.DataFrame | None:
    try:
        run = api.run(f"{ENTITY_PROJECT}/{run_id}")
    except wandb.errors.CommError as e:
        print(f"  [skip] {run_id}: not found on wandb ({e})")
        return None

    # Always per-key. The combined call returns the intersection of rows where
    # every key has a value, which on some runs is dominated by late-training
    # steps (60-80 rows starting from step 400k+) and silently drops early
    # convergence — exactly the part we want to plot.
    if True:
        frames = []
        for k in ALL_KEYS:
            try:
                f = run.history(keys=[k], samples=SAMPLES, pandas=True, x_axis="_step")
            except Exception as e:
                print(f"    {k}: ERR {e}")
                continue
            if f is None or f.empty:
                continue
            print(f"    {k}: {len(f)} rows")
            frames.append(
                f[[c for c in ("_step", "_runtime", "_timestamp", k) if c in f.columns]]
            )
        if not frames:
            print(f"  [skip] {run_id}: no rows for any tracked key")
            return None
        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on="_step", how="outer", suffixes=("", "_dup"))
            for c in list(out.columns):
                if c.endswith("_dup"):
                    base = c[:-4]
                    if base in out.columns:
                        out[base] = out[base].combine_first(out[c])
                    out = out.drop(columns=[c])
        df = out.sort_values("_step").reset_index(drop=True)
    # Epoch-cadence losses and step-cadence (trainer/global_step,
    # scaling/nsamples_processed) live on disjoint `_step` rows after merge.
    # Forward-fill the step counters so epoch-loss rows carry a usable x value.
    df = df.sort_values("_step").reset_index(drop=True)
    for c in (
        "trainer/global_step",
        "scaling/nsamples_processed",
        "_runtime",
        "_timestamp",
    ):
        if c in df.columns:
            df[c] = df[c].ffill()
    df["run_id"] = run_id
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--refresh", action="store_true", help="Re-download even if cache exists"
    )
    p.add_argument("--only", nargs="+", default=None, help="Restrict to these run ids")
    args = p.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=120)

    targets = args.only or all_run_ids()
    print(f"Fetching {len(targets)} runs into {CACHE_DIR}")
    for i, run_id in enumerate(targets, 1):
        out = CACHE_DIR / f"{run_id}.pkl"
        if out.exists() and not args.refresh:
            print(f"[{i:3d}/{len(targets)}] {run_id}: cached")
            continue
        print(f"[{i:3d}/{len(targets)}] {run_id}: fetching ...", flush=True)
        df = fetch_one(api, run_id)
        if df is None:
            # Touch an empty marker so we don't retry every run.
            out.with_suffix(".missing").write_text("")
            continue
        df.to_pickle(out)
        print(f"           wrote {out.name}  rows={len(df)}")


if __name__ == "__main__":
    main()
