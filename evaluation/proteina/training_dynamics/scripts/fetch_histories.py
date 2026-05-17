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

# wandb's scan_history(keys=...) only returns rows where ALL keys are present.
# Per-step and per-epoch metrics live on different rows, so we scan each key
# in its own pass and merge on wandb's row index `_step`. `_runtime` and
# `_timestamp` are wandb-automatic, present on every row.
ALL_KEYS = [
    "train/trans_loss_epoch",
    "train/loss_epoch",
    "train/repa/loss_epoch",
    "trainer/global_step",
    "scaling/nsamples_processed",
]


def _scan(run, key: str) -> pd.DataFrame:
    rows = list(run.scan_history(keys=[key, "_runtime", "_timestamp"]))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # wandb's row index is implicit; reconstruct from the order it returned.
    # Use _timestamp as the merge key — it's a strict monotonic per-row id.
    return df


def fetch_one(api: wandb.Api, run_id: str) -> pd.DataFrame | None:
    try:
        run = api.run(f"{ENTITY_PROJECT}/{run_id}")
    except wandb.errors.CommError as e:
        print(f"  [skip] {run_id}: not found on wandb ({e})")
        return None

    frames = {}
    for k in ALL_KEYS:
        df = _scan(run, k)
        if df.empty:
            print(f"    {k}: empty")
            continue
        print(f"    {k}: {len(df)} rows")
        frames[k] = df[[k, "_runtime", "_timestamp"]]

    if not frames:
        print(f"  [skip] {run_id}: no rows for any tracked key")
        return None

    # Outer-merge on (_runtime, _timestamp). Same wandb-flush event ⇒ same
    # _timestamp across keys; differing flushes get their own row.
    out = None
    for k, f in frames.items():
        if out is None:
            out = f.copy()
        else:
            out = out.merge(f, on=["_runtime", "_timestamp"], how="outer")
    out = out.sort_values("_timestamp").reset_index(drop=True)
    out["run_id"] = run_id
    return out


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
