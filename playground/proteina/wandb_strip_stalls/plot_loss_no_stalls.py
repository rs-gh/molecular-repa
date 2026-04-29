"""Replot train/trans_loss_epoch vs wall time with stalled spans removed.

A "stall" is a run of points where the loss is ~constant over a long wall-time
window (typically a job that hung but kept the wandb heartbeat alive). We drop
those points and collapse the time axis so the remaining curve reflects only
productive training time.

Usage:
    python plot_loss_no_stalls.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

ENTITY_PROJECT = "sr2173-university-of-cambridge/proteina-repa"
KEY = "train/trans_loss_epoch"

# Display name -> (color, linestyle). Match the original report.
RUNS: dict[str, tuple[str, str]] = {
    "0428-1840-proteina_60m_repa_l4_128_per_residue_random": ("tab:green", "-"),
    "0427-1826-proteina_60m_baseline_128_bs80": ("tab:orange", "-"),
    "0428-1652-proteina_60m_repa_l4_128_per_residue_bs80": ("tab:blue", "-"),
    "0418-1759-proteina_60m_repa_esm_l4_128_per_residue": ("tab:blue", "--"),
}


def fetch_run_by_name(api: wandb.Api, name: str):
    runs = list(api.runs(ENTITY_PROJECT, filters={"display_name": name}))
    if not runs:
        raise RuntimeError(f"No run found with display_name={name!r}")
    if len(runs) > 1:
        raise RuntimeError(f"Multiple runs match {name!r}: {[r.id for r in runs]}")
    return runs[0]


def strip_stalls(
    df: pd.DataFrame,
    key: str = KEY,
    window_s: float = 1800.0,
    std_thresh: float = 2e-3,
    min_points: int = 5,
) -> pd.DataFrame:
    """Drop points whose preceding `window_s` of loss values has std < threshold.

    Then build `runtime_corrected_h` by subtracting the removed gaps so the
    surviving points sit on a contiguous time axis.
    """
    df = df.dropna(subset=[key]).sort_values("_runtime").reset_index(drop=True)
    t = df["_runtime"].to_numpy()
    y = df[key].to_numpy()

    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        lo = np.searchsorted(t, t[i] - window_s)
        if i - lo + 1 >= min_points and y[lo : i + 1].std() < std_thresh:
            keep[i] = False

    kept = df[keep].reset_index(drop=True).copy()
    if len(kept) == 0:
        kept["runtime_corrected_h"] = []
        return kept

    raw_t = kept["_runtime"].to_numpy()
    dt = np.diff(raw_t, prepend=raw_t[0])
    pos = dt[dt > 0]
    typical = float(np.median(pos)) if pos.size else 0.0
    # any inter-point jump >> typical cadence is a removed stall; collapse it
    excess = np.where(dt > 5 * typical, np.maximum(dt - typical, 0), 0).cumsum()
    kept["runtime_corrected_h"] = (raw_t - excess) / 3600.0
    kept["runtime_h"] = raw_t / 3600.0
    return kept


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window-s", type=float, default=1800.0)
    p.add_argument("--std-thresh", type=float, default=2e-3)
    p.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "loss_no_stalls.png"
    )
    p.add_argument(
        "--cache", type=Path, default=Path(__file__).parent / "history_cache.pkl"
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch from wandb even if cache exists",
    )
    args = p.parse_args()

    if args.cache.exists() and not args.refresh:
        print(f"Loading cached history from {args.cache}")
        all_df = pd.read_pickle(args.cache)
    else:
        api = wandb.Api()
        frames = []
        for name in RUNS:
            print(f"Fetching {name} ...")
            run = fetch_run_by_name(api, name)
            # scan_history streams all rows including internal _runtime/_timestamp
            rows = [r for r in run.scan_history(keys=[KEY, "_runtime", "_timestamp"])]
            df = pd.DataFrame(rows)
            df["run_name"] = name
            df["run_id"] = run.id
            frames.append(df)
        all_df = pd.concat(frames, ignore_index=True)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_pickle(args.cache)
        print(f"Cached to {args.cache}")

    fig, (ax_raw, ax_clean) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for name, (color, ls) in RUNS.items():
        sub = all_df[all_df["run_name"] == name]
        if sub.empty:
            print(f"WARN: no rows for {name}")
            continue
        cleaned = strip_stalls(sub, window_s=args.window_s, std_thresh=args.std_thresh)
        n_drop = len(sub.dropna(subset=[KEY])) - len(cleaned)
        label = f"{name}  (-{n_drop} pts)"

        raw_sorted = sub.dropna(subset=[KEY]).sort_values("_runtime")
        ax_raw.plot(
            raw_sorted["_runtime"] / 3600.0,
            raw_sorted[KEY],
            color=color,
            linestyle=ls,
            linewidth=1.2,
            label=name,
        )
        ax_clean.plot(
            cleaned["runtime_corrected_h"],
            cleaned[KEY],
            color=color,
            linestyle=ls,
            linewidth=1.2,
            label=label,
        )

    for ax, title in [
        (ax_raw, "raw"),
        (
            ax_clean,
            f"stalls removed (window={args.window_s:.0f}s, std<{args.std_thresh})",
        ),
    ]:
        ax.set_xlabel("Time (hours)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    ax_raw.set_ylabel("train/trans_loss_epoch")
    ax_clean.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
