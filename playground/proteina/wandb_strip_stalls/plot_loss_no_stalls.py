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
    gap_mult: float = 3.0,
    min_gap_s: float = 1500.0,
) -> pd.DataFrame:
    """Collapse inter-log gaps that are both >> typical cadence AND > min_gap_s.

    Each oversized gap is replaced by the run's typical inter-log spacing, so
    the surviving points sit on a contiguous "productive runtime" axis.
    """
    df = df.dropna(subset=[key]).sort_values("_runtime").reset_index(drop=True).copy()
    raw_t = df["_runtime"].to_numpy()
    dt = np.diff(raw_t, prepend=raw_t[0])
    pos = dt[dt > 0]
    typical = float(np.median(pos)) if pos.size else 0.0

    bad = (dt > gap_mult * typical) & (dt > min_gap_s)
    excess = np.where(bad, np.maximum(dt - typical, 0), 0).cumsum()
    df["runtime_corrected_h"] = (raw_t - excess) / 3600.0
    df["runtime_h"] = raw_t / 3600.0
    df.attrs["n_gaps_collapsed"] = int(bad.sum())
    df.attrs["seconds_removed"] = float(excess[-1]) if len(excess) else 0.0
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gap-mult",
        type=float,
        default=3.0,
        help="A gap counts as a stall if dt > gap_mult * median_dt",
    )
    p.add_argument(
        "--min-gap-s",
        type=float,
        default=1500.0,
        help="...AND dt > min_gap_s (absolute floor, seconds)",
    )
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
        cleaned = strip_stalls(sub, gap_mult=args.gap_mult, min_gap_s=args.min_gap_s)
        removed_h = cleaned.attrs["seconds_removed"] / 3600.0
        n_gaps = cleaned.attrs["n_gaps_collapsed"]
        label = f"{name}  (-{removed_h:.1f}h, {n_gaps} gaps)"

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
            f"stalls removed (gap > {args.gap_mult}x median AND > {args.min_gap_s:.0f}s)",
        ),
    ]:
        ax.set_xlabel("Time (hours)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    ax_raw.set_ylabel("train/trans_loss_epoch")
    ax_raw.set_ylim(0.3, 1.0)
    ax_clean.set_ylim(0.3, 1.0)
    ax_clean.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
