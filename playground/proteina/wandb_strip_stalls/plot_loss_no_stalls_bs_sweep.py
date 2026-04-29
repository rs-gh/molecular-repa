"""Same stall-stripping plot for the bs12/bs24/bs80/bs80_lr3x sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

from plot_loss_no_stalls import KEY, fetch_run_by_name, strip_stalls

RUNS: dict[str, tuple[str, str]] = {
    "0429-1345-proteina_60m_repa_l4_128_per_residue_bs24": ("tab:red", "-"),
    "0429-1345-proteina_60m_repa_l4_128_per_residue_bs12": ("tab:green", "-"),
    "0428-1652-proteina_60m_repa_l4_128_per_residue_bs80_lr3x": ("tab:purple", "-"),
    "0428-1652-proteina_60m_repa_l4_128_per_residue_bs80": ("tab:blue", "--"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gap-mult", type=float, default=3.0)
    p.add_argument("--min-gap-s", type=float, default=1500.0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "loss_no_stalls_bs_sweep.png",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).parent / "history_cache_bs_sweep.pkl",
    )
    p.add_argument("--refresh", action="store_true")
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
            rows = list(run.scan_history(keys=[KEY, "_runtime", "_timestamp"]))
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
