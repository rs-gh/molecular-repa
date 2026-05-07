"""Stall-stripped trans-loss overlay for the 10-run REPA comparison panel.

Runs are the ones shown in the wandb report screenshot (2026-05-04): baseline +
9 REPA variants at n=128, mostly per_residue. Some runs have wall-clock stalls
(jobs that hung but kept logging); we strip those before overlaying.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

from plot_loss_no_stalls import KEY, fetch_run_by_name, strip_stalls

# Colors chosen to roughly match the wandb report screenshot.
RUNS: dict[str, tuple[str, str]] = {
    "0427-1826-proteina_60m_baseline_128_bs80": ("tab:orange", "-"),
    "0428-1652-proteina_60m_repa_l4_128_per_residue_bs80": ("tab:blue", "-"),
    "0428-1652-proteina_60m_repa_l4_128_per_residue_bs80_lr3x": ("tab:red", "-"),
    "0428-1840-proteina_60m_repa_l4_128_per_residue_random": ("tab:green", "-"),
    "0429-2348-proteina_60m_repa_l4_128_per_residue_pw_structure": ("darkgreen", "-"),
    "0430-0003-proteina_60m_repa_l4_128_per_residue_pw_torsional": ("tab:purple", "-"),
    "0430-0003-proteina_60m_repa_l4_128_per_residue_mc_edge": ("tab:brown", "-"),
    "0418-1746-proteina_60m_repa_l4_128_per_sample": ("yellowgreen", "-"),
    "0418-1759-proteina_60m_repa_esm_l4_128_per_residue": ("tab:blue", "--"),
    "0419-1830-proteina_60m_repa_esm_l9_128_per_sample": ("darkgreen", "--"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gap-mult", type=float, default=3.0)
    p.add_argument("--min-gap-s", type=float, default=1500.0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "loss_no_stalls_repa_overlay.png",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).parent / "history_cache_repa_overlay.pkl",
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
            frames.append(df)
        all_df = pd.concat(frames, ignore_index=True)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_pickle(args.cache)
        print(f"Cached to {args.cache}")

    fig, (ax_raw, ax_clean) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
    for name, (color, ls) in RUNS.items():
        sub = all_df[all_df["run_name"] == name]
        if sub.empty:
            print(f"WARN: no rows for {name}")
            continue
        cleaned = strip_stalls(sub, gap_mult=args.gap_mult, min_gap_s=args.min_gap_s)
        removed_h = cleaned.attrs["seconds_removed"] / 3600.0
        n_gaps = cleaned.attrs["n_gaps_collapsed"]
        label = f"{name}  (-{removed_h:.1f}h, {n_gaps} gaps)"
        print(f"{name}: removed {removed_h:.2f}h across {n_gaps} gaps")

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
    ax_clean.legend(loc="upper right", fontsize=7)
    ax_raw.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
