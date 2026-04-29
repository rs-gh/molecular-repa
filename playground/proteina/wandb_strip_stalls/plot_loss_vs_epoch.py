"""Plot trans_loss and repa_loss vs epoch for the four bs80 / n=128 runs.

Two side-by-side panels:
    left  — train/trans_loss_epoch vs epoch
    right — train/repa/loss_epoch vs epoch  (baseline has no repa loss; skipped)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

ENTITY_PROJECT = "sr2173-university-of-cambridge/proteina-repa"
TRANS_KEY = "train/trans_loss_epoch"
REPA_KEY = "train/repa/loss_epoch"
EPOCH_KEY = "epoch"

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
    return runs[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "loss_vs_epoch.png"
    )
    p.add_argument(
        "--cache", type=Path, default=Path(__file__).parent / "epoch_history_cache.pkl"
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
            # scan_history filters to rows where ALL keys are present, so query
            # each metric separately and merge on epoch.
            trans = pd.DataFrame(list(run.scan_history(keys=[EPOCH_KEY, TRANS_KEY])))
            try:
                repa = pd.DataFrame(list(run.scan_history(keys=[EPOCH_KEY, REPA_KEY])))
            except Exception:
                repa = pd.DataFrame(columns=[EPOCH_KEY, REPA_KEY])
            if repa.empty:
                df = trans
            else:
                df = trans.merge(repa, on=EPOCH_KEY, how="outer")
            df["run_name"] = name
            frames.append(df)
        all_df = pd.concat(frames, ignore_index=True)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_pickle(args.cache)
        print(f"Cached to {args.cache}")

    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(16, 6))
    for name, (color, ls) in RUNS.items():
        sub = all_df[all_df["run_name"] == name]
        if sub.empty:
            print(f"WARN: no rows for {name}")
            continue
        if TRANS_KEY in sub.columns:
            t = sub.dropna(subset=[EPOCH_KEY, TRANS_KEY]).sort_values(EPOCH_KEY)
            ax_t.plot(
                t[EPOCH_KEY],
                t[TRANS_KEY],
                color=color,
                linestyle=ls,
                linewidth=1.2,
                label=name,
            )
        if REPA_KEY in sub.columns:
            r = sub.dropna(subset=[EPOCH_KEY, REPA_KEY]).sort_values(EPOCH_KEY)
            if len(r):
                ax_r.plot(
                    r[EPOCH_KEY],
                    r[REPA_KEY],
                    color=color,
                    linestyle=ls,
                    linewidth=1.2,
                    label=name,
                )

    ax_t.set_title("train/trans_loss_epoch vs epoch")
    ax_t.set_xlabel("epoch")
    ax_t.set_ylabel("trans loss")
    ax_t.set_ylim(0.3, 1.0)
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="upper right", fontsize=8)

    ax_r.set_title("train/repa/loss_epoch vs epoch  (baseline has none)")
    ax_r.set_xlabel("epoch")
    ax_r.set_ylabel("repa loss")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
