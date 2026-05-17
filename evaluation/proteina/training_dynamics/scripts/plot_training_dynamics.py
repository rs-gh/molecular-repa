"""Plot per-ablation training-dynamics figures.

For each ablation block in `runs.ABLATIONS`, emit one figure per metric:
  - convergence: train/trans_loss_epoch (FM translation loss; comparable across
    baseline and REPA runs)
  - repa: train/repa/loss_epoch (REPA-only; baselines are absent here)

X-axis is `trainer/global_step` by default; pass `--x nsamples` for a
batch-size-fair view (uses `scaling/nsamples_processed`).

Histories must already be cached by `fetch_histories.py`. Missing-on-wandb
runs are tolerated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from runs import ABLATIONS, run_meta  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
FIG_DIR = ROOT / "figures"

METRICS = {
    "trans": {
        "key": "train/trans_loss_epoch",
        "title": "Flow-matching translation loss (train/trans_loss_epoch)",
        "ylabel": "train/trans_loss_epoch",
        "include_baselines": True,
    },
    "repa": {
        "key": "train/repa/loss_epoch",
        "title": "REPA loss (train/repa/loss_epoch)",
        "ylabel": "train/repa/loss_epoch",
        "include_baselines": False,
    },
    "total": {
        "key": "train/loss_epoch",
        "title": "Total training loss (train/loss_epoch = trans + aux + λ·repa)",
        "ylabel": "train/loss_epoch",
        "include_baselines": True,
    },
}

X_AXES = {
    "step": ("trainer/global_step", "optimizer step", 1.0),
    "nsamples": ("scaling/nsamples_processed", "samples processed", 1.0),
}


def load(run_id: str) -> pd.DataFrame | None:
    p = CACHE_DIR / f"{run_id}.pkl"
    if not p.exists():
        return None
    return pd.read_pickle(p)


def plot_ablation(
    ablation: str, run_ids: list[str], metric: str, x_axis: str, smooth: int, ylog: bool
) -> Path | None:
    spec = METRICS[metric]
    key = spec["key"]
    xkey, xlabel, _ = X_AXES[x_axis]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = 0
    for run_id in run_ids:
        meta = run_meta(run_id)
        if meta is None:
            print(f"  WARN: no metadata for {run_id}")
            continue
        label, color, ls, is_baseline = meta
        if is_baseline and not spec["include_baselines"]:
            continue
        df = load(run_id)
        if df is None:
            print(f"  WARN: no cache for {run_id} ({ablation})")
            continue
        if key not in df.columns:
            # REPA loss column won't exist for baselines, etc.
            continue
        sub = df.dropna(subset=[key, xkey]).sort_values(xkey)
        if sub.empty:
            continue
        x = sub[xkey].to_numpy()
        y = sub[key].to_numpy()
        if smooth > 1 and len(y) > smooth:
            y = (
                pd.Series(y)
                .rolling(smooth, min_periods=1, center=True)
                .mean()
                .to_numpy()
            )
        ax.plot(
            x,
            y,
            color=color,
            linestyle=ls,
            linewidth=1.4,
            label=f"{label}  (n={len(sub)})",
            alpha=0.9,
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_xlabel(xlabel)
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(f"{ablation} — {spec['title']}", fontsize=11)
    ax.grid(True, alpha=0.3)
    if ylog:
        ax.set_yscale("log")
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()

    out = FIG_DIR / f"{ablation}__{metric}__{x_axis}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--x", choices=list(X_AXES), default="step")
    p.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling-mean window (in points); 1 disables",
    )
    p.add_argument("--ylog", action="store_true", help="log y-axis")
    p.add_argument(
        "--ablation", nargs="+", default=None, help="Restrict to these ablation keys"
    )
    p.add_argument("--metric", nargs="+", default=list(METRICS), choices=list(METRICS))
    args = p.parse_args()

    ablations = args.ablation or list(ABLATIONS)
    for ab in ablations:
        if ab not in ABLATIONS:
            print(f"unknown ablation: {ab}; known: {list(ABLATIONS)}")
            continue
        for m in args.metric:
            print(f"[{ab} | {m} | x={args.x}]")
            out = plot_ablation(ab, ABLATIONS[ab], m, args.x, args.smooth, args.ylog)
            if out:
                print(f"  -> {out.relative_to(ROOT.parent.parent)}")


if __name__ == "__main__":
    main()
