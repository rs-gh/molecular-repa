"""Plots for the proteina probe sweep — analogues of REPA paper figures.

Reads `sweep_results.csv` (produced by run_sweep.py) and emits three figures
into `evaluation/proteina/representation/figures/`:

  fig_layerwise_P_at_L5.png        — Fig 3a analogue for contacts
      x = transformer layer index, y = P@L/5, one curve per run at a fixed step
      (by default: each run's latest step). Encoder scores drawn as dashed
      horizontal reference lines.

  fig_layerwise_cath_acc.png       — same shape, y = CATH accuracy

  fig_step_progression.png         — Fig 2c analogue
      x = training step (log scale), y = probe metric at each run's "best"
      layer (chosen per run as the argmax layer at the largest step). One
      subplot for P@L/5, one for CATH accuracy, one for each run.

Usage:
  python evaluation/proteina/representation/scripts/plot.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
# Results live in .../representation/results/, figures in .../representation/figures/.
CSV = HERE.parent / "results" / "sweep_results.csv"
# n=512 multi-checkpoint sweep outputs (layerwise + step-progression). Grouped
# into a subdir since these are n=512-specific; the per-size `fig_grid_*`
# plots (plot_per_n.py) stay at the top level.
FIG_DIR = HERE.parent / "figures" / "n512_convergence"


RUN_ORDER = ["baseline", "repa_l0", "repa_l4", "repa_l9"]
RUN_COLORS = {
    "baseline": "#1f77b4",  # blue
    "repa_l0": "#ff7f0e",  # orange
    "repa_l4": "#d62728",  # red
    "repa_l9": "#2ca02c",  # green
    "gearnet": "black",
    "random_gauss": "#7f7f7f",  # gray
    "seq_onehot": "#8c564b",  # brown
    "random_rank": "#bcbd22",  # olive
    "distance_only": "#17becf",  # cyan
    "untrained_proteina": "#e377c2",  # pink
}
RUN_ALIGNED_LAYER = {"baseline": None, "repa_l0": 0, "repa_l4": 4, "repa_l9": 9}

# Baseline rows with a single sentinel layer — plotted as horizontal reference
# lines (same convention as the gearnet encoder row at layer=-1).
BASELINE_SENTINELS = {
    -1: "gearnet",
    -2: "random_gauss",
    -3: "seq_onehot",
    -100: "random_rank",
    -101: "distance_only",
}
# Untrained-proteina sentinel offset: probe row for trunk layer L has
# `layer = -(L + 10)`. Inverse: trunk_layer = -L - 10 for L in [-19, -10].
UNTRAINED_LAYER_MIN, UNTRAINED_LAYER_MAX = -19, -10

# Per-run batch sizes for the 512 _v2 runs (see reference_proteina_batch_sizes.md).
# bs differs because GearNet adds ~10GB GPU memory on top of the main model.
# NOTE (2026-04-19): when adding 128-residue runs, bs is NOT uniform.
#   baseline-128 (job 27971089, current) trained at bs=24 throughout.
#   All REPA-128 variants (gearnet + esm, per_residue + per_sample) train at bs=80.
#   Future baseline-128 runs / restarts will most likely adopt bs=80 — CONFIRM per run
#   before appending an entry (a mid-run bs switch means one run has two regimes, and
#   nsamples = step * bs is no longer a single scalar multiply).
RUN_BATCH_SIZE = {
    "baseline": 6,
    "repa_l0": 4,
    "repa_l4": 4,
    "repa_l9": 4,
}


def _load(manifest: str = None) -> pd.DataFrame:
    df = pd.read_csv(CSV)
    # Drop error rows
    df = df[df["error"].isna() | (df["error"] == "")].copy()
    for c in [
        "p_at_L",
        "p_at_L_2",
        "p_at_L_5",
        "cath_acc",
        "cath_f1",
        "step",
        "layer",
        "dim",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Manifest filter — apples-to-apples: v1 and v2 rows come from different
    # protein samples so their numbers aren't directly comparable. If the
    # caller doesn't specify, auto-pick the most common manifest in the CSV.
    if "manifest" in df.columns:
        if manifest is None:
            counts = df["manifest"].fillna("v1").value_counts()
            manifest = str(counts.index[0]) if len(counts) else "v1"
        df = df[df["manifest"].fillna("v1") == manifest].copy()
        print(f"[plot] manifest filter: keeping manifest='{manifest}' ({len(df)} rows)")
    # Add fair-comparison x-axis: nsamples_processed = step × batch_size.
    # (At 512 res, baseline bs=6, REPA bs=4 — using raw step would misrepresent REPA.)
    df["nsamples"] = df.apply(
        lambda r: int(r["step"]) * RUN_BATCH_SIZE.get(r["run"], 1)
        if r["run"] in RUN_BATCH_SIZE
        else int(r["step"]),
        axis=1,
    )
    return df


def _encoder_values(df: pd.DataFrame, metric: str) -> Dict[str, float]:
    """Mean value per reference run (one of the BASELINE_SENTINELS layers)."""
    out: Dict[str, float] = {}
    for sentinel, name in BASELINE_SENTINELS.items():
        sub = df[df["layer"] == sentinel]
        if sub.empty:
            continue
        out[name] = float(sub[metric].mean())
    return out


def _plot_layerwise(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    # For each run, pick its MAX step present in the CSV as the "trained" snapshot.
    for run in RUN_ORDER:
        sub = df[(df["run"] == run) & (df["layer"] >= 0)]
        if sub.empty:
            continue
        max_step = sub["step"].max()
        snap = sub[sub["step"] == max_step].sort_values("layer")
        ax.plot(
            snap["layer"],
            snap[metric],
            "-o",
            color=RUN_COLORS.get(run, "gray"),
            label=f"{run} @ step {int(max_step):,}",
            linewidth=2,
            markersize=5,
        )

    # Untrained-proteina: plot as a full transformer curve by remapping its
    # sentinel layers -(L+10) back to L ∈ [0, 9]. Lets you visualize the
    # "random weights" shape alongside trained runs.
    untr = df[
        (df["run"] == "untrained_proteina")
        & (df["layer"] >= UNTRAINED_LAYER_MIN)
        & (df["layer"] <= UNTRAINED_LAYER_MAX)
    ].copy()
    if not untr.empty:
        untr["trunk_layer"] = -untr["layer"] - 10  # invert _untrained_layer_sentinel
        untr = untr.sort_values("trunk_layer")
        ax.plot(
            untr["trunk_layer"],
            untr[metric],
            "--s",
            color=RUN_COLORS["untrained_proteina"],
            label="untrained_proteina",
            linewidth=1.5,
            markersize=4,
            alpha=0.8,
        )
        # Mark the alignment layer with a vertical tick
        al = RUN_ALIGNED_LAYER.get(run)
        if al is not None:
            pt = snap[snap["layer"] == al]
            if not pt.empty:
                ax.scatter(
                    pt["layer"],
                    pt[metric],
                    s=150,
                    facecolors="none",
                    edgecolors=RUN_COLORS.get(run, "gray"),
                    linewidths=2,
                    zorder=5,
                )

    # Encoder reference lines
    enc_vals = _encoder_values(df, metric)
    for run, v in enc_vals.items():
        ax.axhline(
            v,
            linestyle="--",
            linewidth=1.2,
            color=RUN_COLORS.get(run, "black"),
            label=f"{run} (frozen)",
            alpha=0.7,
        )

    ax.set_xlabel("Transformer layer")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def _plot_progression(df: pd.DataFrame, out_path: Path) -> None:
    """Two-panel: P@L/5 and CATH acc vs samples processed, one curve per run at its aligned layer.
    X-axis is `nsamples` (step × batch_size) because baseline bs=6, REPA bs=4 at 512 res.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for run in RUN_ORDER:
        al = RUN_ALIGNED_LAYER.get(run)
        sub = df[(df["run"] == run) & (df["layer"] >= 0)]
        if sub.empty:
            continue
        if al is None:
            max_step = sub["step"].max()
            best_row = (
                sub[sub["step"] == max_step]
                .sort_values("p_at_L_5", ascending=False)
                .iloc[0]
            )
            al = int(best_row["layer"])
        cur = sub[sub["layer"] == al].sort_values("nsamples")
        ax1.plot(
            cur["nsamples"],
            cur["p_at_L_5"],
            "-o",
            color=RUN_COLORS.get(run, "gray"),
            label=f"{run} @ L{al}",
            linewidth=2,
            markersize=4,
        )
        ax2.plot(
            cur["nsamples"],
            cur["cath_acc"],
            "-o",
            color=RUN_COLORS.get(run, "gray"),
            label=f"{run} @ L{al}",
            linewidth=2,
            markersize=4,
        )

    for ax, m in [(ax1, "p_at_L_5"), (ax2, "cath_acc")]:
        v = _encoder_values(df, m)
        for run, val in v.items():
            # Skip NaN reference lines — breaks log-scale tight_layout and
            # gives a meaningless "(frozen)" entry in the legend. Happens
            # commonly on CATH when the 40-protein subset has too few labels.
            if val != val:  # NaN check
                continue
            ax.axhline(
                val,
                linestyle="--",
                linewidth=1.2,
                color=RUN_COLORS.get(run, "black"),
                label=f"{run} (frozen)",
                alpha=0.7,
            )

    ax1.set_xlabel("Samples processed (step × batch_size)")
    ax1.set_xscale("log")
    ax1.set_ylabel("P@L/5")
    ax1.set_title("Contact P@L/5 vs training progress")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Samples processed (step × batch_size)")
    ax2.set_xscale("log")
    ax2.set_ylabel("CATH accuracy")
    ax2.set_title("CATH acc vs training progress")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    global CSV, FIG_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(CSV))
    ap.add_argument("--outdir", type=str, default=str(FIG_DIR))
    ap.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Manifest version to plot (v1 = cursor-first, v2+ = reservoir). "
            "Default: whichever appears most in the CSV. v1 and v2 rows come "
            "from different protein subsets and should not be mixed."
        ),
    )
    args = ap.parse_args()

    CSV = Path(args.csv)
    FIG_DIR = Path(args.outdir)

    df = _load(manifest=args.manifest)
    if df.empty:
        print(f"No usable rows in {CSV}")
        return
    print(f"Loaded {len(df)} rows from {CSV}")
    print(df.groupby("run")["step"].nunique().to_string())

    _plot_layerwise(
        df,
        "p_at_L_5",
        "Contact P@L/5 — layer-wise (final checkpoint per run)",
        FIG_DIR / "fig_layerwise_P_at_L5.png",
    )
    _plot_layerwise(
        df,
        "cath_acc",
        "CATH fold accuracy — layer-wise (final checkpoint per run)",
        FIG_DIR / "fig_layerwise_cath_acc.png",
    )
    _plot_progression(df, FIG_DIR / "fig_step_progression.png")


if __name__ == "__main__":
    main()
