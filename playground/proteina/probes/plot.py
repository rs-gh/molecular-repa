"""Plots for the proteina probe sweep — analogues of REPA paper figures.

Reads `sweep_results.csv` (produced by run_sweep.py) and emits three figures
into `playground/proteina/probes/figures/`:

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
  python playground/proteina/probes/plot.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
CSV = HERE / "sweep_results.csv"
FIG_DIR = HERE / "figures"


RUN_ORDER = ["baseline", "repa_l0", "repa_l4", "repa_l9"]
RUN_COLORS = {
    "baseline": "#1f77b4",  # blue
    "repa_l0": "#ff7f0e",  # orange
    "repa_l4": "#d62728",  # red
    "repa_l9": "#2ca02c",  # green
    "gearnet": "black",
}
RUN_ALIGNED_LAYER = {"baseline": None, "repa_l0": 0, "repa_l4": 4, "repa_l9": 9}

# Per-run batch sizes for the 512 _v2 runs (see reference_proteina_batch_sizes.md).
# bs differs because GearNet adds ~10GB GPU memory on top of the main model.
RUN_BATCH_SIZE = {
    "baseline": 6,
    "repa_l0": 4,
    "repa_l4": 4,
    "repa_l9": 4,
}


def _load() -> pd.DataFrame:
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
    """Mean value per encoder (sentinel layer = -1)."""
    enc = df[df["layer"] == -1]
    out: Dict[str, float] = {}
    for run, grp in enc.groupby("run"):
        out[run] = float(grp[metric].mean())
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(CSV))
    ap.add_argument("--outdir", type=str, default=str(FIG_DIR))
    args = ap.parse_args()

    global CSV, FIG_DIR
    CSV = Path(args.csv)
    FIG_DIR = Path(args.outdir)

    df = _load()
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
