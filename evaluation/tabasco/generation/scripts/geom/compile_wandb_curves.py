#!/usr/bin/env python
"""Pull per-epoch validation metrics from WandB and produce training curves + epoch-matched comparison.

"Validation" metrics = computed during training by callbacks (100 mols/epoch, logged to WandB).
"Evaluation" metrics = post-hoc generation from final checkpoints (1000 mols, on disk).
See compile_results.py for evaluation metrics.

Usage:
    source .venv/bin/activate
    python evaluation/tabasco/generation/scripts/tabasco/geom/compile_wandb_curves.py

Outputs (in evaluation/tabasco/generation/results/geom/validation/):
    validation_curves.csv              - All per-epoch validation metrics for every model
    figures/validation_curves.png      - Line plots of validation metrics over training
    figures/validation_epoch_matched.png - Bar chart at epoch 15 (apples-to-apples)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WANDB_PROJECT = "sr2173-university-of-cambridge/tabasco"

RUNS = {
    "baseline": ["s105bkm0", "yy363ps7"],  # part1 (epochs 0-7) + part2 (epochs 8-32)
    "chemeleon_additive_same": ["0fbrr8vx"],
    "chemeleon_additive_fused": ["x3c4vid0"],
    "chemeleon_tradeoff_same": ["cqjant8r"],
    "chemeleon_tradeoff_fused": ["7u3l0zpy"],
    "mace_additive": ["7kuaxjk4", "1cj5gk44"],  # GPU live (ep 0-3) + cached (ep 3-15)
    "mace_tradeoff": ["uq02ccie", "5s25bbx3"],  # GPU live (ep 0-3) + cached (ep 3-15)
}

MODEL_ORDER = [
    "baseline",
    "chemeleon_additive_same",
    "chemeleon_additive_fused",
    "chemeleon_tradeoff_same",
    "chemeleon_tradeoff_fused",
    "mace_additive",
    "mace_tradeoff",
]

COLORS = {
    "baseline": "#4C72B0",
    "chemeleon_additive_same": "#55A868",
    "chemeleon_additive_fused": "#8ECA6E",
    "chemeleon_tradeoff_same": "#C44E52",
    "chemeleon_tradeoff_fused": "#E8907E",
    "mace_additive": "#9467BD",
    "mace_tradeoff": "#B8A9D4",
}

# Metrics to pull (wandb key -> display name, higher_is_better)
METRICS = {
    "val/validity": ("Validity", True),
    "val/loss": ("Val Loss", False),
    "val/atom_type_distribution": ("Atom Type Dist", True),
    "val/novelty": ("Novelty", True),
    "val/pb_intersection": ("PB Intersection", True),
    "val/pb_bond_lengths": ("PB Bond Lengths", True),
    "val/pb_bond_angles": ("PB Bond Angles", True),
    "val/pb_internal_steric_clash": ("PB Steric Clash", True),
    "val/connectivity": ("Connectivity", True),
}

# Subset for the training curves plot (most informative)
CURVE_METRICS = [
    "val/loss",
    "val/validity",
    "val/atom_type_distribution",
    "val/pb_intersection",
    "val/pb_bond_lengths",
    "val/pb_bond_angles",
    "val/pb_internal_steric_clash",
    "val/novelty",
    "val/connectivity",
]

# Subset for the epoch-matched bar chart
BAR_METRICS = [
    "val/validity",
    "val/atom_type_distribution",
    "val/novelty",
    "val/pb_intersection",
    "val/pb_bond_lengths",
    "val/pb_bond_angles",
    "val/pb_internal_steric_clash",
    "val/connectivity",
]

OUTPUT_DIR = Path("evaluation/tabasco/generation/results/geom/validation")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def fetch_run_history(api: wandb.Api, run_id: str) -> pd.DataFrame:
    """Fetch per-epoch validation metrics from a single WandB run."""
    run = api.run(f"{WANDB_PROJECT}/{run_id}")
    keys = ["epoch", "trainer/global_step"] + list(METRICS.keys())
    h = run.history(samples=100000, keys=keys)
    # Keep only rows with validation metrics (drop training-step rows)
    h = h.dropna(subset=["val/validity"])
    return h


def load_all_runs() -> pd.DataFrame:
    """Load and combine per-epoch metrics for all models."""
    api = wandb.Api()
    frames = []

    for model_name in MODEL_ORDER:
        run_ids = RUNS[model_name]
        parts = []
        for rid in run_ids:
            print(f"  Fetching {model_name} ({rid})...")
            h = fetch_run_history(api, rid)
            parts.append(h)

        combined = pd.concat(parts, ignore_index=True)
        combined = combined.sort_values("epoch").reset_index(drop=True)
        combined["model"] = model_name
        frames.append(combined)

    df = pd.concat(frames, ignore_index=True)
    df["epoch"] = df["epoch"].astype(int)
    df["trainer/global_step"] = df["trainer/global_step"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_training_curves(df: pd.DataFrame, output_path: Path):
    """Line plots of metrics over training epochs, one line per model."""
    n = len(CURVE_METRICS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, metric_key in enumerate(CURVE_METRICS):
        ax = axes[i]
        display_name, higher_better = METRICS[metric_key]

        for model_name in MODEL_ORDER:
            mdf = df[df["model"] == model_name].sort_values("epoch")
            ax.plot(
                mdf["epoch"],
                mdf[metric_key],
                color=COLORS[model_name],
                label=model_name.replace("_", " "),
                marker=".",
                markersize=4,
                linewidth=1.5,
            )

        ax.set_title(display_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        arrow = "\u2191" if higher_better else "\u2193"
        ax.set_ylabel(
            f"{arrow} {'higher' if higher_better else 'lower'} is better", fontsize=8
        )
        ax.grid(alpha=0.3)

        # Add vertical line at epoch 15 (last common epoch)
        ax.axvline(x=15, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(MODEL_ORDER),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "GEOM Validation Curves - Per-Epoch (100 mols, from WandB callbacks)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {output_path}")
    plt.close(fig)


def plot_epoch_matched(df: pd.DataFrame, match_epoch: int, output_path: Path):
    """Bar chart comparing all models at a fixed epoch."""
    n = len(BAR_METRICS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    # Extract values at the matched epoch
    matched = df[df["epoch"] == match_epoch].set_index("model")
    models = [m for m in MODEL_ORDER if m in matched.index]
    x = np.arange(len(models))
    bar_width = 0.6

    for i, metric_key in enumerate(BAR_METRICS):
        ax = axes[i]
        display_name, higher_better = METRICS[metric_key]
        values = matched.loc[models, metric_key].values.astype(float)
        colors = [COLORS[m] for m in models]

        # Highlight best
        best_idx = int(np.argmax(values) if higher_better else np.argmin(values))
        edge_colors = ["black" if j == best_idx else "none" for j in range(len(models))]
        linewidths = [2.0 if j == best_idx else 0.0 for j in range(len(models))]

        bars = ax.bar(
            x,
            values,
            bar_width,
            color=colors,
            edgecolor=edge_colors,
            linewidth=linewidths,
        )

        ax.set_title(display_name, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8)

        # Tighten y-axis
        vmin, vmax = values.min(), values.max()
        vrange = vmax - vmin
        if vrange > 0 and vrange < 0.15 * vmax:
            margin = vrange * 1.5
            ax.set_ylim(max(0, vmin - margin), vmax + margin * 0.5)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}" if val < 10 else f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

        arrow = "\u2191" if higher_better else "\u2193"
        ax.set_ylabel(
            f"{arrow} {'higher' if higher_better else 'lower'} is better", fontsize=8
        )

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"GEOM Validation Comparison - Epoch {match_epoch} (100 mols, from WandB callbacks)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Compile WandB training curves")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory for output files",
    )
    parser.add_argument(
        "--match_epoch",
        type=int,
        default=15,
        help="Epoch for the apples-to-apples bar chart (default: 15)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Fetching per-epoch metrics from WandB...")
    df = load_all_runs()

    # Save CSV
    csv_path = out / "validation_curves.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path} ({len(df)} rows)")

    # Print epoch-matched summary
    match_epoch = args.match_epoch
    matched = df[df["epoch"] == match_epoch]
    if matched.empty:
        print(f"\nWARNING: No data at epoch {match_epoch}")
    else:
        print(f"\n--- Epoch {match_epoch} validation comparison ---")
        cols = ["model"] + [k for k in BAR_METRICS if k in matched.columns]
        print(matched[cols].to_string(index=False))

    # Plot validation curves
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(df, fig_dir / "validation_curves.png")

    # Plot epoch-matched bar chart
    if not matched.empty:
        plot_epoch_matched(df, match_epoch, fig_dir / "validation_epoch_matched.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
