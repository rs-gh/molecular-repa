#!/usr/bin/env python
"""Compile evaluation results into summary tables and comparison plots.

"Evaluation" metrics = post-hoc generation from final checkpoints (1000 mols, on disk).
"Validation" metrics = computed during training by callbacks (100 mols/epoch, logged to WandB).
See compile_wandb_curves.py for validation metrics.

Usage:
    source .venv/bin/activate
    python evaluation/scripts/compile_results.py evaluation/results/tabasco/geom

Outputs (written to the results directory):
    evaluation_summary.csv       — All evaluation metrics in one table
    evaluation_comparison.png    — Bar charts of key evaluation metrics across models
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Metrics to include in the bar-chart comparison (order matters for layout).
KEY_METRICS = [
    ("validity", "Validity", True),  # (json_key, display_name, higher_is_better)
    ("fcd", "FCD", False),
    ("atom_type_dist", "Atom Type Dist", True),
    ("novelty", "Novelty", True),
    ("diversity", "Diversity", True),
    ("pb_intersection", "PB Intersection", True),
    ("pb_bond_lengths", "PB Bond Lengths", True),
    ("pb_bond_angles", "PB Bond Angles", True),
    ("pb_internal_steric_clash", "PB Steric Clash", True),
]

MODEL_ORDER = [
    "baseline",
    "additive_same",
    "additive_fused",
    "tradeoff_same",
    "tradeoff_fused",
]

COLORS = {
    "baseline": "#4C72B0",
    "additive_same": "#55A868",
    "additive_fused": "#8ECA6E",
    "tradeoff_same": "#C44E52",
    "tradeoff_fused": "#E8907E",
}


def load_results(results_dir: Path) -> dict[str, dict]:
    """Load metrics.json from each model subdirectory."""
    results = {}
    for subdir in sorted(results_dir.iterdir()):
        metrics_file = subdir / "metrics.json"
        if subdir.is_dir() and metrics_file.exists():
            with open(metrics_file) as f:
                results[subdir.name] = json.load(f)
    return results


def build_summary_table(results: dict[str, dict]) -> pd.DataFrame:
    """Build a DataFrame with one row per model, columns = metrics."""
    rows = []
    for name, data in results.items():
        row = {"model": name}
        row["epoch"] = data.get("meta", {}).get("epoch", "")
        row["step"] = data.get("meta", {}).get("global_step", "")
        row["num_mols"] = data.get("generation", {}).get("num_mols", "")
        row["gen_time_s"] = round(data.get("generation", {}).get("time_seconds", 0), 1)
        for k, v in data.get("metrics", {}).items():
            row[k] = round(v, 4) if isinstance(v, float) else v
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by MODEL_ORDER if possible
    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    df["_order"] = df["model"].map(lambda x: order_map.get(x, 999))
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return df


def plot_comparison(df: pd.DataFrame, output_path: Path):
    """Create bar-chart grid comparing key metrics across models."""
    n_metrics = len(KEY_METRICS)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    models = df["model"].tolist()
    x = np.arange(len(models))
    bar_width = 0.6

    for i, (key, display, higher_better) in enumerate(KEY_METRICS):
        ax = axes[i]
        values = df[key].values.astype(float)
        colors = [COLORS.get(m, "#999999") for m in models]

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

        ax.set_title(display, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("_", "\n") for m in models], fontsize=8, rotation=0
        )

        # Adjust y-axis to show differences clearly
        vmin, vmax = values.min(), values.max()
        vrange = vmax - vmin
        if vrange > 0 and vrange < 0.15 * vmax:
            margin = vrange * 1.5
            ax.set_ylim(max(0, vmin - margin), vmax + margin * 0.5)

        # Value labels
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

    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "GEOM Evaluation Comparison — Final Checkpoints (1000 mols)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compile evaluation results")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing per-model result subdirectories",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory")
        sys.exit(1)

    results = load_results(results_dir)
    if not results:
        print(f"No metrics.json files found in {results_dir}/*/")
        sys.exit(1)

    print(f"Found {len(results)} models: {', '.join(results.keys())}")

    # Build and save summary table
    df = build_summary_table(results)
    csv_path = results_dir / "evaluation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved table to {csv_path}")

    # Print to console
    print()
    print(df.to_string(index=False))
    print()

    # Plot comparison
    plot_path = results_dir / "evaluation_comparison.png"
    plot_comparison(df, plot_path)


if __name__ == "__main__":
    main()
