"""Bar chart comparison of generation metrics across sizes and runs.

Reads from the sweep JSONL outputs (results/n128/, n256/, n512_sm/) and
produces a 2x3 grid: rows = FID_PDB / fS_C, columns = n=128/256/512.

Usage:
    python evaluation/proteina/generation/scripts/plot_sample_matched.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE.parent / "results"
FIGURES_DIR = HERE.parent / "figures"

# ── Data ──────────────────────────────────────────────────────────────────────

SIZES = {
    "n128": {"label": "n=128", "samples": "19.5M samples", "dir": "n128"},
    "n256": {"label": "n=256", "samples": "7.0M samples", "dir": "n256"},
    "n512_sm": {"label": "n=512", "samples": "3.0M samples", "dir": "n512_sm"},
}

# Canonical display order and style per run
RUNS = {
    "baseline_128": ("Baseline", "#4A90D9"),
    "repa_l0_128": ("REPA L0", "#55A868"),
    "repa_l4_128": ("REPA L4", "#E74C3C"),
    "repa_l9_128": ("REPA L9", "#F39C12"),
    "baseline_256": ("Baseline", "#4A90D9"),
    "repa_l0_256": ("REPA L0", "#55A868"),
    "repa_l4_256": ("REPA L4", "#E74C3C"),
    "repa_l9_256": ("REPA L9", "#F39C12"),
    "baseline_512_sm": ("Baseline", "#4A90D9"),
    "repa_l0_512_sm": ("REPA L0", "#55A868"),
    "repa_l4_512_sm": ("REPA L4", "#E74C3C"),
    "repa_l9_512_sm": ("REPA L9", "#F39C12"),
}

RUN_ORDER = ["Baseline", "REPA L0", "REPA L4", "REPA L9"]
RUN_COLORS = {
    "Baseline": "#4A90D9",
    "REPA L0": "#55A868",
    "REPA L4": "#E74C3C",
    "REPA L9": "#F39C12",
}

METRICS = {
    "_res_PDB_FID": ("PDB FID", True),  # lower is better
    "_res_fS_C": ("Fold Score (Class)", False),  # higher is better
}


def load_results() -> dict:
    """Return {size_key: {run_label: {metric: value}}}."""
    data = {}
    for size_key, cfg in SIZES.items():
        jsonl = RESULTS_ROOT / cfg["dir"] / "sweep_results.jsonl"
        size_data: dict[str, dict] = {}
        if jsonl.exists():
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if "error" in r:
                        continue
                    run_key = r["run"]
                    if run_key not in RUNS:
                        continue
                    label = RUNS[run_key][0]
                    size_data[label] = {m: r.get(m) for m in METRICS}
        data[size_key] = size_data
    return data


def plot(data: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    n_metrics = len(METRICS)
    n_sizes = len(SIZES)
    # Layout matches representation fig_grid_*: rows = model size, cols = metric.
    fig, axes = plt.subplots(
        n_sizes,
        n_metrics,
        figsize=(4.5 * n_metrics, 3.3 * n_sizes),
        sharey="col",
    )

    bar_width = 0.18
    x_positions = np.arange(len(RUN_ORDER))

    for row, (size_key, size_cfg) in enumerate(SIZES.items()):
        size_data = data[size_key]

        for col, (metric_key, (metric_label, lower_better)) in enumerate(
            METRICS.items()
        ):
            ax = axes[row, col]

            vals = []
            colors = []
            missing = []
            for run_label in RUN_ORDER:
                if (
                    run_label in size_data
                    and size_data[run_label].get(metric_key) is not None
                ):
                    vals.append(size_data[run_label][metric_key])
                    missing.append(False)
                else:
                    vals.append(0)
                    missing.append(True)
                colors.append(RUN_COLORS[run_label])

            bars = ax.bar(
                x_positions,
                vals,
                width=bar_width * 3.5,
                color=colors,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )

            # Annotate bars
            max_val = max(v for v in vals if v > 0) if any(v > 0 for v in vals) else 1
            for bar, val, is_missing, run_label in zip(bars, vals, missing, RUN_ORDER):
                if is_missing:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        max_val * 0.05,
                        "pending",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="gray",
                        style="italic",
                    )
                else:
                    fmt = f"{val:.0f}" if lower_better else f"{val:.2f}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + max_val * 0.02,
                        fmt,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

            # Highlight best bar
            finite_vals = [
                (i, v) for i, (v, m) in enumerate(zip(vals, missing)) if not m
            ]
            if finite_vals:
                best_i = (
                    min(finite_vals, key=lambda x: x[1])[0]
                    if lower_better
                    else max(finite_vals, key=lambda x: x[1])[0]
                )
                bars[best_i].set_edgecolor("gold")
                bars[best_i].set_linewidth(2.5)

            ax.set_xticks(x_positions)
            ax.set_xticklabels(RUN_ORDER, fontsize=9)
            ax.grid(axis="y", alpha=0.3, zorder=0)
            ax.set_axisbelow(True)

            if row == 0:
                direction = "↓ better" if lower_better else "↑ better"
                ax.set_title(
                    f"{metric_label} ({direction})",
                    fontsize=11,
                    fontweight="bold",
                )

            if col == 0:
                ax.set_ylabel(
                    f"{size_cfg['label']}\n{size_cfg['samples']}\n200 generated",
                    fontsize=10,
                    fontweight="bold",
                )

    # Legend
    legend_patches = [mpatches.Patch(color=RUN_COLORS[r], label=r) for r in RUN_ORDER]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Generation quality — sample-matched comparison\n(single training-cap length per size)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "fig_grid_sample_matched.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    data = load_results()
    plot(data)
