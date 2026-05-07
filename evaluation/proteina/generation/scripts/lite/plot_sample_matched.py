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
GENERATION_ROOT = HERE.parent.parent
RESULTS_ROOT = GENERATION_ROOT / "results" / "lite"
FIGURES_DIR = GENERATION_ROOT / "figures" / "lite"

# -- Data ----------------------------------------------------------------------

SIZES = {
    "n128": {"label": "n=128", "dir": "n128_lite"},
    "n256": {"label": "n=256", "dir": "n256_lite"},
    "n512_sm": {"label": "n=512", "dir": "n512_sm_lite"},
}

# Per-cell samples_M lookup (used for the per-bar x-tick). Steps are read from
# the JSONL at runtime; samples_M is hard-coded because it depends on the
# run-specific bs schedule (not derivable from step alone):
#   n=128: baseline bs=24 fixed; repa_l* bs=24->80@220K
#          baseline 800K -> 19.20M, repa_l* 400K -> 19.68M
#   n=256: bs=12->24 ramp fired at different relative steps per run, so
#          samples = epoch_at_step_400K * 267,789 (PDB train n<=256 subset).
#          epochs from CSV: baseline 21, l4 22, l9 25, l0 26
#   n=512: baseline bs=6 fixed -> 500K = 3.00M; repa_l* bs=4 fixed -> 750K = 3.00M
SAMPLES_M = {
    ("n128", "Baseline"): 19.2,
    ("n128", "REPA L0"): 19.7,
    ("n128", "REPA L4"): 19.7,
    ("n128", "REPA L9"): 19.7,
    ("n256", "Baseline"): 5.6,
    ("n256", "REPA L0"): 7.0,
    ("n256", "REPA L4"): 5.9,
    ("n256", "REPA L9"): 6.7,
    ("n512_sm", "Baseline"): 3.0,
    ("n512_sm", "REPA L0"): 3.0,
    ("n512_sm", "REPA L4"): 3.0,
    ("n512_sm", "REPA L9"): 3.0,
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

#! Tuples are (label, lower_is_better, n_note).
#! FID / fS / fJSD use every generated PDB (200 for n=128/n=512_sm, 240 for n=256);
#! designability / scRMSD use a fixed 100-PDB subset.
METRICS = {
    "_res_PDB_FID": ("PDB FID", True, "N=200"),
    "_res_fS_C": ("Fold Score (Class)", False, "N=200"),
    "_res_PDB_fJSD_C": ("fJSD (Class)", True, "N=200"),
    "_res_PDB_fJSD_A": ("fJSD (AA)", True, "N=200"),
    "_res_PDB_fJSD_T": ("fJSD (Topology)", True, "N=200"),
    "_res_designability_rate": ("Designability", False, "N=100"),
    "_res_scRMSD_mean": ("scRMSD (A)", True, "N=100"),
}

#! eval_output is two levels above HERE (scripts -> generation -> proteina -> evaluation -> repo-root)
REPO_ROOT = GENERATION_ROOT.parent.parent.parent
EVAL_OUTPUT_DIR = REPO_ROOT / "eval_output"


def _load_per_run_csv_metrics(config_name: str, run_key: str, step) -> dict:
    """Pull metrics from the per-run evaluate.py CSV when JSONL doesn't have them.

    Designability columns are populated post-hoc via eval_designability_only.sh
    after the sweep JSONL was already written; this reads the merged CSV so
    the plot picks up backfilled columns without mutating the JSONL.
    """
    import pandas as pd

    config_slug = config_name.replace("/", "_")
    suffix = f"sweep_{run_key}_step_{step}"
    csv_path = (
        EVAL_OUTPUT_DIR
        / f"{config_slug}_{suffix}"
        / f"results_{config_slug}_{suffix}_fid.csv"
    )
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {}
    out = {}
    for m in METRICS:
        if m in df.columns:
            v = df[m].iloc[0]
            if pd.notna(v):
                out[m] = float(v)
    return out


def load_results() -> tuple[dict, dict]:
    """Return ({size_key: {run_label: {metric: value}}},
    {size_key: {run_label: step}})."""
    data = {}
    steps_by_size: dict[str, dict[str, int]] = {}
    for size_key, cfg in SIZES.items():
        jsonl = RESULTS_ROOT / cfg["dir"] / "sweep_results.jsonl"
        size_data: dict[str, dict] = {}
        size_steps: dict[str, int] = {}
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
                    metrics = {m: r.get(m) for m in METRICS}
                    #! Backfill missing metrics (e.g. designability) from per-run CSV
                    missing = [m for m, v in metrics.items() if v is None]
                    if missing and r.get("config_name"):
                        csv_metrics = _load_per_run_csv_metrics(
                            r["config_name"], run_key, r["step"]
                        )
                        for m in missing:
                            if m in csv_metrics:
                                metrics[m] = csv_metrics[m]
                    size_data[label] = metrics
                    size_steps[label] = int(r["step"])
        data[size_key] = size_data
        steps_by_size[size_key] = size_steps
    return data, steps_by_size


def _xtick_label(size_key: str, run_label: str, step: int | None) -> str:
    """Two-line tick: '<run>\nstep <K>K | <samples>M smp'."""
    samples = SAMPLES_M.get((size_key, run_label))
    if step is None or samples is None:
        return run_label
    return f"{run_label}\nstep {step // 1000}K | {samples:.1f}M smp"


def plot(data: dict, steps_by_size: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    n_metrics = len(METRICS)
    n_sizes = len(SIZES)
    # Layout matches representation fig_grid_*: rows = model size, cols = metric.
    fig, axes = plt.subplots(
        n_sizes,
        n_metrics,
        figsize=(4.8 * n_metrics, 4.0 * n_sizes),
        sharey="col",
    )

    bar_width = 0.18
    x_positions = np.arange(len(RUN_ORDER))

    for row, (size_key, size_cfg) in enumerate(SIZES.items()):
        size_data = data[size_key]
        size_steps = steps_by_size.get(size_key, {})

        for col, (metric_key, (metric_label, lower_better, n_note)) in enumerate(
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
                    #! Small values (designability, scRMSD) need 2dp; large (FID) don't
                    fmt = f"{val:.2f}" if abs(val) < 10 else f"{val:.0f}"
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
            ax.set_xticklabels(
                [_xtick_label(size_key, r, size_steps.get(r)) for r in RUN_ORDER],
                fontsize=7.5,
                rotation=20,
                ha="right",
            )
            ax.grid(axis="y", alpha=0.3, zorder=0)
            ax.set_axisbelow(True)

            if row == 0:
                direction = "v better" if lower_better else "^ better"
                ax.set_title(
                    f"{metric_label} ({n_note}, {direction})",
                    fontsize=11,
                    fontweight="bold",
                )

            if col == 0:
                ax.set_ylabel(
                    size_cfg["label"],
                    fontsize=11,
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

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.suptitle(
        "Generation quality - cross-size comparison\n"
        "Per-bar x-tick: step | samples-seen. n=128 and n=512_sm are sample-matched within each row; "
        "n=256 is NOT (bs=12->24 ramp fired at different relative steps).\n"
        "Sampling per checkpoint: n=128/512 -> 200 PDBs (nres=[L] x 200/length); "
        "n=256 -> 240 PDBs (nres=[256] x 200/length, gen_bs=80 rounds up via split_nlens). "
        "Designability/scRMSD subsample N=100.",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    out = FIGURES_DIR / "fig_grid_sample_matched.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    data, steps_by_size = load_results()
    plot(data, steps_by_size)
