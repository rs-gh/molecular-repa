"""Bar chart comparison of generation metrics for the n=128 sweep.

Reads from results/n128/sweep_results.csv and produces a 1 x len(METRICS)
grid:
    cols  = metric (PDB FID, fS, fJSD, designability, scRMSD, novelty)
    bars  = 4 runs (baseline, REPA L0, REPA L4, REPA L9)

n=128 is the L0/L4/L9 layer ablation, sample-matched at ~19.5M samples per
run. Unlike plot_n128_L4_bs80.py (which is L4-only with two ckpts), this is
a single-checkpoint comparison spanning the layer dimension.

Per-run x-tick labels embed batch-size schedule and samples-seen so the
bar height is interpretable as "metric at ~equal training budget":
    baseline_128  bs=24 fixed         step 800K = 19.20M samples
    repa_l*_128   bs=24->80 at 220K   step 400K = 19.68M samples

Usage:
    python evaluation/proteina/generation/scripts/plot_n128_sweep.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results" / "n128"
FIGURES_DIR = HERE.parent / "figures" / "n128"

# (display_label, colour, bs_schedule_str, samples_M)
# Samples derived in evaluation/proteina/lib/checkpoints.py:
#   baseline:  800K x 24                     = 19.20M
#   repa_l*:   220K x 24 + 180K x 80         = 19.68M
RUNS = {
    "baseline_128": ("Baseline", "#4A90D9", "bs=24", 19.20),
    "repa_l0_128": ("REPA L0", "#55A868", "bs=24->80@220K", 19.68),
    "repa_l4_128": ("REPA L4", "#E74C3C", "bs=24->80@220K", 19.68),
    "repa_l9_128": ("REPA L9", "#C44E52", "bs=24->80@220K", 19.68),
}
RUN_LABELS = [v[0] for v in RUNS.values()]
RUN_COLORS = {v[0]: v[1] for v in RUNS.values()}
KEY_TO_LABEL = {k: v[0] for k, v in RUNS.items()}
RUN_BS = {v[0]: v[2] for v in RUNS.values()}
RUN_SAMPLES = {v[0]: v[3] for v in RUNS.values()}

METRICS = {
    "_res_PDB_FID": ("PDB FID", True, "N=200"),
    "_res_fS_T": ("Fold Score (Topo)", False, "N=200"),
    "_res_PDB_fJSD_T": ("fJSD (Topology)", True, "N=200"),
    "_res_designability_rate": ("Designability", False, "N=100"),
    "_res_scRMSD_mean": ("scRMSD (A)", True, "N=100"),
    "_res_novelty_rate": ("Novelty", False, "N=200"),
}


def load_rows() -> list[dict]:
    csv = RESULTS_DIR / "sweep_results.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No sweep results at {csv}")
    df = pd.read_csv(csv)
    df = df[df["run"].isin(RUNS)]
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]
    return df.to_dict("records")


def organize(rows: list[dict]) -> tuple[dict, dict]:
    """Return ({run_label: {metric: value}}, {run_label: step})."""
    out: dict[str, dict] = {}
    steps: dict[str, int] = {}
    for r in rows:
        label = KEY_TO_LABEL[r["run"]]
        out[label] = {
            m: (
                None
                if (v := r.get(m)) is None or (isinstance(v, float) and math.isnan(v))
                else v
            )
            for m in METRICS
        }
        steps[label] = int(r["step"])
    return out, steps


def _xtick_label(label: str, step: int | None) -> str:
    """Compact two-line tick: '<run>\nstep <K>K | <samples>M smp'.

    Per-run bs schedule is documented in the figure subtitle (it's
    near-uniform: baseline bs=24, REPA bs=24->80@220K), so the tick stays
    short to fit one per bar.
    """
    samples = RUN_SAMPLES[label]
    if step is None:
        return f"{label}\nlast | {samples:.1f}M smp"
    return f"{label}\nstep {step // 1000}K | {samples:.1f}M smp"


def plot(data: dict, steps: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.2 * n_metrics, 5.6))
    x = np.arange(len(RUN_LABELS))

    for col, (mkey, (mlabel, lower_better, nnote)) in enumerate(METRICS.items()):
        ax = axes[col]

        vals, missing = [], []
        for lbl in RUN_LABELS:
            v = data.get(lbl, {}).get(mkey)
            if v is None:
                vals.append(0.0)
                missing.append(True)
            else:
                vals.append(float(v))
                missing.append(False)
        colors = [RUN_COLORS[lbl] for lbl in RUN_LABELS]

        bars = ax.bar(
            x,
            vals,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            width=0.65,
        )

        max_v = max((v for v, m in zip(vals, missing) if not m), default=1.0) or 1.0
        for bar, v, is_missing in zip(bars, vals, missing):
            if is_missing:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    max_v * 0.05,
                    "pending",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="gray",
                    style="italic",
                )
            else:
                fmt = f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + max_v * 0.02,
                    fmt,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        finite = [(i, v) for i, (v, m) in enumerate(zip(vals, missing)) if not m]
        if finite:
            best_i = (
                min(finite, key=lambda t: t[1])[0]
                if lower_better
                else max(finite, key=lambda t: t[1])[0]
            )
            bars[best_i].set_edgecolor("gold")
            bars[best_i].set_linewidth(2.5)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [_xtick_label(lbl, steps.get(lbl)) for lbl in RUN_LABELS],
            fontsize=7.5,
            rotation=20,
            ha="right",
        )
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        direction = "v better" if lower_better else "^ better"
        ax.set_title(
            f"{mlabel}\n({nnote}, {direction})", fontsize=10, fontweight="bold"
        )

    legend_patches = [
        mpatches.Patch(color=RUN_COLORS[lbl], label=lbl) for lbl in RUN_LABELS
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(RUN_LABELS),
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.suptitle(
        "n=128 sweep - generation metrics at sample-matched checkpoints (~19.5M samples)\n"
        "Baseline trained at fixed bs=24; REPA students switched bs=24->80 at step 220K. "
        "Each bar is one EMA checkpoint; tick labels show per-run bs/step/samples.\n"
        "Sampling per checkpoint: nres_lens=[128] x 200 samples/length = 200 PDBs "
        "(FID/fS/fJSD/novelty use all 200; designability/scRMSD subsample N=100).",
        fontsize=10,
        fontweight="bold",
        y=0.99,
    )
    out = FIGURES_DIR / "n128_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    rows = load_rows()
    data, steps = organize(rows)
    plot(data, steps)
