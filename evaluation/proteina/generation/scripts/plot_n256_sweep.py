"""Bar chart comparison of generation metrics for the n=256 sweep.

Reads from results/n256/sweep_results.csv (CSV preferred over JSONL because
the n256 JSONL is a partial snapshot missing designability / novelty
columns) and produces a 1 x len(METRICS) grid:
    cols  = metric (PDB FID, fS, fJSD, designability, scRMSD, novelty)
    bars  = 4 runs (baseline, REPA L0, REPA L4, REPA L9)

All runs sit at step 400K but the bs=12->24 transition fired at different
points across runs (cluster restart timing), so they are NOT exactly
sample-matched: epoch counters at step 400K are 21/22/25/26 -> ~5.6M /
5.9M / 6.7M / 7.0M samples (PDB train n=256 subset = 267,789 chains).
Per-bar x-tick labels surface this so the bar height is interpretable.

Usage:
    python evaluation/proteina/generation/scripts/plot_n256_sweep.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results" / "n256"
FIGURES_DIR = HERE.parent / "figures" / "n256"

# (display_label, colour, samples_M_at_step_400K).
# Samples = epoch_at_step_400K * 267,789 (PDB train, 50<=L<=256). Epochs
# are pulled from the EMA ckpt filenames in results/n256/sweep_results.csv:
#   baseline_256: epoch 21 -> 21 * 267,789 = 5.62M
#   repa_l4_256:  epoch 22 ->                 5.89M
#   repa_l9_256:  epoch 25 ->                 6.69M
#   repa_l0_256:  epoch 26 ->                 6.96M
RUNS = {
    "baseline_256": ("Baseline", "#4A90D9", 5.6),
    "repa_l0_256": ("REPA L0", "#55A868", 7.0),
    "repa_l4_256": ("REPA L4", "#E74C3C", 5.9),
    "repa_l9_256": ("REPA L9", "#C44E52", 6.7),
}
RUN_LABELS = [v[0] for v in RUNS.values()]
RUN_COLORS = {v[0]: v[1] for v in RUNS.values()}
RUN_SAMPLES = {v[0]: v[2] for v in RUNS.values()}
KEY_TO_LABEL = {k: v[0] for k, v in RUNS.items()}

METRICS = {
    "_res_PDB_FID": ("PDB FID", True, "N=200"),
    "_res_fS_T": ("Fold Score (Topo)", False, "N=200"),
    "_res_PDB_fJSD_T": ("fJSD (Topology)", True, "N=200"),
    "_res_designability_rate": ("Designability", False, "N=100"),
    "_res_scRMSD_mean": ("scRMSD (A)", True, "N=100"),
    "_res_novelty_rate": ("Novelty", False, "N=240"),
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
    """Two-line tick: '<run>\nstep <K>K | <samples>M smp'."""
    samples = RUN_SAMPLES[label]
    if step is None:
        return f"{label}\nlast | ~{samples:.1f}M smp"
    return f"{label}\nstep {step // 1000}K | ~{samples:.1f}M smp"


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
        "n=256 sweep - generation metrics (all runs at step 400K, NOT sample-matched)\n"
        "All 4 runs share dataset pdb_lmdb_256 (bs=12->24 bumped 2026-04-19), "
        "but the bs ramp fired at different relative steps; samples seen = epoch x 267,789.\n"
        "Sampling per checkpoint: nres_lens=[256] x 200 samples/length, gen_bs=80 rounds up "
        "via split_nlens -> 240 PDBs (FID/fS/fJSD/novelty use all 240; designability/scRMSD subsample N=100).",
        fontsize=10,
        fontweight="bold",
        y=0.99,
    )
    out = FIGURES_DIR / "n256_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    rows = load_rows()
    data, steps = organize(rows)
    plot(data, steps)
