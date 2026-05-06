"""Bar chart comparison of generation metrics for the n=128 L4 bs=80 sweep.

Reads from results/n128_L4_bs80/sweep_results.jsonl and produces a
2 x len(METRICS) grid:
    rows  = step bucket (mid ~8M / end ~16M samples)
    cols  = metric (PDB FID, fS, fJSD, designability, scRMSD)
    bars  = 4 runs (baseline, REPA L4, REPA L4 lr3x, REPA L4 random)

Mirrors plot_sample_matched.py but groups by step instead of model size.

Renamed 2026-05-06 from plot_n128_bs80_sweep.py -> plot_n128_L4_bs80.py to
make the layer-4-only scope explicit (cf. plot_n128_sweep.py, which is the
L0/L4/L9 layer ablation).

Usage:
    python evaluation/proteina/generation/scripts/plot_n128_L4_bs80.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results" / "n128_L4_bs80"
FIGURES_DIR = HERE.parent / "figures" / "n128_L4_bs80"

# 4 runs in canonical display order; colour scheme matches plot_sample_matched.py
RUNS = {
    "baseline_128_bs80": ("Baseline", "#4A90D9"),
    "repa_l4_128_bs80": ("REPA L4", "#E74C3C"),
    "repa_l4_128_bs80_lr3x": ("REPA L4 (lr3x)", "#F39C12"),
    "repa_l4_128_random": ("REPA L4 (random)", "#55A868"),
}
RUN_ORDER = list(RUNS.values())  # [(label, colour), ...]
RUN_LABELS = [lbl for lbl, _ in RUN_ORDER]
RUN_COLORS = {lbl: c for lbl, c in RUN_ORDER}
KEY_TO_LABEL = {k: v[0] for k, v in RUNS.items()}

# Two step buckets. lr3x's "end" lands at step ~165K (last-EMA, ~13M samples)
# instead of 200K (16M); plotter accepts any step as long as it's not the
# 100K mid bucket.
MID_STEP = 100000
SAMPLES_LABEL = {
    "mid": "8M samples (step 100K)",
    "end": "~16M samples (step 200K / lr3x last)",
}

METRICS = {
    "_res_PDB_FID": ("PDB FID", True, "N=200"),
    "_res_fS_T": ("Fold Score (Topo)", False, "N=200"),
    "_res_PDB_fJSD_T": ("fJSD (Topology)", True, "N=200"),
    "_res_designability_rate": ("Designability", False, "N=100"),
    "_res_scRMSD_mean": ("scRMSD (A)", True, "N=100"),
    "_res_novelty_rate": ("Novelty", False, "N=200"),
}


def load_rows() -> list[dict]:
    jsonl = RESULTS_DIR / "sweep_results.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"No sweep results at {jsonl}")
    rows = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r or r.get("run") not in RUNS:
                continue
            r["_step_bucket"] = "mid" if int(r["step"]) == MID_STEP else "end"
            rows.append(r)
    return rows


def organize(rows: list[dict]) -> tuple[dict, dict]:
    """Return ({bucket: {run_label: {metric: value}}},
    {bucket: {run_label: step}})."""
    out: dict[str, dict[str, dict]] = {"mid": {}, "end": {}}
    steps: dict[str, dict[str, int]] = {"mid": {}, "end": {}}
    for r in rows:
        bucket = r["_step_bucket"]
        label = KEY_TO_LABEL[r["run"]]
        out[bucket][label] = {m: r.get(m) for m in METRICS}
        steps[bucket][label] = int(r["step"])
    return out, steps


def _xtick_label(label: str, step: int | None) -> str:
    """Two-line tick: '<run>\nstep <K>K | <samples>M smp'.

    All n128_L4_bs80 runs train at bs=80 throughout, so samples = step * 80.
    """
    if step is None:
        return label
    samples_M = step * 80 / 1e6
    return f"{label}\nstep {step // 1000}K | {samples_M:.1f}M smp"


def plot(data: dict, steps: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(2, n_metrics, figsize=(4.2 * n_metrics, 7.8), sharey="col")
    x = np.arange(len(RUN_LABELS))

    for row, bucket in enumerate(["mid", "end"]):
        bucket_data = data[bucket]

        for col, (mkey, (mlabel, lower_better, nnote)) in enumerate(METRICS.items()):
            ax = axes[row, col]

            vals, missing = [], []
            for lbl in RUN_LABELS:
                v = bucket_data.get(lbl, {}).get(mkey)
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

            # Highlight best
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
                [_xtick_label(lbl, steps[bucket].get(lbl)) for lbl in RUN_LABELS],
                fontsize=7.5,
                rotation=20,
                ha="right",
            )
            ax.grid(axis="y", alpha=0.3, zorder=0)
            ax.set_axisbelow(True)

            if row == 0:
                direction = "v better" if lower_better else "^ better"
                ax.set_title(
                    f"{mlabel}\n({nnote}, {direction})", fontsize=10, fontweight="bold"
                )
            if col == 0:
                ax.set_ylabel(SAMPLES_LABEL[bucket], fontsize=10, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=RUN_COLORS[lbl], label=lbl) for lbl in RUN_LABELS
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(RUN_LABELS),
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "n=128 L4 bs=80 sweep - generation metrics at sample-matched checkpoints\n"
        "Sampling per checkpoint: nres_lens=[128] x 200 samples/length = 200 PDBs "
        "(FID/fS/fJSD/novelty use all 200; designability/scRMSD subsample N=100).",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "n128_L4_bs80.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    rows = load_rows()
    data, steps = organize(rows)
    plot(data, steps)
