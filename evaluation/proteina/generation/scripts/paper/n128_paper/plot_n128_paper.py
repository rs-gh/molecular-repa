"""Bar chart comparison for the n=128 paper-protocol sweeps.

Reads from results/n128_paper_layer/, n128_paper_encoder/, and
n128_paper_bs_lr/ sweep_results.csv and produces a 3 x len(METRICS) grid:
    rows  = ablation (layer / encoder / bs+lr)
    cols  = metric (PDB FID, AFDB FID, fS_T, fJSD_T, designability, scRMSD,
                    diversity, novelty)
    bars  = runs within that ablation

Paper protocol (n=128):
    Generation: 500 PDBs at L∈{50,75,100,125} × 125/length
    FID/fJSD/fS: full 500-PDB pool
    Designability: 200 PDBs (50/length × 4 lengths)
    Diversity (and novelty if available): designable subset of those 200

The active metric filter is "fid,designability,diversity", so novelty rows
will usually render as "pending" on freshly-evaluated checkpoints; older rows
that pre-date the filter retain populated novelty fields.

Missing rows render as "pending" placeholders, so the plot is useful at any
point during a sweep — partial data still renders.

Usage:
    python evaluation/proteina/generation/scripts/plot_n128_paper.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent.parent
GENERATION_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from utils._results_io import load_sweep_rows  # noqa: E402

RESULTS_ROOT = GENERATION_ROOT / "results" / "paper"
FIGURES_DIR = GENERATION_ROOT / "figures" / "paper" / "n128_paper"

# Per-ablation config: dir name + ordered run list. Each run tuple is
# (run_name, display_label, color). Colours group conceptually:
#   Baselines: blue family
#   REPA L0/L4/L9: green/red/orange (matching n256 plot)
#   Encoder ablation: each target gets a distinct hue
#   bs/lr ablation: baseline tints (blue) vs REPA tints (red); intensity
#                   tracks step / config strength
ABLATIONS = {
    "layer": {
        "label": "Layer ablation\n(L0/L4/L9 vs baseline, bs=80)",
        "dir": "n128_paper_layer",
        "runs": [
            ("baseline_128_bs80_step200k", "Baseline", "#4A90D9"),
            ("repa_l0_128_bs80_steplast", "REPA L0", "#55A868"),
            ("repa_l4_128_bs80_step200k", "REPA L4", "#E74C3C"),
            ("repa_l9_128_bs80_steplast", "REPA L9", "#F39C12"),
        ],
    },
    "encoder": {
        "label": "Encoder ablation\n(REPA L4 with 6 target encoders)",
        "dir": "n128_paper_encoder",
        "runs": [
            ("baseline_128_bs80_step200k", "Baseline", "#4A90D9"),
            ("repa_l4_128_bs80_step200k", "CA-GearNet", "#E74C3C"),
            ("repa_l4_128_random_step200k", "GearNet random", "#7F7F7F"),
            ("repa_l4_128_pw_structure_step100k", "PW-Structure", "#9B59B6"),
            ("repa_l4_128_pw_torsional_step100k", "PW-Torsional", "#C44893"),
            ("repa_mpnn_l4_128_bs80_step200k", "ProteinMPNN", "#8B4513"),
            ("repa_esm_l4_128_step200k", "ESM2", "#1ABC9C"),
        ],
    },
    "bs_lr": {
        "label": "Batch size + LR ablation\n(bs ∈ {24,80} × lr ∈ {1×,3×} × ±REPA)",
        "dir": "n128_paper_bs_lr",
        "runs": [
            ("baseline_128_bs24_step200k", "BL bs24 200k", "#A6CEE3"),
            ("baseline_128_bs24_step400k", "BL bs24 400k", "#4A90D9"),
            ("baseline_128_bs80_step200k", "BL bs80 200k", "#1F4E79"),
            ("baseline_128_bs80_lr3x_step200k", "BL bs80 lr3× 200k", "#5E3C99"),
            ("repa_l4_128_bs24_step200k", "L4 bs24 200k", "#FBB4AE"),
            ("repa_l4_128_bs24_step400k", "L4 bs24 400k", "#E74C3C"),
            ("repa_l4_128_bs80_step200k", "L4 bs80 200k", "#8B0000"),
            ("repa_l4_128_bs80_lr3x_steplast", "L4 bs80 lr3× last", "#C44893"),
        ],
    },
}

# (display label, lower_is_better). N-note is shared across all paper-protocol
# panels — generation pool is 500, designability pool is 200, diversity/novelty
# run on the designable subset.
# TODO: re-add novelty once we settle on a meaningful metric. Centroid novelty
# (max-TM vs ~4400 training centroids, threshold 0.5) saturates near 1.0 for any
# decent generator and doesn't separate runs; foldseek novelty was never
# persisting (silent exception in compute_novelty_foldseek). Decide between:
# (a) keep centroid but drop the threshold and report max_tm distribution,
# (b) fix the foldseek path, or (c) adopt a different similarity reference.
METRICS = {
    "_res_PDB_FID": ("PDB FID", True),
    "_res_AFDB_FID": ("AFDB FID", True),
    "_res_fS_T": ("Fold Score (Topo)", False),
    "_res_PDB_fJSD_T": ("PDB fJSD (Topo)", True),
    "_res_designability_rate": ("Designability", False),
    "_res_scRMSD_mean": ("scRMSD mean (Å)", True),
    "_res_diversity_clusters_mean": ("Diversity (clusters)", False),
    "_res_diversity_pairwise_tm_mean": ("Diversity (pairwise TM)", True),
}

N_NOTES = {
    "_res_PDB_FID": "N=500",
    "_res_AFDB_FID": "N=500",
    "_res_fS_T": "N=500",
    "_res_PDB_fJSD_T": "N=500",
    "_res_designability_rate": "N=200",
    "_res_scRMSD_mean": "N=200",
    "_res_diversity_clusters_mean": "designable",
    "_res_diversity_pairwise_tm_mean": "designable",
}


# Rows whose designability subset was below this threshold ran on a corrupted
# length composition (PDB index +125 shift bug — see investigation 2026-05-07).
# Their FID/fJSD/fS_* columns are still valid (those metrics aggregate over the
# full 500-PDB pool, length-agnostic), but designability/scRMSD/plddt/diversity/
# novelty are biased and must be hidden until the dirs are re-evaluated.
MIN_DESIGNABILITY_N = 175

# Metric prefixes that are invalidated by the shift bug (computed on the
# per-length subset). FID-family metrics keep their values.
DOWNSTREAM_METRIC_PREFIXES = (
    "_res_designability_",
    "_res_scRMSD_",
    "_res_plddt_",
    "_res_tm_score_",
    "_res_diversity_",
    "_res_novelty_",
)


def _scrub_corrupt_designability(row: dict) -> dict:
    """Drop downstream metrics on rows whose designability ran on a partial
    or shifted subset (designability_n < MIN_DESIGNABILITY_N). The FID family
    is preserved."""
    n = row.get("_res_designability_n")
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return row
    if int(n) >= MIN_DESIGNABILITY_N:
        return row
    cleaned = dict(row)
    for k in list(cleaned.keys()):
        if any(k.startswith(p) for p in DOWNSTREAM_METRIC_PREFIXES):
            cleaned[k] = float("nan")
    return cleaned


def load_ablation_rows(ablation_dir: Path) -> dict[str, dict]:
    """Return {run_name: row_dict} for an ablation's sweep_results.jsonl.

    Rows with sub-threshold designability_n have their downstream metrics
    NaN'd (FID columns retained). Returns empty dict if jsonl is missing.
    """
    rows = load_sweep_rows(ablation_dir / "sweep_results.jsonl")
    return {r["run"]: _scrub_corrupt_designability(r) for r in rows}


def _val(row: dict | None, mkey: str) -> float | None:
    """Pull a metric value from a row, mapping NaN/None to None."""
    if row is None:
        return None
    v = row.get(mkey)
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return float(v)


def _xtick_label(label: str, step: int | None) -> str:
    """Two-line tick: '<run>\\nstep <K>K' (or 'last' if step unknown)."""
    if step is None:
        return f"{label}\n(pending)"
    return f"{label}\nstep {step // 1000}K"


def plot() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n_metrics = len(METRICS)
    n_ablations = len(ABLATIONS)
    fig, axes = plt.subplots(
        n_ablations,
        n_metrics,
        figsize=(3.4 * n_metrics, 4.2 * n_ablations),
        squeeze=False,
    )

    # Pre-load all ablation data
    ablation_data: dict[str, dict[str, dict]] = {}
    for akey, acfg in ABLATIONS.items():
        ablation_data[akey] = load_ablation_rows(RESULTS_ROOT / acfg["dir"])

    # One row per ablation, one column per metric
    for arow, (akey, acfg) in enumerate(ABLATIONS.items()):
        rows = ablation_data[akey]
        runs = acfg["runs"]
        x = np.arange(len(runs))
        colors = [r[2] for r in runs]
        tick_labels = [
            _xtick_label(
                label,
                int(rows[run_name]["step"])
                if run_name in rows and "step" in rows[run_name]
                else None,
            )
            for run_name, label, _color in runs
        ]

        for acol, (mkey, (mlabel, lower_better)) in enumerate(METRICS.items()):
            ax = axes[arow][acol]
            vals: list[float] = []
            missing: list[bool] = []
            for run_name, _label, _color in runs:
                v = _val(rows.get(run_name), mkey)
                vals.append(0.0 if v is None else v)
                missing.append(v is None)

            bars = ax.bar(
                x,
                vals,
                color=colors,
                edgecolor="white",
                linewidth=0.8,
                width=0.7,
                zorder=3,
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
                        fontsize=7,
                        fontweight="bold",
                    )

            # Highlight best within this panel
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
            ax.set_xticklabels(tick_labels, fontsize=6.5, rotation=0)
            ax.grid(axis="y", alpha=0.3, zorder=0)
            ax.set_axisbelow(True)

            if arow == 0:
                direction = "↓ better" if lower_better else "↑ better"
                ax.set_title(
                    f"{mlabel}\n({N_NOTES[mkey]}, {direction})",
                    fontsize=9,
                    fontweight="bold",
                )

            if acol == 0:
                ax.set_ylabel(acfg["label"], fontsize=9, fontweight="bold")

    # Legend: one patch per unique (label, color) across all ablations
    seen: set[tuple[str, str]] = set()
    legend_entries: list[tuple[str, str]] = []
    for acfg in ABLATIONS.values():
        for _run_name, label, color in acfg["runs"]:
            key = (label, color)
            if key not in seen:
                seen.add(key)
                legend_entries.append(key)

    handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in legend_entries]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 6),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.suptitle(
        "n=128 paper-protocol sweep — generation metrics across 3 ablations\n"
        "Pool: 500 PDBs at L∈{50,75,100,125} × 125; designability on 50/L × 4 lengths = 200; "
        "diversity on designable subset, reported both ways: clusters (higher = more diverse) "
        "and pairwise TM (lower = more diverse). Novelty omitted (TODO: re-add).\n"
        f"FID/fJSD/fS_* shown for all populated rows. Designability/scRMSD/diversity hidden "
        f"when designability_n < {MIN_DESIGNABILITY_N} (PDB-index-shift bug; values are "
        f"length-biased until re-eval).\n"
        "Best per panel highlighted in gold.",
        fontsize=10,
        fontweight="bold",
        y=0.99,
    )
    out = FIGURES_DIR / "n128_paper_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    plot()
