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

from evaluation.proteina.lib.plot_labels import compose_title_suffix  # noqa: E402

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
    "pretrained": {
        "label": "External reference\n(NVIDIA NGC DFS-60M v1.3, 12L)",
        "dir": "n128_paper_pretrained",
        "runs": [
            ("pretrained_dfs_60m_n128_paper", "Pretrained DFS-60M", "#9467bd"),
        ],
    },
    "layer": {
        "label": "Layer ablation\n(L0/L4/L9 vs baseline, bs=80, step=200K)",
        "dir": "n128_paper_layer",
        "runs": [
            ("baseline_128_bs80_step200k", "Baseline", "#4A90D9"),
            ("repa_l0_128_bs80_step200k", "REPA L0", "#55A868"),
            ("repa_l4_128_bs80_step200k", "REPA L4", "#E74C3C"),
            ("repa_l9_128_bs80_step200k", "REPA L9", "#F39C12"),
        ],
    },
    "layer_per_residue": {
        "label": "Layer ablation — per_residue\n(mixed bs=24→80 @ step 220k, fair cross-layer)",
        "dir": "n128_paper_layer_per_residue",
        "runs": [
            ("baseline_128_bs24_step400k", "Baseline (bs24)", "#4A90D9"),
            ("repa_l0_128_per_residue_step400k", "REPA L0", "#55A868"),
            ("repa_l4_128_per_residue_step400k", "REPA L4", "#E74C3C"),
            ("repa_l9_128_per_residue_step400k", "REPA L9", "#F39C12"),
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
    "lambda": {
        "label": "λ ablation\n(REPA L4, bs=80, λ ∈ {0.5, 2.0}; λ=1.0 pending)",
        "dir": "n128_paper_lambda",
        "runs": [
            ("repa_l4_128_bs80_step200k", "λ=0.5", "#E74C3C"),
            ("repa_l4_128_bs80_lambda2_steplast", "λ=2.0", "#8B0000"),
        ],
    },
    "wd": {
        "label": "Weight-decay ablation\n(REPA L4, bs=80, wd default vs 1e-2)",
        "dir": "n128_paper_wd",
        "runs": [
            ("repa_l4_128_bs80_step200k", "wd default", "#E74C3C"),
            ("repa_l4_128_bs80_wd1e2_step200k", "wd=1e-2", "#8B0000"),
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
#
# Foldseek novelty: backfilled across all 59 paper rows on 2026-05-13 after
# fixing the silent-failure bug in compute_novelty_foldseek (default
# --alignment-type 2 returned empty m8 on CA-only PDBs; switched to type 1).
# Both max_tm_mean (continuous, threshold-free) and rate <0.5 (paper-style)
# are emitted so plots can show either.
METRICS = {
    "_res_PDB_FID": ("PDB FID", True),
    "_res_PDB_fJSD_C": ("PDB fJSD C", True),
    "_res_PDB_fJSD_A": ("PDB fJSD A", True),
    "_res_PDB_fJSD_T": ("PDB fJSD T", True),
    "_res_AFDB_FID": ("AFDB FID", True),
    "_res_AFDB_fJSD_C": ("AFDB fJSD C", True),
    "_res_AFDB_fJSD_A": ("AFDB fJSD A", True),
    "_res_AFDB_fJSD_T": ("AFDB fJSD T", True),
    "_res_fS_T": ("Fold Score (Topo)", False),
    "_res_designability_rate": ("Designability", False),
    "_res_scRMSD_mean": ("scRMSD mean (Å)", True),
    # Diversity: clusters_total = sum of clusters across the designable subset
    # (matches n=256 table and Proteina paper Table 1 "Cluster" count column).
    # Earlier we reported clusters_mean (per length-bin avg), which understated
    # the count by ~4× when designability spanned all 4 lengths.
    "_res_diversity_clusters_total": ("Diversity (clusters)", False),
    "_res_diversity_pairwise_tm_mean": ("Diversity (pairwise TM)", True),
    # SS composition + JSD vs reference (PDB / AFDB), all-samples and designable-only.
    # Fractions are display-only ("higher better" is meaningless on a 3-bin
    # composition); JSDs are lower-better. Catches all-α-helix mode collapse.
    "_res_ss_frac_H": ("SS %H", False),
    "_res_ss_frac_E": ("SS %E", False),
    # Foldseek novelty vs PDB and AFDB-SwissProt DBs. max_tm_mean is the
    # continuous score (lower = more novel); rate is the paper-style
    # fraction with max-TM < 0.5.
    "_res_novelty_foldseek_pdb_max_tm_mean": ("Foldseek max-TM (PDB)", True),
    "_res_novelty_foldseek_pdb_rate": ("Foldseek novelty rate (PDB)", False),
    "_res_novelty_foldseek_afdb_swissprot_max_tm_mean": (
        "Foldseek max-TM (AFDB)",
        True,
    ),
    "_res_novelty_foldseek_afdb_swissprot_rate": (
        "Foldseek novelty rate (AFDB)",
        False,
    ),
    "_res_ss_jsd_pdb": ("SS JSD (PDB)", True),
    "_res_ss_jsd_afdb": ("SS JSD (AFDB)", True),
    "_res_ss_jsd_pdb_designable": ("SS JSD des. (PDB)", True),
    "_res_ss_jsd_afdb_designable": ("SS JSD des. (AFDB)", True),
}

N_NOTES = {
    "_res_PDB_FID": "N=500",
    "_res_PDB_fJSD_C": "N=500",
    "_res_PDB_fJSD_A": "N=500",
    "_res_PDB_fJSD_T": "N=500",
    "_res_AFDB_FID": "N=500",
    "_res_AFDB_fJSD_C": "N=500",
    "_res_AFDB_fJSD_A": "N=500",
    "_res_AFDB_fJSD_T": "N=500",
    "_res_fS_T": "N=500",
    "_res_designability_rate": "N=200",
    "_res_scRMSD_mean": "N=200",
    "_res_diversity_clusters_total": "designable",
    "_res_diversity_pairwise_tm_mean": "designable",
    "_res_novelty_foldseek_pdb_max_tm_mean": "designable",
    "_res_novelty_foldseek_pdb_rate": "designable",
    "_res_novelty_foldseek_afdb_swissprot_max_tm_mean": "designable",
    "_res_novelty_foldseek_afdb_swissprot_rate": "designable",
    "_res_ss_frac_H": "N=500",
    "_res_ss_frac_E": "N=500",
    "_res_ss_jsd_pdb": "N=500",
    "_res_ss_jsd_afdb": "N=500",
    "_res_ss_jsd_pdb_designable": "designable",
    "_res_ss_jsd_afdb_designable": "designable",
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
                # Append shared-metadata suffix (e.g. " (CA-GearNet, PDB)") to
                # the row title so the reader sees the block-level training
                # config without it being repeated in every bar label.
                suffix = compose_title_suffix([run_name for run_name, _, _ in runs])
                ax.set_ylabel(f"{acfg['label']}{suffix}", fontsize=9, fontweight="bold")

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
