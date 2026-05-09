"""Bar chart comparison for the n=256 paper-protocol sweeps.

Reads from results/n256_paper_pdb/, n256_paper_random/, and n256_paper_afdb/
sweep_results.csv and produces a 3 x len(METRICS) grid:
    rows  = profile (PDB main, PDB+random ablation, AFDB main)
    cols  = metric (PDB FID, AFDB FID, fS_T, fJSD_T, designability, scRMSD,
                    diversity, novelty)
    bars  = runs within that profile

Paper protocol:
    Generation: 1125 PDBs across {50,75,...,250} x 125/length
    FID/fJSD/fS: full 1125-PDB pool
    Designability: 250 PDBs (50/length x 5 paper lengths)
    Diversity/novelty: designable subset of those 250

Missing rows render as "pending" placeholders, so the plot is useful at any
point during a sweep — partial data still renders.

Usage:
    python evaluation/proteina/generation/scripts/plot_n256_paper.py
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
FIGURES_DIR = GENERATION_ROOT / "figures" / "paper" / "n256_paper"

# External-reference run overlaid as a dashed horizontal line on every panel.
# 12-layer NVIDIA NGC release ckpt at 1.3M steps; not directly comparable to
# our 10-layer 60M (see project_proteina_60m_layer_mismatch.md), but useful
# as an order-of-magnitude target.
PRETRAINED_PROFILE_DIR = "n256_paper_pretrained"
PRETRAINED_RUN_NAME = "pretrained_dfs_60m_n256_paper"
PRETRAINED_LABEL = "Pretrained 60M (NGC, 1.3M steps, 12L)"
PRETRAINED_LINE_COLOR = "#222222"

# Train-set chains per epoch (used to convert epoch -> samples_M for the
# x-tick labels). Pulled from the same source as plot_n256_sweep.py for PDB;
# AFDB derived from step * bs / epoch on the ep20 ckpts (200K * 24 / 20).
TRAIN_CHAINS_PER_EPOCH = {
    "pdb": 267_789,  # PDB train, 50<=L<=256 (matches plot_n256_sweep.py)
    "afdb": 240_000,  # AFDB-Swissprot, 50<=L<=256 (= 200K * 24 / 20 on ep20 ckpts; both AFDB runs started post-2026-04-18 SDPA bump → bs=24 throughout)
}

# Per-profile config: dir name + ordered run list. Each run tuple is
# (run_name, display_label, color, train_set, epoch). train_set picks the
# chains-per-epoch table; epoch * chains_per_epoch -> samples_M shown on the
# x-tick.
#
# Colors group conceptually:
#   Baselines (PDB / AFDB): blue family
#   REPA L0/L4/L9: green/red/orange (matching plot_n256_sweep.py)
#   Random encoder ablation: gray (control)
#   AFDB-trained REPA: purple (distinguish from PDB-trained L4)
PROFILES = {
    "n256_paper_pdb": {
        "label": "PDB-trained — main comparison",
        "dir": "n256_paper_pdb",
        "runs": [
            ("baseline_256_ep21", "Baseline", "#4A90D9", "pdb", 21),
            ("repa_l0_256_ep26", "REPA L0", "#55A868", "pdb", 26),
            ("repa_l4_256_ep22", "REPA L4", "#E74C3C", "pdb", 22),
            ("repa_l9_256_ep25", "REPA L9", "#F39C12", "pdb", 25),
        ],
    },
    "n256_paper_random": {
        "label": "PDB-trained — random-encoder ablation (epoch ≤17)",
        "dir": "n256_paper_random",
        "runs": [
            ("baseline_256_ep21", "Baseline", "#4A90D9", "pdb", 21),
            ("repa_l9_256_ep17", "REPA L9", "#F39C12", "pdb", 17),
            ("repa_l4_256_random_ep17", "REPA L4 random", "#7F7F7F", "pdb", 17),
        ],
    },
    "n256_paper_afdb": {
        "label": "AFDB-Swissprot-trained (epoch 20)",
        "dir": "n256_paper_afdb",
        "runs": [
            ("baseline_afdb_256_ep20", "Baseline", "#4A90D9", "afdb", 20),
            ("repa_l4_afdb_256_ep20", "REPA L4", "#9B59B6", "afdb", 20),
        ],
    },
    # Encoder ablation — ProteinMPNN. Anchors (Baseline, CA-GearNet=L4 ep22)
    # mirrored from n256_paper_pdb so the bar chart shows MPNN vs the encoder
    # block reference points within one panel.
    "n256_paper_mpnn": {
        "label": "Encoder ablation — ProteinMPNN (PDB)",
        "dir": "n256_paper_mpnn",
        "runs": [
            ("baseline_256_ep21", "Baseline", "#4A90D9", "pdb", 21),
            ("repa_l4_256_ep22", "CA-GearNet (L4)", "#E74C3C", "pdb", 22),
            (
                "repa_mpnn_l4_256_per_residue_step300k",
                "ProteinMPNN (L4)",
                "#8B4513",
                "pdb",
                26,
            ),
        ],
    },
    # Encoder ablation — ESM2 (L9-t30, bs=12). Same anchor pattern. Note ESM
    # row carries a double caveat (bs mismatch + L9-t30 ≠ L4 default).
    "n256_paper_esm": {
        "label": "Encoder ablation — ESM2 L9-t30 (PDB, bs=12*)",
        "dir": "n256_paper_esm",
        "runs": [
            ("baseline_256_ep21", "Baseline", "#4A90D9", "pdb", 21),
            ("repa_l4_256_ep22", "CA-GearNet (L4)", "#E74C3C", "pdb", 22),
            # ESM bs=12 throughout; epoch=14 is approx (3.86M smp / chains-per-epoch).
            ("repa_esm_l9_t30_256_steplast", "ESM2 (L9-t30)", "#1ABC9C", "pdb", 14),
        ],
    },
    # λ ablation — REPA L4 with λ ∈ {0.5, 1.0, 2.0}. λ=0.5 anchored at both
    # ep22/400K (default reference) and ep13/300K (step-matched to λ=1.0).
    "n256_paper_lambda": {
        "label": "λ ablation — REPA L4 (PDB)",
        "dir": "n256_paper_lambda",
        "runs": [
            ("repa_l4_256_ep22", "λ=0.5 ep22", "#E74C3C", "pdb", 22),
            ("repa_l4_256_ep13_step300k", "λ=0.5 ep13/300K", "#FBB4AE", "pdb", 13),
            (
                "repa_l4_256_per_residue_lambda1_step300k",
                "λ=1.0 ep26",
                "#C44893",
                "pdb",
                26,
            ),
            (
                "repa_l4_256_per_residue_lambda2_step200k",
                "λ=2.0 ep17",
                "#8B0000",
                "pdb",
                17,
            ),
        ],
    },
    # L4 step-trajectory — same run as the main L4, two ckpts (ep13/300K and
    # ep22/400K) so the per-step trajectory is visible at a glance.
    "n256_paper_l4_step300k": {
        "label": "REPA L4 step trajectory (PDB)",
        "dir": "n256_paper_l4_step300k",
        "runs": [
            ("repa_l4_256_ep13_step300k", "L4 ep13/300K", "#FBB4AE", "pdb", 13),
            ("repa_l4_256_ep22", "L4 ep22/400K", "#E74C3C", "pdb", 22),
        ],
    },
    # Averaging ablation — per_residue vs per_sample for L0/L4/L9. Each
    # per_sample bar paired with its per_residue anchor; lighter tints = the
    # per_sample variant.
    "n256_paper_averaging": {
        "label": "Averaging ablation — per_residue vs per_sample (PDB, L4)",
        "dir": "n256_paper_averaging",
        "runs": [
            ("repa_l0_256_ep26", "L0 per_residue", "#55A868", "pdb", 26),
            ("repa_l0_256_per_sample_steplast", "L0 per_sample", "#A6D7B5", "pdb", 28),
            ("repa_l4_256_ep22", "L4 per_residue", "#E74C3C", "pdb", 22),
            ("repa_l4_256_per_sample_step400k", "L4 per_sample", "#FBB4AE", "pdb", 25),
            ("repa_l9_256_ep25", "L9 per_residue", "#F39C12", "pdb", 25),
            ("repa_l9_256_per_sample_steplast", "L9 per_sample", "#FCD9A4", "pdb", 28),
        ],
    },
}

# (display label, lower_is_better). N-note is shared across all paper-protocol
# panels — generation pool is 1125, designability pool is 250, diversity/novelty
# run on the designable subset (variable, ~75-175 typical).
METRICS = {
    "_res_PDB_FID": ("PDB FID", True),
    "_res_AFDB_FID": ("AFDB FID", True),
    "_res_fS_T": ("Fold Score (Topo)", False),
    "_res_PDB_fJSD_T": ("PDB fJSD (Topo)", True),
    "_res_designability_rate": ("Designability", False),
    "_res_scRMSD_mean": ("scRMSD mean (Å)", True),
    "_res_diversity_pairwise_tm_mean": ("Diversity (pairwise TM)", True),
    "_res_diversity_clusters_total": ("Diversity (clusters total)", False),
    "_res_novelty_rate": ("Novelty (centroid)", False),
    # SS composition + JSD vs reference. Display %H, %E (fractions sum to 1
    # with %C); JSD vs PDB and AFDB, all-samples and designable-only.
    "_res_ss_frac_H": ("SS %H", False),
    "_res_ss_frac_E": ("SS %E", False),
    "_res_ss_jsd_pdb": ("SS JSD (PDB)", True),
    "_res_ss_jsd_afdb": ("SS JSD (AFDB)", True),
    "_res_ss_jsd_pdb_designable": ("SS JSD des. (PDB)", True),
    "_res_ss_jsd_afdb_designable": ("SS JSD des. (AFDB)", True),
}

N_NOTES = {
    "_res_PDB_FID": "N=1125",
    "_res_AFDB_FID": "N=1125",
    "_res_fS_T": "N=1125",
    "_res_PDB_fJSD_T": "N=1125",
    "_res_designability_rate": "N=250",
    "_res_scRMSD_mean": "N=250",
    "_res_diversity_pairwise_tm_mean": "designable",
    "_res_diversity_clusters_total": "designable",
    "_res_novelty_rate": "designable",
    "_res_ss_frac_H": "N=1125",
    "_res_ss_frac_E": "N=1125",
    "_res_ss_jsd_pdb": "N=1125",
    "_res_ss_jsd_afdb": "N=1125",
    "_res_ss_jsd_pdb_designable": "designable",
    "_res_ss_jsd_afdb_designable": "designable",
}


def load_profile_rows(profile_dir: Path) -> dict[str, dict]:
    """Return {run_name: row_dict} for a profile's sweep_results.jsonl.

    Returns empty dict if no JSONL exists yet (sweep hasn't produced any
    completed checkpoints).
    """
    rows = load_sweep_rows(profile_dir / "sweep_results.jsonl")
    return {r["run"]: r for r in rows}


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


def _xtick_label(label: str, epoch: int, train_set: str, step: int | None) -> str:
    """Three-line tick: '<run>\\nep<E> | step <K>K\\n~<S>M smp'.

    Mirrors the format in plot_n256_sweep.py (run + step + samples) but adds
    the epoch number since the paper-protocol run names already encode it.
    samples_M = epoch * chains_per_epoch (varies by train set).
    """
    samples_m = epoch * TRAIN_CHAINS_PER_EPOCH[train_set] / 1e6
    step_str = f"step {step // 1000}K" if step is not None else "last"
    return f"{label}\nep{epoch} | {step_str}\n~{samples_m:.2f}M smp"


def plot() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n_metrics = len(METRICS)
    n_profiles = len(PROFILES)
    fig, axes = plt.subplots(
        n_profiles,
        n_metrics,
        figsize=(3.4 * n_metrics, 4.0 * n_profiles),
        squeeze=False,
    )

    # Pre-load all profile data
    profile_data: dict[str, dict[str, dict]] = {}
    for pkey, pcfg in PROFILES.items():
        profile_data[pkey] = load_profile_rows(RESULTS_ROOT / pcfg["dir"])

    # Load pretrained reference row (single jsonl). Missing keys are silently
    # skipped per panel; the pretrained sweep didn't run centroid novelty so
    # the novelty panel will have no overlay.
    pretrained_rows = load_profile_rows(RESULTS_ROOT / PRETRAINED_PROFILE_DIR)
    pretrained_row = pretrained_rows.get(PRETRAINED_RUN_NAME)

    # One row per profile, one column per metric
    for prow, (pkey, pcfg) in enumerate(PROFILES.items()):
        rows = profile_data[pkey]
        runs = pcfg["runs"]
        x = np.arange(len(runs))
        colors = [r[2] for r in runs]
        # Build x-tick labels with epoch / step / samples per run.
        tick_labels = [
            _xtick_label(
                label,
                epoch,
                train_set,
                int(rows[run_name]["step"])
                if run_name in rows and "step" in rows[run_name]
                else None,
            )
            for run_name, label, _color, train_set, epoch in runs
        ]

        for pcol, (mkey, (mlabel, lower_better)) in enumerate(METRICS.items()):
            ax = axes[prow][pcol]
            vals: list[float] = []
            missing: list[bool] = []
            for run_name, _label, _color, _train_set, _epoch in runs:
                v = _val(rows.get(run_name), mkey)
                vals.append(0.0 if v is None else v)
                missing.append(v is None)

            bars = ax.bar(
                x,
                vals,
                color=colors,
                edgecolor="white",
                linewidth=0.8,
                width=0.65,
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
                        fontsize=8,
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

            # Pretrained reference line (NGC 1.3M-step ckpt). Drawn after bars
            # so y-limits expand if it sits above our values; labelled once
            # via the figure-level legend below.
            pre_v = _val(pretrained_row, mkey)
            if pre_v is not None:
                ax.axhline(
                    pre_v,
                    color=PRETRAINED_LINE_COLOR,
                    linestyle="--",
                    linewidth=1.2,
                    zorder=4,
                )
                ax.text(
                    ax.get_xlim()[1],
                    pre_v,
                    f" {pre_v:.2f}" if abs(pre_v) < 10 else f" {pre_v:.0f}",
                    ha="left",
                    va="center",
                    fontsize=7,
                    color=PRETRAINED_LINE_COLOR,
                    style="italic",
                )

            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, fontsize=7)
            ax.grid(axis="y", alpha=0.3, zorder=0)
            ax.set_axisbelow(True)

            if prow == 0:
                direction = "↓ better" if lower_better else "↑ better"
                ax.set_title(
                    f"{mlabel}\n({N_NOTES[mkey]}, {direction})",
                    fontsize=9,
                    fontweight="bold",
                )

            if pcol == 0:
                ax.set_ylabel(pcfg["label"], fontsize=9, fontweight="bold")

    # Legend: one patch per unique (label, color) across all profiles
    seen: set[tuple[str, str]] = set()
    legend_entries: list[tuple[str, str]] = []
    for pcfg in PROFILES.values():
        for _run_name, label, color, _train_set, _epoch in pcfg["runs"]:
            key = (label, color)
            if key not in seen:
                seen.add(key)
                legend_entries.append(key)

    handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in legend_entries]
    if pretrained_row is not None:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=PRETRAINED_LINE_COLOR,
                linestyle="--",
                linewidth=1.5,
                label=PRETRAINED_LABEL,
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 5),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.suptitle(
        "n=256 paper-protocol sweep — generation metrics across 8 profiles\n"
        "Pool: 1125 PDBs at L∈{50,75,…,250} × 125; designability on 50/L × 5 lengths = 250; "
        "diversity/novelty on designable subset.\n"
        "Best per panel highlighted in gold.",
        fontsize=10,
        fontweight="bold",
        y=0.99,
    )
    out = FIGURES_DIR / "n256_paper_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    plot()
