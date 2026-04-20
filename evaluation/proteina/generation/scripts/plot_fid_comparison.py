"""Bar-chart comparison of FID evaluation metrics across proteina runs.

Reads from evaluation/proteina/generation/results/pdb/fid/ CSVs and
highlights the best model per metric. Companion to plot_fid_convergence.py
(which shows metrics over training steps).

Usage:
    python evaluation/proteina/generation/scripts/plot_fid_comparison.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Layout: scripts/ sits next to results/ and figures/ under generation/.
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pdb", "fid")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# Which metrics exist and whether lower is better
METRIC_INFO = {
    "_res_PDB_FID": ("PDB FID", True),
    "_res_AFDB_FID": ("AFDB FID", True),
    "_res_PDB_fJSD_C": ("PDB Fold JSD (Class)", True),
    "_res_PDB_fJSD_A": ("PDB Fold JSD (Architecture)", True),
    "_res_PDB_fJSD_T": ("PDB Fold JSD (Topology)", True),
    "_res_AFDB_fJSD_C": ("AFDB Fold JSD (Class)", True),
    "_res_AFDB_fJSD_A": ("AFDB Fold JSD (Architecture)", True),
    "_res_AFDB_fJSD_T": ("AFDB Fold JSD (Topology)", True),
    "_res_fS_C": ("Fold Score (Class, max 5)", False),
    "_res_fS_A": ("Fold Score (Architecture, max 43)", False),
    "_res_fS_T": ("Fold Score (Topology, max 1336)", False),
    "_res_designability_rate": ("Designability rate (scRMSD < 2Å)", False),
    "_res_scRMSD_mean": ("scRMSD mean (Å)", True),
    "_res_scRMSD_median": ("scRMSD median (Å)", True),
    "_res_tm_score_self_mean": ("Self-TM mean", False),
    "_res_plddt_mean": ("ESMFold pLDDT mean", False),
    "_res_plddt_median": ("ESMFold pLDDT median", False),
    "_res_diversity_clusters_mean": ("Diversity (mean clusters/bin)", False),
}

# Display names and colours per run.
# For each run we pick the *latest* available checkpoint so the comparison
# is at the most-trained point.  When a _840k CSV exists we prefer it over
# the earlier checkpoint.
RUNS = {
    "inference_fid_60m_baseline.csv": ("Baseline", "#4C72B0"),
    "inference_fid_60m_repa_840k.csv": ("REPA (L4)", "#DD8452"),
    "inference_fid_60m_repa_layer0_840k.csv": ("REPA-L0", "#55A868"),
    "inference_fid_60m_repa_layer9_840k.csv": ("REPA-L9", "#C44E52"),
}

# Early-checkpoint versions used for the "early vs latest" grouped comparison.
# Includes the baseline at 535K to show its degradation to 742K.
EARLY_RUNS = {
    "results_inference_fid_60m_baseline_fid_535k.csv.bak": ("Baseline", "#4C72B0"),
    "inference_fid_60m_repa.csv": ("REPA (L4)", "#DD8452"),
    "inference_fid_60m_repa_layer0.csv": ("REPA-L0", "#55A868"),
    "inference_fid_60m_repa_layer9.csv": ("REPA-L9", "#C44E52"),
}

# Fallback global_step for CSVs that are missing it
DEFAULT_STEPS = {
    "inference_fid_60m_repa_layer9.csv": 420000,
}

# Batch sizes differ: baseline=6, REPA=4 (GearNet memory overhead on A100).
# Used to compute samples_seen = global_step * batch_size.
# NOTE (2026-04-19): when adding 128-residue runs, bs is NOT uniform.
#   baseline-128 (job 27971089, current) trained at bs=24 throughout.
#   All REPA-128 variants (gearnet + esm, per_residue + per_sample) train at bs=80.
#   Future baseline-128 runs / restarts will most likely adopt bs=80 — CONFIRM per run
#   before appending an entry (a mid-run bs switch means one run has two regimes, and
#   nsamples = step * bs is no longer a single scalar multiply).
BATCH_SIZES = {
    "Baseline": 6,
    "REPA (L4)": 4,
    "REPA-L0": 4,
    "REPA-L9": 4,
}


def _load_run_set(run_dict):
    """Load a set of result CSVs into a list of (label, color, step, metric_dict)."""
    results = []
    for fname, (label, color) in run_dict.items():
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        row = df.iloc[0]
        step = row.get("global_step", DEFAULT_STEPS.get(fname, np.nan))
        if pd.isna(step):
            step = DEFAULT_STEPS.get(fname, np.nan)
        metric_vals = {m: row[m] if m in row.index else np.nan for m in METRIC_INFO}
        results.append(
            (label, color, int(step) if not pd.isna(step) else None, metric_vals)
        )
    return results


def load_results():
    """Load latest-checkpoint results."""
    return _load_run_set(RUNS)


def load_early_results():
    """Load early-checkpoint (420K) results for the REPA runs."""
    return _load_run_set(EARLY_RUNS)


def _step_labels(results):
    """Labels with samples seen, e.g. 'Baseline\n4.5M samples'."""
    labels = []
    for name, _, step, _ in results:
        if step is not None:
            bs = BATCH_SIZES.get(name, 1)
            samples = step * bs
            labels.append(f"{name}\n{samples / 1e6:.1f}M samples")
        else:
            labels.append(name)
    return labels


def _build_grouped_data(results_late, results_early):
    """Build a dict mapping run label → {step, color, metrics} for early+late."""
    by_label = {}
    for label, color, step, metrics in results_late:
        by_label.setdefault(label, {"color": color})
        by_label[label]["late"] = (step, metrics)
    for label, color, step, metrics in results_early:
        by_label.setdefault(label, {"color": color})
        by_label[label]["early"] = (step, metrics)
    return by_label


def plot_fid_bars(results_late, results_early):
    """FID comparison bar chart (2 panels) with early+late grouped bars."""
    grouped = _build_grouped_data(results_late, results_early)
    run_labels = [r[0] for r in results_late]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key in zip(axes, ["_res_PDB_FID", "_res_AFDB_FID"]):
        x = np.arange(len(run_labels))
        w = 0.35
        all_vals = []

        # Early bars (lighter, hatched)
        early_vals = []
        early_present = []
        for label in run_labels:
            info = grouped[label]
            if "early" in info:
                early_vals.append(info["early"][1][key])
                early_present.append(True)
            else:
                early_vals.append(0)
                early_present.append(False)

        # Late bars
        late_vals = [grouped[label]["late"][1][key] for label in run_labels]
        all_vals = [v for v in early_vals + late_vals if v > 0]

        if any(early_present):
            bars_e = ax.bar(
                x - w / 2,
                early_vals,
                w,
                label="~0.5x samples",
                color=[grouped[lab]["color"] for lab in run_labels],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.5,
                hatch="//",
            )
        bars_l = ax.bar(
            x + (w / 2 if any(early_present) else 0),
            late_vals,
            w,
            label="Latest ckpt",
            color=[grouped[lab]["color"] for lab in run_labels],
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_ylabel("FID (lower = better)")
        ax.set_title(METRIC_INFO[key][0])
        ax.set_ylim(0, max(all_vals) * 1.2)
        ax.set_xticks(x)
        step_labels = []
        for label in run_labels:
            info = grouped[label]
            parts = [label]
            if "late" in info and info["late"][0]:
                bs = BATCH_SIZES.get(label, 1)
                samples = info["late"][0] * bs
                parts.append(f"{samples / 1e6:.1f}M samples")
            step_labels.append("\n".join(parts))
        ax.set_xticklabels(step_labels)

        # Annotate late bars
        for bar, v in zip(bars_l, late_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + max(all_vals) * 0.01,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        # Annotate early bars
        if any(early_present):
            for bar, v, present in zip(bars_e, early_vals, early_present):
                if present:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + max(all_vals) * 0.01,
                        f"{v:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="gray",
                    )

        best_idx = int(np.argmin(late_vals))
        bars_l[best_idx].set_edgecolor("gold")
        bars_l[best_idx].set_linewidth(2.5)

    axes[0].legend(fontsize=9)
    n_samples = "~6,125"
    fig.suptitle(
        f"FID Evaluation ({n_samples} samples/run)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fid_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_fjsd_bars(results_late, results_early):
    """Fold JSD comparison (grouped bars: PDB vs AFDB at each CATH level).

    Uses the latest checkpoint; annotates improvement from early checkpoint
    with arrows where available.
    """
    grouped = _build_grouped_data(results_late, results_early)
    run_labels = [r[0] for r in results_late]
    x = np.arange(len(run_labels))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    tick_labels = _step_labels(results_late)
    for ax, level, level_name in zip(
        axes, ["C", "A", "T"], ["Class", "Architecture", "Topology"]
    ):
        pdb_key = f"_res_PDB_fJSD_{level}"
        afdb_key = f"_res_AFDB_fJSD_{level}"
        pdb_vals = [grouped[lab]["late"][1][pdb_key] for lab in run_labels]
        afdb_vals = [grouped[lab]["late"][1][afdb_key] for lab in run_labels]

        bars_pdb = ax.bar(
            x - w / 2,
            pdb_vals,
            w,
            label="vs PDB",
            color="#4C72B0",
            edgecolor="black",
            linewidth=0.5,
        )
        bars_afdb = ax.bar(
            x + w / 2,
            afdb_vals,
            w,
            label="vs AFDB",
            color="#DD8452",
            edgecolor="black",
            linewidth=0.5,
        )

        # Add early-checkpoint markers as scatter points
        for i, label in enumerate(run_labels):
            info = grouped[label]
            if "early" in info:
                pdb_early = info["early"][1][pdb_key]
                afdb_early = info["early"][1][afdb_key]
                ax.scatter(
                    i - w / 2,
                    pdb_early,
                    marker="v",
                    color="#4C72B0",
                    s=40,
                    zorder=5,
                    alpha=0.5,
                )
                ax.scatter(
                    i + w / 2,
                    afdb_early,
                    marker="v",
                    color="#DD8452",
                    s=40,
                    zorder=5,
                    alpha=0.5,
                )

        # Highlight best (lowest) bar per reference set
        best_pdb = int(np.argmin(pdb_vals))
        bars_pdb[best_pdb].set_edgecolor("gold")
        bars_pdb[best_pdb].set_linewidth(2.5)
        best_afdb = int(np.argmin(afdb_vals))
        bars_afdb[best_afdb].set_edgecolor("gold")
        bars_afdb[best_afdb].set_linewidth(2.5)

        ax.set_ylabel("Fold JSD (lower = better)")
        ax.set_title(f"Fold JSD — {level_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=20, ha="right")
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(max(pdb_vals), max(afdb_vals)) * 1.3)

    fig.suptitle(
        "Fold Jensen-Shannon Divergence (latest ckpt, triangles = ~0.5x samples)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fjsd_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_fold_score_bars(results_late, results_early):
    """Fold score comparison (3 panels) with early+late grouped bars."""
    grouped = _build_grouped_data(results_late, results_early)
    run_labels = [r[0] for r in results_late]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, level in zip(axes, ["C", "A", "T"]):
        key = f"_res_fS_{level}"
        x = np.arange(len(run_labels))
        w = 0.35

        early_vals = []
        early_present = []
        for label in run_labels:
            info = grouped[label]
            if "early" in info:
                early_vals.append(info["early"][1][key])
                early_present.append(True)
            else:
                early_vals.append(0)
                early_present.append(False)

        late_vals = [grouped[label]["late"][1][key] for label in run_labels]
        all_vals = [v for v in early_vals + late_vals if v > 0]

        if any(early_present):
            ax.bar(
                x - w / 2,
                early_vals,
                w,
                label="~0.5x samples",
                color=[grouped[lab]["color"] for lab in run_labels],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.5,
                hatch="//",
            )
        bars_l = ax.bar(
            x + (w / 2 if any(early_present) else 0),
            late_vals,
            w,
            label="Latest ckpt",
            color=[grouped[lab]["color"] for lab in run_labels],
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_ylabel("Fold Score (higher = better)")
        ax.set_title(METRIC_INFO[key][0])
        ax.set_ylim(0, max(all_vals) * 1.25)
        ax.set_xticks(x)
        ax.set_xticklabels(_step_labels(results_late))

        for bar, v in zip(bars_l, late_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + max(all_vals) * 0.02,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        best_idx = int(np.argmax(late_vals))
        bars_l[best_idx].set_edgecolor("gold")
        bars_l[best_idx].set_linewidth(2.5)

    axes[0].legend(fontsize=9)
    fig.suptitle(
        "Fold Score — Inception Score variant",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fold_score_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def print_summary_table(results):
    """Print a text summary table with best values highlighted."""
    labels = [r[0] for r in results]
    steps = [r[2] for r in results]

    samples = [
        f"{s * BATCH_SIZES.get(lab, 1) / 1e6:.1f}M" if s else "?"
        for lab, s in zip(labels, steps)
    ]
    header = (
        f"{'Metric':<30}" + "".join(f"{lab:>14}" for lab in labels) + "   Direction"
    )
    print("\n" + "=" * len(header))
    print(f"{'Steps':30}" + "".join(f"{s or '?':>14}" for s in steps))
    print(f"{'Samples seen':30}" + "".join(f"{s:>14}" for s in samples))
    print(header)
    print("=" * len(header))
    for key, (display_name, lower_better) in METRIC_INFO.items():
        vals = [r[3][key] for r in results]
        finite_vals = [v for v in vals if not pd.isna(v)]
        if not finite_vals:
            continue  # metric absent from every CSV in this set
        best = min(finite_vals) if lower_better else max(finite_vals)
        row = f"{display_name:<30}"
        for v in vals:
            if pd.isna(v):
                row += f"{'—':>11}  "
            else:
                marker = " *" if v == best else "  "
                row += f"{v:>11.3f}{marker}"
        row += f"   {'Lower' if lower_better else 'Higher'} = better"
        print(row)
    print("=" * len(header))
    print("* = best in row")


def main():
    results_late = load_results()
    results_early = load_early_results()
    if not results_late:
        print("No result CSVs found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_fid_bars(results_late, results_early)
    plot_fjsd_bars(results_late, results_early)
    plot_fold_score_bars(results_late, results_early)
    print("\n--- Latest checkpoint comparison ---")
    print_summary_table(results_late)
    if results_early:
        print("\n--- Early (420K) checkpoint comparison ---")
        # Prepend baseline for context
        baseline = [r for r in results_late if r[0] == "Baseline"]
        print_summary_table(baseline + results_early)


if __name__ == "__main__":
    main()
