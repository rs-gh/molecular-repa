"""Plot all evaluation metrics vs training step for proteina runs.

Usage:
    python evaluation/proteina/scripts/plot_fid_convergence.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pdb", "fid")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")

# All metric columns from the result CSVs
METRICS = [
    "_res_PDB_FID",
    "_res_PDB_fJSD_C",
    "_res_PDB_fJSD_A",
    "_res_PDB_fJSD_T",
    "_res_AFDB_FID",
    "_res_AFDB_fJSD_C",
    "_res_AFDB_fJSD_A",
    "_res_AFDB_fJSD_T",
    "_res_fS_C",
    "_res_fS_A",
    "_res_fS_T",
]

# Human-readable labels for each metric
METRIC_LABELS = {
    "_res_PDB_FID": "PDB FID",
    "_res_PDB_fJSD_C": "PDB Fold JSD (Class)",
    "_res_PDB_fJSD_A": "PDB Fold JSD (Architecture)",
    "_res_PDB_fJSD_T": "PDB Fold JSD (Topology)",
    "_res_AFDB_FID": "AFDB FID",
    "_res_AFDB_fJSD_C": "AFDB Fold JSD (Class)",
    "_res_AFDB_fJSD_A": "AFDB Fold JSD (Architecture)",
    "_res_AFDB_fJSD_T": "AFDB Fold JSD (Topology)",
    "_res_fS_C": "Fold Score (Class)",
    "_res_fS_A": "Fold Score (Architecture)",
    "_res_fS_T": "Fold Score (Topology)",
}

# Style mapping: (color, marker, linestyle)
STYLES = {
    "Baseline (60M)": ("#4A90D9", "o", "-"),
    "REPA (L4)": ("#E74C3C", "s", "-"),
    "REPA (L0)": ("#55A868", "D", "-"),
    "REPA (L9)": ("#F39C12", "^", "-"),
}

# Batch sizes differ across runs: baseline=6, REPA=4 (GearNet memory overhead).
# Multiply global_step by batch_size to get samples seen for fair comparison.
BATCH_SIZES = {
    "Baseline (60M)": 6,
    "REPA (L4)": 4,
    "REPA (L0)": 4,
    "REPA (L9)": 4,
}


def load_results():
    """Load all result CSVs and the .bak file into a unified dataframe."""
    records = []

    runs = {
        "inference_fid_60m_baseline.csv": "Baseline (60M)",
        "inference_fid_60m_repa.csv": "REPA (L4)",
        "inference_fid_60m_repa_layer0.csv": "REPA (L0)",
        "inference_fid_60m_repa_layer9.csv": "REPA (L9)",
        "inference_fid_60m_repa_840k.csv": "REPA (L4)",
        "inference_fid_60m_repa_layer0_840k.csv": "REPA (L0)",
        "inference_fid_60m_repa_layer9_840k.csv": "REPA (L9)",
    }

    # Default steps for runs where global_step is missing from the CSV.
    # These are known from the training logs / checkpoint names.
    default_steps = {
        "inference_fid_60m_repa_layer9.csv": 420000,
    }

    for fname, label in runs.items():
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            step = row.get("global_step", np.nan)
            if pd.isna(step):
                step = default_steps.get(fname, np.nan)
            record = {
                "run": label,
                "global_step": int(step) if not pd.isna(step) else np.nan,
            }
            for metric in METRICS:
                record[metric] = row[metric]
            records.append(record)

    # Include the .bak file as an extra baseline data point
    bak_path = os.path.join(
        RESULTS_DIR, "results_inference_fid_60m_baseline_fid_535k.csv.bak"
    )
    if os.path.exists(bak_path):
        df = pd.read_csv(bak_path)
        for _, row in df.iterrows():
            record = {
                "run": "Baseline (60M)",
                "global_step": int(row["global_step"]),
            }
            for metric in METRICS:
                record[metric] = row[metric]
            records.append(record)

    df = pd.DataFrame(records).sort_values(["run", "global_step"])
    # Add samples_seen column for fair cross-run comparison
    df["samples_seen"] = df.apply(
        lambda r: r["global_step"] * BATCH_SIZES.get(r["run"], 1)
        if not pd.isna(r["global_step"])
        else np.nan,
        axis=1,
    )
    return df


def plot_metric(data: pd.DataFrame, metric: str, ax):
    """Plot a single metric vs samples seen on the given axis."""
    for run_name, group in data.groupby("run"):
        group = group.dropna(subset=["samples_seen"]).sort_values("samples_seen")
        if group.empty:
            continue
        color, marker, ls = STYLES.get(run_name, ("gray", "x", ":"))
        ax.plot(
            group["samples_seen"],
            group[metric],
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2,
            linestyle=ls,
            label=run_name,
        )
        # Annotate each point with its step count (still useful context)
        for _, row in group.iterrows():
            ax.annotate(
                f"{int(row['global_step']) // 1000}K steps",
                (row["samples_seen"], row[metric]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                color=color,
            )

    ax.set_xlabel("Samples Seen", fontsize=10)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
    ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))


def main():
    data = load_results()
    print("Loaded data:")
    print(data.to_string(index=False))
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- FID plot (2 panels) ---
    fig_fid, axes_fid = plt.subplots(1, 2, figsize=(14, 5))
    plot_metric(data, "_res_PDB_FID", axes_fid[0])
    plot_metric(data, "_res_AFDB_FID", axes_fid[1])
    axes_fid[0].legend(fontsize=9)
    fig_fid.suptitle(
        "FID vs Samples Seen (BS=6 baseline, BS=4 REPA)", fontsize=14, fontweight="bold"
    )
    fig_fid.tight_layout()
    fid_path = os.path.join(OUTPUT_DIR, "fid_convergence.png")
    fig_fid.savefig(fid_path, dpi=150, bbox_inches="tight")
    print(f"Saved {fid_path}")

    # --- fJSD plot (3x2 grid: PDB row, AFDB row; columns = C, A, T) ---
    fig_jsd, axes_jsd = plt.subplots(2, 3, figsize=(18, 10))
    for i, ref in enumerate(["PDB", "AFDB"]):
        for j, ss in enumerate(["C", "A", "T"]):
            metric = f"_res_{ref}_fJSD_{ss}"
            plot_metric(data, metric, axes_jsd[i, j])
    axes_jsd[0, 0].legend(fontsize=9)
    fig_jsd.suptitle(
        "Fold JSD vs Samples Seen (BS=6 baseline, BS=4 REPA)",
        fontsize=14,
        fontweight="bold",
    )
    fig_jsd.tight_layout()
    jsd_path = os.path.join(OUTPUT_DIR, "fjsd_convergence.png")
    fig_jsd.savefig(jsd_path, dpi=150, bbox_inches="tight")
    print(f"Saved {jsd_path}")

    # --- Feature scores plot (3 panels: C, A, T) ---
    fig_fs, axes_fs = plt.subplots(1, 3, figsize=(18, 5))
    for j, ss in enumerate(["C", "A", "T"]):
        metric = f"_res_fS_{ss}"
        plot_metric(data, metric, axes_fs[j])
    axes_fs[0].legend(fontsize=9)
    fig_fs.suptitle(
        "Fold Scores vs Samples Seen (BS=6 baseline, BS=4 REPA)",
        fontsize=14,
        fontweight="bold",
    )
    fig_fs.tight_layout()
    fs_path = os.path.join(OUTPUT_DIR, "feature_scores_convergence.png")
    fig_fs.savefig(fs_path, dpi=150, bbox_inches="tight")
    print(f"Saved {fs_path}")


if __name__ == "__main__":
    main()
