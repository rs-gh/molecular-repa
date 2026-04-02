"""Plot and tabulate FID evaluation results across proteina runs.

Usage:
    source .venv/bin/activate
    python playground/proteina/eval_results.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os

OUTDIR = "playground/proteina/figures"
os.makedirs(OUTDIR, exist_ok=True)

# ── Results from quick-pass eval (25 samples/len, ~1000 proteins per run) ──

runs = ["Baseline", "REPA (L4)", "REPA-L0", "REPA-L9"]
steps = [266_000, 252_000, 241_500, 241_500]
n_samples = 1000  # approximate

metrics = {
    # (values, lower_is_better)
    "PDB_FID": ([547.9, 434.2, 665.5, 905.8], True),
    "AFDB_FID": ([550.2, 454.4, 694.8, 884.6], True),
    "PDB_fJSD_C": ([0.201, 0.890, 0.804, 0.059], True),
    "PDB_fJSD_A": ([0.939, 1.183, 1.733, 1.515], True),
    "PDB_fJSD_T": ([3.020, 3.028, 3.944, 4.024], True),
    "AFDB_fJSD_C": ([0.354, 0.526, 0.365, 0.282], True),
    "AFDB_fJSD_A": ([0.780, 0.882, 1.248, 1.447], True),
    "AFDB_fJSD_T": ([2.995, 2.935, 4.016, 3.978], True),
    "fS_C": ([2.50, 1.92, 1.94, 2.28], False),
    "fS_A": ([4.81, 3.78, 2.50, 3.05], False),
    "fS_T": ([22.33, 25.06, 22.09, 11.83], False),
}

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── Figure 1: FID comparison (the headline metric) ──

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, key in zip(axes, ["PDB_FID", "AFDB_FID"]):
    vals = metrics[key][0]
    bars = ax.bar(runs, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("FID (lower = better)")
    ax.set_title(key.replace("_", " "))
    ax.set_ylim(0, max(vals) * 1.15)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 15,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    # Highlight best
    best_idx = np.argmin(vals)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)

fig.suptitle(
    f"FID Evaluation (quick pass: ~{n_samples} samples/run)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fid_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUTDIR}/fid_comparison.png")

# ── Figure 2: fJSD at C/A/T levels ──

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, level in zip(axes, ["C", "A", "T"]):
    pdb_vals = metrics[f"PDB_fJSD_{level}"][0]
    afdb_vals = metrics[f"AFDB_fJSD_{level}"][0]
    x = np.arange(len(runs))
    w = 0.35
    bars1 = ax.bar(
        x - w / 2,
        pdb_vals,
        w,
        label="vs PDB",
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + w / 2,
        afdb_vals,
        w,
        label="vs AFDB",
        color="#DD8452",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("fJSD (lower = better)")
    ax.set_title(
        f"fJSD_{level} ({['Class','Architecture','Topology'][['C','A','T'].index(level)]})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=20, ha="right")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(pdb_vals), max(afdb_vals)) * 1.2)

fig.suptitle(
    f"Fold Jensen-Shannon Divergence (quick pass: ~{n_samples} samples/run)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fjsd_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUTDIR}/fjsd_comparison.png")

# ── Figure 3: Fold Scores ──

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, level in zip(axes, ["C", "A", "T"]):
    vals = metrics[f"fS_{level}"][0]
    bars = ax.bar(runs, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Fold Score (higher = better)")
    ax.set_title(
        f"fS_{level} ({['Class (max 5)','Architecture (max 43)','Topology (max 1336)'][['C','A','T'].index(level)]})"
    )
    ax.set_ylim(0, max(vals) * 1.2)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(vals) * 0.02,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)

fig.suptitle(
    f"Fold Score — Inception Score variant (quick pass: ~{n_samples} samples/run)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fold_score_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUTDIR}/fold_score_comparison.png")

# ── Print summary table ──

print("\n" + "=" * 90)
print(
    f"{'Metric':<16} {'Baseline':>12} {'REPA (L4)':>12} {'REPA-L0':>12} {'REPA-L9':>12}   Direction"
)
print("=" * 90)
for name, (vals, lower) in metrics.items():
    best = min(vals) if lower else max(vals)
    row = f"{name:<16}"
    for v in vals:
        marker = " *" if v == best else "  "
        row += f"{v:>11.3f}{marker}"
    row += f"   {'Lower' if lower else 'Higher'} = better"
    print(row)
print("=" * 90)
print(f"\nTraining steps:  {steps}")
print(f"Samples per run: ~{n_samples} (25 samples/len x 40 lengths)")
print("\n* = best in row")

print("""
NOTE ON SAMPLE SIZE
-------------------
These results use only ~1,000 generated proteins per run (25 samples across
40 lengths from 60-255 residues). This is a QUICK PASS for pipeline validation.

Why this matters:
- FID estimates the Frechet distance between two Gaussians fitted to 512-dim
  GearNet features. With ~1000 samples, the covariance estimate has high
  variance — FID can easily shift by 50-100 points between runs with the
  same checkpoint.
- fJSD averages softmax probabilities across samples. With few samples per
  length bin, rare folds are underrepresented and the estimated distribution
  is noisy.
- fS (fold score) divides samples into splits for IS computation. With only
  ~1000 samples and 1 split, the estimate has high variance.

For publication-quality numbers, run with nsamples_per_len=125 (~5,000 proteins)
which matches the reference inference_fid_ca.yaml config. This gives ~5x more
samples for covariance estimation and much tighter confidence intervals.

Additionally, the runs have different training steps (242k-266k), so differences
may partly reflect training duration rather than REPA effectiveness.
""")
