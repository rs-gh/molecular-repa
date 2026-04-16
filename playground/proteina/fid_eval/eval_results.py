"""Plot FID evaluation results across proteina runs at multiple checkpoints.

Now includes both 420k and 840k checkpoints for REPA models, plus
baseline at 535k and 742k. Produces:
  1. FID comparison bar chart (latest checkpoint per model)
  2. FID vs training step line plot (training efficiency)
  3. fJSD comparison
  4. Fold score comparison

Usage:
    source .venv/bin/activate
    python playground/proteina/fid_eval/eval_results.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os

OUTDIR = "playground/proteina/fid_eval/figures"
os.makedirs(OUTDIR, exist_ok=True)

# ── All evaluation data points (step, PDB_FID, AFDB_FID, fJSD_C, fS_C, fS_A, fS_T) ──

data = {
    "Baseline": [
        # (step, PDB_FID, AFDB_FID, PDB_fJSD_C, fS_C, fS_A, fS_T)
        (535500, 518.4, 532.9, 0.218, 2.21, 4.72, 22.2),
        (742000, 648.0, 622.0, 1.288, 2.44, 4.23, 19.4),
    ],
    "REPA (L4)": [
        (420000, 657.0, 659.7, 1.075, 1.78, 3.38, 15.8),
        (840000, 580.1, 569.8, 0.896, 1.82, 3.48, 18.3),
    ],
    "REPA-L0": [
        (420000, 599.1, 611.1, 0.842, 1.84, 3.06, 23.5),
        (836500, 401.8, 393.8, 0.533, 1.93, 4.26, 18.9),
    ],
    "REPA-L9": [
        (420000, 879.2, 887.1, 0.094, 2.11, 3.38, 14.7),
        (847000, 614.2, 677.0, 0.237, 2.15, 4.93, 28.6),
    ],
}

colors = {
    "Baseline": "#4C72B0",
    "REPA (L4)": "#DD8452",
    "REPA-L0": "#55A868",
    "REPA-L9": "#C44E52",
}

# ── Figure 1: FID vs Training Step (the key training efficiency plot) ──

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, fid_idx, title in [(axes[0], 1, "PDB FID"), (axes[1], 2, "AFDB FID")]:
    for label, points in data.items():
        steps = [p[0] for p in points]
        fids = [p[fid_idx] for p in points]
        ax.plot(
            steps,
            fids,
            "o-",
            label=label,
            color=colors[label],
            linewidth=2,
            markersize=8,
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("FID (lower = better)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(300, 950)

fig.suptitle(
    "FID vs Training Step — REPA improves, Baseline degrades",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fid_vs_step.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUTDIR}/fid_vs_step.png")

# ── Figure 2: FID bar chart (latest checkpoint per model) ──

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
runs = list(data.keys())
bar_colors = [colors[r] for r in runs]

for ax, fid_idx, title in [(axes[0], 1, "PDB FID"), (axes[1], 2, "AFDB FID")]:
    vals = [data[r][-1][fid_idx] for r in runs]  # latest checkpoint
    steps = [data[r][-1][0] for r in runs]
    labels = [f"{r}\n({s // 1000}k)" for r, s in zip(runs, steps)]
    bars = ax.bar(labels, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("FID (lower = better)")
    ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.15)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 10,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    best_idx = np.argmin(vals)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)

fig.suptitle("FID at Latest Checkpoint", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fid_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUTDIR}/fid_comparison.png")

# ── Figure 3: fJSD_C vs step ──

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
for label, points in data.items():
    steps = [p[0] for p in points]
    fjsd = [p[3] for p in points]
    ax.plot(
        steps, fjsd, "o-", label=label, color=colors[label], linewidth=2, markersize=8
    )
ax.set_xlabel("Training Step")
ax.set_ylabel("fJSD_C (lower = better)")
ax.set_title("Fold Class Diversity vs Training Step")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fjsd_vs_step.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUTDIR}/fjsd_vs_step.png")

# ── Figure 4: Fold Scores at latest checkpoint ──

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, fs_idx, level in [(axes[0], 4, "C"), (axes[1], 5, "A"), (axes[2], 6, "T")]:
    vals = [data[r][-1][fs_idx] for r in runs]
    steps = [data[r][-1][0] for r in runs]
    labels = [f"{r}\n({s // 1000}k)" for r, s in zip(runs, steps)]
    bars = ax.bar(labels, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Fold Score (higher = better)")
    ax.set_title(f"fS_{level}")
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

fig.suptitle("Fold Score at Latest Checkpoint", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fold_score_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUTDIR}/fold_score_comparison.png")

# ── Print summary table ──

print("\n" + "=" * 100)
print(
    f"{'Model':<12} {'Step':>8} {'PDB FID':>9} {'AFDB FID':>9} {'fJSD_C':>8} {'fS_C':>6} {'fS_A':>6} {'fS_T':>6}"
)
print("=" * 100)
for label, points in data.items():
    for p in points:
        print(
            f"{label:<12} {p[0]:>8,} {p[1]:>9.1f} {p[2]:>9.1f} {p[3]:>8.3f} {p[4]:>6.2f} {p[5]:>6.2f} {p[6]:>6.1f}"
        )
print("=" * 100)
