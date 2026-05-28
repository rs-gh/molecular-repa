"""Fig 5 — Gen-vs-rep bridge: better student reps predict better generations.

Lead pair (cleanest defensible): dihedral MAE -> designability, xclean PDB.
Each point is a (run, step) checkpoint; lines connect a run's trajectory.

For each checkpoint joined across rep (xclean) and gen (n256 convergence PDB).
"""

import csv
import json
import re
import os
import sys
from collections import defaultdict
from statistics import mean
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from style import classify_family, setup_axes, legend

ROOT = "/home/sr2173/git/molecular-repa"
XCLEAN = f"{ROOT}/evaluation/proteina/representation/results/paper/n256_xclean_afdb_pdb/pretrained_sweep_results.csv"
GEN = f"{ROOT}/evaluation/proteina/generation/results/paper/n256_convergence_pdb/sweep_results.clean.jsonl"
OUT = f"{ROOT}/docs/masters-report/figures/fig04_gen_vs_rep.png"


def fam(run):
    return re.sub(r"_step\d+k$", "", run)


# Rep: dihedral MAE best-layer per (family, step)
rep = defaultdict(lambda: defaultdict(list))
with open(XCLEAN) as f:
    for row in csv.DictReader(f):
        if row.get("probe_kind") != "dihedral":
            continue
        v = row.get("dih_mae_total_deg")
        if not v:
            continue
        try:
            v = float(v)
            step = int(row["step"])
            layer = int(row["layer"])
        except Exception:
            continue
        rep[fam(row["run"])][step].append((layer, v))
rep_data = {
    f: {s: min(vs, key=lambda x: x[1])[1] for s, vs in steps.items()}
    for f, steps in rep.items()
}

# Gen: designability rate seed-mean per (family, step)

gen = defaultdict(lambda: defaultdict(list))
with open(GEN) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        d = r.get("_res_designability_rate")
        if d is None:
            continue
        gen[fam(r["run"])][r["step"]].append(d)
gen_data = {f: {s: mean(vs) for s, vs in steps.items()} for f, steps in gen.items()}

# Build per-family trajectories of (rep, gen) at matched steps
families_to_show = [
    "baseline_256_bs24_2gpu",  # floor reference
    "repa_mpnn_l9_256_per_residue",  # best: wins both dihedral (x) and Des (y)
    "repa_l4_256_per_residue_random_bs24_2gpu",  # falsifier: random target
]

fig, ax = plt.subplots(figsize=(7.5, 5.4))
TARGET_STEP = 400000
target_points = []  # (x, y, color, marker, label) at TARGET_STEP

for family in families_to_show:
    if family not in rep_data or family not in gen_data:
        continue
    color, marker, label, z = classify_family(family)
    common = sorted(set(rep_data[family]) & set(gen_data[family]))
    if not common:
        continue
    xs = [rep_data[family][s] for s in common]
    ys = [gen_data[family][s] for s in common]
    sizes = [30 + (s / 1600000) * 100 for s in common]
    ax.plot(xs, ys, "-", color=color, alpha=0.4, linewidth=1.2, zorder=z - 1)
    ax.scatter(
        xs,
        ys,
        marker=marker,
        color=color,
        s=sizes,
        label=label,
        edgecolors="white",
        linewidths=0.5,
        zorder=z,
        alpha=0.85,
    )
    # Capture point at TARGET_STEP for the same-compute arrow
    if TARGET_STEP in rep_data[family] and TARGET_STEP in gen_data[family]:
        target_points.append(
            (
                rep_data[family][TARGET_STEP],
                gen_data[family][TARGET_STEP],
                color,
                marker,
                label,
            )
        )

setup_axes(
    ax,
    title="REPA delivers better generation quality at the same training compute",
    xlabel="Dihedral MAE (xclean PDB, best layer, °)",
    ylabel="Designability rate ↑",
)
ax.invert_xaxis()  # so "better rep" reads left-to-right
legend(ax, loc="lower right")

# === Same-compute arrow at TARGET_STEP ===
# Highlight the three 400K points and draw arrows in increasing-quality order
target_points.sort(key=lambda p: p[1])  # sort by designability (low -> high)
for x, y, color, marker, _label in target_points:
    ax.scatter(
        [x],
        [y],
        marker=marker,
        color=color,
        s=260,
        edgecolors="black",
        linewidths=1.4,
        zorder=25,
    )
for i in range(len(target_points) - 1):
    x0, y0, *_ = target_points[i]
    x1, y1, *_ = target_points[i + 1]
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>", color="black", lw=1.8, alpha=0.7, shrinkA=12, shrinkB=12
        ),
        zorder=20,
    )


plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")
