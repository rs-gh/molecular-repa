"""Fig 4 (combined) — Gen-vs-rep bridge, 2 panels: FID (left), designability (right).

Merges fig4_gen_vs_rep (designability) and fig4b_gen_vs_rep_fid (FID) into one
figure. x-axis = student representation quality (dihedral MAE, xclean PDB,
best layer), inverted so "better rep" reads left-to-right. Each point is a
(run, step) checkpoint; lines connect a run's trajectory.

Conventions vs the source figures:
  - No panel titles.
  - Random target removed: baseline + REPA L9-MPNN only.
  - 400K checkpoints bordered (not enlarged), with a dashed baseline ->
    L9-MPNN same-compute arrow and a "400K" label by the leftmost dot.
  - FID panel y-axis is NOT inverted (low FID at the bottom).
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
from style import classify_family, setup_axes, legend, use_report_style

use_report_style()

ROOT = "/home/sr2173/git/molecular-repa"
XCLEAN = f"{ROOT}/evaluation/proteina/representation/results/paper/n256_xclean_afdb_pdb/pretrained_sweep_results.csv"
GEN = f"{ROOT}/evaluation/proteina/generation/results/paper/n256_convergence_pdb/sweep_results.clean.jsonl"
OUT = f"{ROOT}/docs/masters-report/figures/fig04_fid_des_gen_vs_rep.png"

TARGET_STEP = 400000
FAMILIES = [
    "baseline_256_bs24_2gpu",  # floor reference
    "repa_mpnn_l9_256_per_residue",  # best: wins dihedral (x), FID and Des (y)
    "repa_l4_256_per_residue_random_bs24_2gpu",  # random-target falsifier
]


def fam(run):
    return re.sub(r"_step\d+k$", "", run)


# --- Rep: dihedral MAE best-layer per (family, step) ---
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


# --- Gen: seed-mean of a metric per (family, step) ---
def gen_trajec(key):
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
            d = r.get(key)
            if d is None:
                continue
            gen[fam(r["run"])][r["step"]].append(d)
    return {f: {s: mean(vs) for s, vs in steps.items()} for f, steps in gen.items()}


def panel(ax, gen_data, ylabel, set_ylim01=False):
    target_points = []  # (x, y, color, marker, label) at TARGET_STEP
    for family in FAMILIES:
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
        xlabel="Dihedral MAE (xclean PDB, best layer, $^\\circ$)",
        ylabel=ylabel,
    )
    ax.invert_xaxis()  # so "better rep" reads left-to-right
    if set_ylim01:
        ax.set_ylim(0.0, 1.0)
    legend(ax, loc="lower right")

    # Border (don't enlarge) the 400K points; dashed baseline -> best arrow.
    target_marker_size = 30 + (TARGET_STEP / 1600000) * 100
    for x, y, color, marker, _label in target_points:
        ax.scatter(
            [x],
            [y],
            marker=marker,
            color=color,
            s=target_marker_size,
            edgecolors="black",
            linewidths=1.4,
            zorder=25,
        )
    by_label = {p[4]: p for p in target_points}
    start = by_label.get("Baseline")
    end = by_label.get("REPA L9-MPNN")
    if start and end:
        ax.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                lw=1.0,
                alpha=0.6,
                linestyle="--",
                shrinkA=5,
                shrinkB=5,
            ),
            zorder=20,
        )
    if target_points:
        # x-axis is inverted, so leftmost == largest dihedral MAE.
        leftmost = max(target_points, key=lambda p: p[0])
        ax.annotate(
            "400K",
            xy=(leftmost[0], leftmost[1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="black",
            zorder=26,
        )


fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

# Left = FID-PDB (lower is better; y NOT inverted)
panel(axes[0], gen_trajec("_res_PDB_FID"), ylabel="FID-PDB $\\downarrow$")

# Right = designability rate (higher is better)
panel(
    axes[1],
    gen_trajec("_res_designability_rate"),
    ylabel="Designability rate $\\uparrow$",
    set_ylim01=True,
)

fig.suptitle(
    "Generation quality vs. student representation quality across training checkpoints "
    "(PDB, $n{=}256$)",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")
