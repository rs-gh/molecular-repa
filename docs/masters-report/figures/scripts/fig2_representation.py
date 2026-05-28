"""Fig 2 — Representation decodability: FM alone vs. REPA, 3-panel.

n=256 PDB-trained models, baseline + REPA together (no problem/solution split):
  IF top-1 (xclean)      — per-residue local geometry
  Dihedral MAE (xclean)  — per-residue local geometry
  CATH-A (cleantrain)    — per-chain fold class

Drops NVIDIA-60M reference (same-architecture-more-training was always a
shaky "ceiling"). The companion alignment story (CKNNA) is Fig 3.
"""

import csv
import re
import os
import sys
from collections import defaultdict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from style import classify_family, setup_axes, plot_trajectory, legend, log_step_axis

ROOT = "/home/sr2173/git/molecular-repa"
XCLEAN = f"{ROOT}/evaluation/proteina/representation/results/paper/n256_xclean_afdb_pdb/pretrained_sweep_results.csv"
CLEANTRAIN = f"{ROOT}/evaluation/proteina/representation/results/paper/n256_convergence_cleantrain_pdb/pretrained_sweep_results.csv"
OUT = f"{ROOT}/docs/masters-report/figures/fig02_representation.png"


def fam(run):
    return re.sub(r"_step\d+k$", "", run)


def best_layer(csv_path, probe_kind, value_col, level=None, higher_better=True):
    by = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("probe_kind") != probe_kind:
                continue
            if level is not None and row.get("cath_level") != level:
                continue
            v = row.get(value_col)
            if not v:
                continue
            try:
                v = float(v)
                step = int(row["step"])
                layer = int(row["layer"])
            except Exception:
                continue
            by[fam(row["run"])][step].append((layer, v))
    out = {}
    for f_, steps in by.items():
        out[f_] = {
            s: (max if higher_better else min)(v for _, v in vs)
            for s, vs in steps.items()
        }
    return out


def plot_traj_panel(ax, data, families, title, ylabel, higher_better=True):
    for family in families:
        if family not in data:
            continue
        color, marker, label, z = classify_family(family)
        steps = sorted(data[family])
        steps_k = [s / 1000 for s in steps]
        vals = [data[family][s] for s in steps]
        plot_trajectory(ax, steps_k, vals, None, color, marker, label, zorder=z)
    setup_axes(ax, title=title, ylabel=ylabel)
    log_step_axis(ax)
    legend(ax, loc="lower right" if higher_better else "upper right")


FAMS = [
    "baseline_256_bs24_2gpu",
    "repa_l9_256_per_residue_bs24_2gpu",  # L9-GearNet, green circle
    "repa_mpnn_l9_256_per_residue",  # L9-MPNN, green triangle
    "repa_l4_256_per_residue_random_bs24_2gpu",  # L4-random, grey x
]

# Pull data
if_data = best_layer(XCLEAN, "inverse_folding", "if_top1_acc", higher_better=True)
dih_data = best_layer(XCLEAN, "dihedral", "dih_mae_total_deg", higher_better=False)
cath_data = best_layer(
    CLEANTRAIN, "cath", "cath_accuracy", level="A", higher_better=True
)

# Build figure — 3 panels in a row
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
plot_traj_panel(
    axes[0], if_data, FAMS, "IF top-1 (xclean PDB) ↑", "top-1 acc", higher_better=True
)
plot_traj_panel(
    axes[1],
    dih_data,
    FAMS,
    "Dihedral MAE (xclean PDB) ↓",
    "MAE (°)",
    higher_better=False,
)
plot_traj_panel(
    axes[2],
    cath_data,
    FAMS,
    "CATH-A (cleantrain PDB) ↑",
    "CATH-A top-1",
    higher_better=True,
)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")
