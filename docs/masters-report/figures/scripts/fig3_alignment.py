"""Fig 3 — Representation alignment (CKNNA), 2-panel.

Left:  alignment to GearNet (the REPA target), per-layer at step 1M.
       Baseline + REPA L9-GearNet + REPA L9-MPNN.
Right: REPA-L9-GearNet's alignment to three different target encoders
       (GearNet, MPNN, ESM2) — Platonic convergence: aligning to GearNet
       alone also raises alignment to MPNN and ESM2.

n=256 PDB, step 1M, t=1.0, per-residue, k=10.
"""

import json
import os
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from style import COLORS, MARKERS, setup_axes, legend, use_report_style

use_report_style()

ROOT = "/home/sr2173/git/molecular-repa"
SRC = f"{ROOT}/evaluation/proteina/alignment/results/cknna_matrix_per_residue.jsonl"
OUT = f"{ROOT}/docs/masters-report/figures/fig03_alignment.png"

rows = []
with open(SRC) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass


def by_layer(model, target):
    pts = [
        (r["layer"], r["cknna"])
        for r in rows
        if r["model"] == model and r["encoder"] == target
    ]
    return sorted(pts)


def model_style(model):
    m = model.lower()
    if "baseline" in m:
        return COLORS["baseline"], MARKERS["baseline"], "Baseline", 10
    is_l9 = "_l9" in m
    is_mpnn = "mpnn" in m
    layer = "L9" if is_l9 else "L4"
    enc = "MPNN" if is_mpnn else "GearNet"
    return COLORS[layer], MARKERS[enc], f"REPA {layer}-{enc}", 8


fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)

# LEFT — each REPA variant's alignment to ITS OWN target encoder, vs baseline floor
ax = axes[0]
OWN = [
    (
        "baseline",
        "gearnet",
        "Baseline (floor)",
        COLORS["baseline"],
        MARKERS["baseline"],
        10,
    ),
    (
        "repa_gearnet_l9",
        "gearnet",
        "REPA L9-GearNet $\\rightarrow$ GearNet",
        COLORS["L9"],
        MARKERS["GearNet"],
        8,
    ),
    (
        "repa_mpnn_l9",
        "mpnn",
        "REPA L9-MPNN $\\rightarrow$ MPNN",
        COLORS["L9"],
        MARKERS["MPNN"],
        8,
    ),
]
for model, target, label, color, marker, z in OWN:
    pts = by_layer(model, target)
    if not pts:
        continue
    ax.plot(
        [p[0] for p in pts],
        [p[1] for p in pts],
        "-",
        marker=marker,
        color=color,
        label=label,
        linewidth=1.8,
        markersize=5.5,
        zorder=z,
        alpha=0.95,
    )
setup_axes(
    ax,
    title="(a) Alignment to own target encoder $\\uparrow$",
    xlabel="Trunk layer index",
    ylabel="CKNNA (per-residue, k=10)",
)
ax.set_xticks(list(range(0, 10)))
legend(ax, loc="upper left")

# RIGHT — Platonic: REPA-L9-GN to GearNet vs MPNN vs ESM2
ax = axes[1]
b_pts = by_layer("baseline", "gearnet")
if b_pts:
    ax.plot(
        [p[0] for p in b_pts],
        [p[1] for p in b_pts],
        "-",
        marker=MARKERS["baseline"],
        color=COLORS["baseline"],
        label="Baseline (any target)",
        linewidth=1.5,
        markersize=5,
        alpha=0.7,
        zorder=5,
    )
TARGETS = [
    ("gearnet", "GearNet (target)", COLORS["L9"], "o"),
    ("mpnn", "ProteinMPNN (off-diag)", COLORS["L4"], "^"),
    ("esm2", "ESM2 (off-diag)", COLORS["random"], "x"),
]
for target, label, color, marker in TARGETS:
    pts = by_layer("repa_gearnet_l9", target)
    if not pts:
        continue
    ax.plot(
        [p[0] for p in pts],
        [p[1] for p in pts],
        "-",
        marker=marker,
        color=color,
        label=f"REPA L9-GN $\\rightarrow$ {label}",
        linewidth=1.8,
        markersize=6,
        zorder=10,
    )
setup_axes(
    ax,
    title="(b) REPA-L9-GN propagates alignment off-diagonally (Platonic) $\\uparrow$",
    xlabel="Trunk layer index",
)
ax.set_xticks(list(range(0, 10)))
legend(ax, loc="upper left")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")
