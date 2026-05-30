"""Fig 5 — REPA concentrates the designable subset, not the whole set.

2-panel envelope:
  x = whole-set pwTM      (lower = more diverse over the whole generated set)
  y = designable pwTM     (lower = more diverse within designable subset)

Diagonal (y = x): designable subset is as diverse as whole set (no
concentration). Points ABOVE the diagonal: designable subset is more
concentrated than the whole set — the REPA trade-off mechanism (the
designability filter selects REPA's concentrated high-confidence modes).

Each point: one (model, step) cell. Marker size scales with training step.
Trajectories connect a model's checkpoints in step order.

Data: results/variance/wholeset_vs_designable_diversity.json (36 cells).
"""

import json
import os
import sys
import re
from collections import defaultdict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from style import COLORS, MARKERS, setup_axes, legend, use_report_style

use_report_style()

ROOT = "/home/sr2173/git/molecular-repa"
SRC = f"{ROOT}/evaluation/proteina/generation/results/variance/wholeset_vs_designable_diversity.json"
OUT = f"{ROOT}/docs/masters-report/figures/fig05_td_tradeoff.png"

with open(SRC) as f:
    data = json.load(f)


def parse_key(key):
    """Parse e.g. 'PDB_L9_GN_1000K' -> (dataset, family_label, step)."""
    m = re.match(r"(PDB|AFDB)_(.+)_(\d+)K$", key)
    if not m:
        return None
    dataset, fam_label, step_k = m.group(1), m.group(2), int(m.group(3))
    return dataset, fam_label, step_k * 1000


def family_style(fam_label):
    """Pick color/marker/display label from the JSON family label."""
    f = fam_label.lower()
    if "baseline" in f:
        return COLORS["baseline"], MARKERS["baseline"], "Baseline", 10
    if "rand" in f:
        return COLORS["random"], MARKERS["random"], "REPA L4-random", 5
    is_l9 = "_l9" in f or f.startswith("l9") or "l9" in f.split("_")[0]
    is_mpnn = "mpnn" in f
    layer = "L9" if is_l9 else "L4"
    enc = "MPNN" if is_mpnn else "GearNet"
    return COLORS[layer], MARKERS[enc], f"REPA {layer}-{enc}", 8


def gather(dataset_tag, families_to_show):
    """Return {family_label: [(step, whole, des), ...]} sorted by step."""
    by = defaultdict(list)
    for key, v in data.items():
        p = parse_key(key)
        if p is None:
            continue
        ds, fam_label, step = p
        if ds != dataset_tag:
            continue
        if fam_label not in families_to_show:
            continue
        by[fam_label].append((step, v["whole_pwtm"], v["des_pwtm"]))
    for f_ in by:
        by[f_].sort(key=lambda t: t[0])
    return by


def plot_panel(ax, dataset_tag, families_to_show, title):
    by = gather(dataset_tag, families_to_show)
    # Diagonal first (behind everything)
    all_vals = [v for fam in by.values() for (_, w, d) in fam for v in (w, d)]
    if all_vals:
        lo, hi = min(all_vals) - 0.02, max(all_vals) + 0.04
        ax.plot(
            [lo, hi],
            [lo, hi],
            "--",
            color="#888888",
            linewidth=1.0,
            alpha=0.6,
            zorder=1,
            label="y = x (no concentration)",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    # Each family
    for fam_label in families_to_show:
        if fam_label not in by:
            continue
        color, marker, label, z = family_style(fam_label)
        pts = by[fam_label]
        steps = [p[0] for p in pts]
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        sizes = [40 + (s / 1600000) * 140 for s in steps]
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
            alpha=0.9,
        )
    setup_axes(
        ax,
        title=title,
        xlabel="whole-set pwTM (lower = more diverse)",
        ylabel="designable pwTM (lower = more diverse)",
    )
    legend(ax, loc="upper left")
    ax.set_aspect("equal", adjustable="box")


fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))

# PDB — baseline + L9-GN (clearest concentration) + L9-MPNN + L4-rand (control)
plot_panel(
    axes[0],
    "PDB",
    ["baseline", "L9_GN", "L9_MPNN", "L4_rand"],
    title="PDB: REPA concentrates designable, not whole-set",
)

# AFDB — baseline + L4-GN (clean concentration) + L9-MPNN (the falsifier, stays on diagonal)
plot_panel(
    axes[1],
    "AFDB",
    ["baseline", "L4_GN", "L9_MPNN"],
    title="AFDB: MPNN-L9 stays on diagonal (no concentration)",
)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")
