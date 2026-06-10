"""Fig 1c — Headline combined: FID + designability convergence, 2x2 grid.

Merges fig1_headline_fid (FID, log-y) and fig1b_headline_designability
(designability, linear-y) into one figure.

Layout (rows = dataset, columns = metric):
    PDB-FID  | PDB-designability
    AFDB-FID | AFDB-designability

Variant selection matches the two source figures so all four panels read
together:
    PDB:  baseline + REPA L9-MPNN + L4-random
    AFDB: baseline + REPA L4-GearNet + L4-random
"""

import json
import re
import os
import sys
from collections import defaultdict
from statistics import mean
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))
from style import (
    classify_family,
    setup_axes,
    plot_trajectory,
    log_step_axis,
    use_report_style,
)

use_report_style()

ROOT = "/home/sr2173/git/molecular-repa"
OUT = os.environ.get(
    "FIG_OUT", f"{ROOT}/docs/masters-report/figures/fig01_fid_des_convergence.png"
)

PDB_JSONL = f"{ROOT}/evaluation/proteina/generation/results/paper/n256_convergence_pdb/sweep_results.clean.jsonl"
AFDB_JSONL = f"{ROOT}/evaluation/proteina/generation/results/paper/n256_convergence_afdb/sweep_results.clean.jsonl"

PDB_FAMS = [
    "baseline_256_bs24_2gpu",
    "repa_mpnn_l9_256_per_residue",
    "repa_l4_256_per_residue_random_bs24_2gpu",
]
AFDB_FAMS = ["baseline_afdb_256", "repa_l4_afdb_256", "repa_l4_afdb_256_random"]


def load(p):
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def family(run):
    return re.sub(r"_step\d+k$", "", run)


def trajec(rows, key):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[family(r["run"])][r["step"]].append(r.get(key))
    out = {}
    for fam in agg:
        out[fam] = {}
        for s in sorted(agg[fam]):
            vals = [v for v in agg[fam][s] if v is not None]
            if vals:
                out[fam][s] = (mean(vals), min(vals), max(vals), len(vals))
    return out


def draw(ax, tj, fams_to_show, title, ylabel):
    for fam in fams_to_show:
        if fam not in tj:
            continue
        color, marker, label, z = classify_family(fam)
        steps = sorted(tj[fam])
        steps_k = [s / 1000 for s in steps]
        means = [tj[fam][s][0] for s in steps]
        mins = [tj[fam][s][1] for s in steps]
        maxs = [tj[fam][s][2] for s in steps]
        # line + markers (suppress helper's SD band); draw a min--max band instead
        plot_trajectory(ax, steps_k, means, None, color, marker, label, zorder=z)
        if any(mx > mn for mn, mx in zip(mins, maxs)):
            ax.fill_between(steps_k, mins, maxs, color=color, alpha=0.18, zorder=z - 1)
    setup_axes(ax, title=title, ylabel=ylabel)
    log_step_axis(ax, label="Training step (log scale)")


def fid_panel(ax, tj, fams, title, ylabel):
    draw(ax, tj, fams, title, ylabel)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.FixedLocator([250, 300, 400, 500, 700, 1000]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def des_panel(ax, tj, fams, title, ylabel):
    draw(ax, tj, fams, title, ylabel)
    ax.set_ylim(0.0, 1.0)


pdb = load(PDB_JSONL)
afdb = load(AFDB_JSONL)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Row 0 = PDB
fid_panel(
    axes[0, 0],
    trajec(pdb, "_res_PDB_FID"),
    PDB_FAMS,
    title="(a) PDB-trained: FPSD $\\downarrow$",
    ylabel="FPSD-PDB (1.1K refs, log scale)",
)
des_panel(
    axes[0, 1],
    trajec(pdb, "_res_designability_rate"),
    PDB_FAMS,
    title="(b) PDB-trained: designability $\\uparrow$",
    ylabel="Designability rate",
)

# Row 1 = AFDB
fid_panel(
    axes[1, 0],
    trajec(afdb, "_res_AFDB_FID"),
    AFDB_FAMS,
    title="(c) AFDB-trained: FPSD $\\downarrow$",
    ylabel="FPSD-AFDB (4.5K refs, log scale)",
)
des_panel(
    axes[1, 1],
    trajec(afdb, "_res_designability_rate"),
    AFDB_FAMS,
    title="(d) AFDB-trained: designability $\\uparrow$",
    ylabel="Designability rate",
)

# One shared legend below the grid (dedup by label across all four panels).
handles, labels = [], []
seen = set()
for ax in axes.flat:
    for h, lbl in zip(*ax.get_legend_handles_labels()):
        if lbl not in seen:
            seen.add(lbl)
            handles.append(h)
            labels.append(lbl)
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=len(labels),
    fontsize=9,
    frameon=True,
    framealpha=0.85,
    bbox_to_anchor=(0.5, -0.02),
)

fig.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")
