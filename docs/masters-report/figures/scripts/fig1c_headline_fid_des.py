"""Fig 1c — Headline combined: FID + designability convergence, 2x2 grid.

Merges fig1_headline_fid (FID, log-y) and fig1b_headline_designability
(designability, linear-y) into one figure.

Layout (rows = dataset, columns = metric):
    PDB-FID  | PDB-designability
    AFDB-FID | AFDB-designability

Variant selection matches the two source figures so all four panels read
together:
    PDB:  baseline + REPA L9-MPNN + L4-random
    AFDB: baseline + REPA L4-GearNet
"""

import json
import re
import os
import sys
from collections import defaultdict
from statistics import mean, stdev
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))
from style import classify_family, setup_axes, plot_trajectory, legend, log_step_axis

ROOT = "/home/sr2173/git/molecular-repa"
OUT = f"{ROOT}/docs/masters-report/figures/fig01_fid_des_convergence.png"

PDB_JSONL = f"{ROOT}/evaluation/proteina/generation/results/paper/n256_convergence_pdb/sweep_results.clean.jsonl"
AFDB_JSONL = f"{ROOT}/evaluation/proteina/generation/results/paper/n256_convergence_afdb/sweep_results.clean.jsonl"

PDB_FAMS = [
    "baseline_256_bs24_2gpu",
    "repa_mpnn_l9_256_per_residue",
    "repa_l4_256_per_residue_random_bs24_2gpu",
]
AFDB_FAMS = ["baseline_afdb_256", "repa_l4_afdb_256"]


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
                out[fam][s] = (
                    mean(vals),
                    stdev(vals) if len(vals) > 1 else 0.0,
                    len(vals),
                )
    return out


def draw(ax, tj, fams_to_show, title, ylabel):
    for fam in fams_to_show:
        if fam not in tj:
            continue
        color, marker, label, z = classify_family(fam)
        steps = sorted(tj[fam])
        steps_k = [s / 1000 for s in steps]
        means = [tj[fam][s][0] for s in steps]
        stds = [tj[fam][s][1] for s in steps]
        plot_trajectory(ax, steps_k, means, stds, color, marker, label, zorder=z)
    setup_axes(ax, title=title, ylabel=ylabel)
    log_step_axis(ax)


def fid_panel(ax, tj, fams, title, ylabel):
    draw(ax, tj, fams, title, ylabel)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.FixedLocator([250, 300, 400, 500, 700, 1000]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    legend(ax, loc="upper right")


def des_panel(ax, tj, fams, title, ylabel):
    draw(ax, tj, fams, title, ylabel)
    ax.set_ylim(0.0, 1.0)
    legend(ax, loc="lower right")


pdb = load(PDB_JSONL)
afdb = load(AFDB_JSONL)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Row 0 = PDB
fid_panel(
    axes[0, 0],
    trajec(pdb, "_res_PDB_FID"),
    PDB_FAMS,
    title="PDB-trained — FID vs PDB reference ↓",
    ylabel="FID-PDB",
)
des_panel(
    axes[0, 1],
    trajec(pdb, "_res_designability_rate"),
    PDB_FAMS,
    title="PDB-trained — designability ↑",
    ylabel="Designability rate",
)

# Row 1 = AFDB
fid_panel(
    axes[1, 0],
    trajec(afdb, "_res_AFDB_FID"),
    AFDB_FAMS,
    title="AFDB-trained — FID vs AFDB reference ↓",
    ylabel="FID-AFDB",
)
des_panel(
    axes[1, 1],
    trajec(afdb, "_res_designability_rate"),
    AFDB_FAMS,
    title="AFDB-trained — designability ↑",
    ylabel="Designability rate",
)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")
