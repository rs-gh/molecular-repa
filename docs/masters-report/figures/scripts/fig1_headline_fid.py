"""Fig 1 — Headline: FID convergence, 2-panel (PDB, AFDB).

PDB panel: baseline + REPA L9-MPNN (cleanest step-matched winner) + L4-random.
AFDB panel: baseline + REPA L4-GearNet (durable winner).

Reads convergence sweep clean.jsonl, plots in-distribution FID vs step
with 3-seed +/- SD bands.
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

sys.path.insert(0, os.path.dirname(__file__))
from style import classify_family, setup_axes, plot_trajectory, legend, log_step_axis

ROOT = "/home/sr2173/git/molecular-repa"
OUT = f"{ROOT}/docs/masters-report/figures/fig01_fid_convergence.png"


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


def plot_panel(ax, tj, fams_to_show, title, ylabel):
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
    ax.set_yscale("log")
    import matplotlib.ticker as mticker

    ax.yaxis.set_major_locator(mticker.FixedLocator([250, 300, 400, 500, 700, 1000]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    legend(ax, loc="upper right")


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)

# PDB — baseline + L9-MPNN + L4-random
pdb = load(
    f"{ROOT}/evaluation/proteina/generation/results/paper/n256_convergence_pdb/sweep_results.clean.jsonl"
)
plot_panel(
    axes[0],
    trajec(pdb, "_res_PDB_FID"),
    [
        "baseline_256_bs24_2gpu",
        "repa_mpnn_l9_256_per_residue",
        "repa_l4_256_per_residue_random_bs24_2gpu",
    ],
    title="PDB-trained — FID vs PDB reference ↓",
    ylabel="FID-PDB",
)

# AFDB — baseline + L4-GearNet
afdb = load(
    f"{ROOT}/evaluation/proteina/generation/results/paper/n256_convergence_afdb/sweep_results.clean.jsonl"
)
plot_panel(
    axes[1],
    trajec(afdb, "_res_AFDB_FID"),
    ["baseline_afdb_256", "repa_l4_afdb_256"],
    title="AFDB-trained — FID vs AFDB reference ↓",
    ylabel="FID-AFDB",
)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")
