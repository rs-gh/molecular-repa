"""Fig 6.6 -- Deterministic (ODE) sampling over training: does the *model*
get genuinely better?

ODE sampling reads the learned distribution with no injected noise, so it
isolates what the trunk has learned from what the sampler recovers. We follow
the baseline and the strongest variant per regime across training.

Layout (rows = metric, cols = dataset):
    PDB-FPSD  | AFDB-FPSD     (log-y, lower = better)
    PDB-Des   | AFDB-Des      (linear-y, higher = better)

Story:
  AFDB -- REPA-L4-GearNet beats baseline on BOTH FPSD and designability,
          persistently -> a genuinely better learned model.
  PDB  -- REPA-L9-MPNN wins designability but loses FPSD late
          -> redistribution, not a uniformly better model.

Single-seed (ODE was not multi-seeded); read direction and persistence, not
precise magnitudes. Starts at 400K (the 100-200K points are barely-trained
artefacts).
"""

import json
import re
import os
import sys
import glob
from collections import defaultdict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from style import COLORS, MARKERS, setup_axes, log_step_axis, use_report_style

use_report_style()

ROOT = "/home/sr2173/git/molecular-repa"
GEN = f"{ROOT}/evaluation/proteina/generation"
OUT = os.environ.get(
    "FIG_OUT", f"{ROOT}/docs/masters-report/figures/fig6_ode_trajectory.png"
)
MIN_STEP_K = 400  # drop barely-trained 100-200K points


def base(run):
    return re.sub(r"_step\d+k?$", "", run or "")


# (run-base, display, color, marker) per regime
PDB = [
    ("baseline_256_bs24_2gpu", "Baseline", COLORS["baseline"], MARKERS["baseline"]),
    ("repa_mpnn_l9_256_per_residue", "REPA L9-MPNN", COLORS["L9"], MARKERS["MPNN"]),
]
AFDB = [
    ("baseline_afdb_256", "Baseline", COLORS["baseline"], MARKERS["baseline"]),
    ("repa_l4_afdb_256", "REPA L4-GearNet", COLORS["L4"], MARKERS["GearNet"]),
]

# accumulate ODE rows: data[run][step_k][field] = [values]
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for f in glob.glob(f"{GEN}/results/**/sweep_results*.jsonl", recursive=True):
    if ".bak" in f:
        continue
    for ln in open(f):
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if "error" in d or d.get("sampler_tag") != "ode" or not d.get("step"):
            continue
        bv = base(d.get("run"))
        sk = d["step"] // 1000
        for fld in ("_res_designability_rate", "_res_PDB_FID", "_res_AFDB_FID"):
            if d.get(fld) is not None:
                data[bv][sk][fld].append(d[fld])


def series(run, field):
    out = []
    for sk in sorted(data.get(run, {})):
        if sk < MIN_STEP_K:
            continue
        vals = data[run][sk].get(field, [])
        if vals:
            out.append((sk, sum(vals) / len(vals)))
    return [s for s, _ in out], [v for _, v in out]


fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))


def draw(ax, regime, field, title, ylabel, logy=False):
    for run, disp, color, marker in regime:
        sk, ys = series(run, field)
        if not sk:
            continue
        ax.plot(
            sk,
            ys,
            "-",
            marker=marker,
            color=color,
            markersize=5,
            linewidth=1.8,
            zorder=8,
            label=disp,
        )
    setup_axes(ax, title=title, ylabel=ylabel)
    if logy:
        ax.set_yscale("log")
    log_step_axis(ax)


draw(
    axes[0][0],
    PDB,
    "_res_PDB_FID",
    r"(a) PDB-trained: FPSD $\downarrow$",
    "FPSD (ODE, log)",
    logy=True,
)
draw(
    axes[0][1],
    AFDB,
    "_res_AFDB_FID",
    r"(b) AFDB-trained: FPSD $\downarrow$",
    "FPSD (ODE, log)",
    logy=True,
)
draw(
    axes[1][0],
    PDB,
    "_res_designability_rate",
    r"(c) PDB-trained: designability $\uparrow$",
    "Designability (ODE)",
)
draw(
    axes[1][1],
    AFDB,
    "_res_designability_rate",
    r"(d) AFDB-trained: designability $\uparrow$",
    "Designability (ODE)",
)
for ax in (axes[1][0], axes[1][1]):
    ax.set_ylim(-0.02, 0.5)

# Single shared legend below the grid (dedup by label).
handles, labels = [], []
for ax in axes.flat:
    for h, lbl in zip(*ax.get_legend_handles_labels()):
        if lbl not in labels:
            handles.append(h)
            labels.append(lbl)
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=len(labels),
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved {OUT}")
