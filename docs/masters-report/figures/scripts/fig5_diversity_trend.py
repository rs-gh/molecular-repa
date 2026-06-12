"""Fig (Ch6) — over training, REPA's whole set grows more diverse while its
designable subset grows more concentrated. Replaces the 400K/700K/1.0M
snapshot Table 6.8.

PDB, n<=256, default sampler (gamma=0.45). Two panels, each dense (per-
checkpoint, seed-averaged with a min/max band):

  Left  — WHOLE-set fold spread fS-A (HIGHER = more diverse). REPA stays at or
          above the baseline: the whole set does not lose diversity.
  Right — DESIGNABLE-subset pwTM (LOWER = more diverse). REPA climbs above the
          baseline: it concentrates the designable subset.

The contrast is the point: whole-set diversity up, designable diversity down.
Baseline + REPA L9-MPNN, with a shared legend.
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from style import COLORS, MARKERS, log_step_axis, setup_axes, use_report_style

use_report_style()

ROOT = "/home/sr2173/git/molecular-repa"
GEN = f"{ROOT}/evaluation/proteina/generation"
JSON_SRC = f"{GEN}/results/variance/wholeset_vs_designable_diversity.json"
OUT = f"{ROOT}/docs/masters-report/figures/fig5_diversity_trend.png"

# run-base (PDB, default sampler) -> (display, color, marker, json-key)
SERIES = {
    "baseline_256_bs24_2gpu": (
        "Baseline",
        COLORS["baseline"],
        MARKERS["baseline"],
        "PDB_baseline",
    ),
    "repa_mpnn_l9_256_per_residue": (
        "REPA L9-MPNN",
        COLORS["L9"],
        MARKERS["MPNN"],
        "PDB_L9_MPNN",
    ),
}


def base(run):
    return re.sub(r"_step\d+k?$", "", run or "")


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var**0.5


# ---- dense whole-set fS-A + designable-subset pwTM from convergence records ----
raw = defaultdict(lambda: defaultdict(lambda: {"des": [], "fsa": []}))
for f in glob.glob(f"{GEN}/results/**/sweep_results*.jsonl", recursive=True):
    if ".bak" in f or "pdb" not in f.lower():
        continue
    for ln in open(f):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if "error" in d or "_res_diversity_pairwise_tm_mean" not in d:
            continue
        if d.get("sampler_tag") not in ("sde_n0.45", None):
            continue
        bv = base(d.get("run"))
        if bv not in SERIES or not d.get("step"):
            continue
        raw[bv][d["step"]]["des"].append(d.get("_res_diversity_pairwise_tm_mean"))
        raw[bv][d["step"]]["fsa"].append(d.get("_res_fS_A"))

# ---- sparse whole-set pwTM from json ----
wholej = json.load(open(JSON_SRC))


def whole_series(json_key):
    pts = []
    for k, v in wholej.items():
        m = re.match(rf"{json_key}_(\d+)K$", k)
        if m:
            pts.append((int(m.group(1)), v["whole_pwtm"]))
    return sorted(pts)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.8))


# Per-panel cap: truncate the baseline to the furthest step of the genuine
# trained REPA variant(s) shown, so no curve extends past the last variant point.
_trained = [bv for bv in SERIES if bv.startswith("repa") and "random" not in bv]
CAP_STEP = max((max(raw[bv]) for bv in _trained if raw[bv]), default=None)


def aligned(bv, field):
    """Dense per-checkpoint series: (steps_k, means, mins, maxs)."""
    sk, ms, los, his = [], [], [], []
    is_ref = bv.startswith("baseline") or "random" in bv
    for s in sorted(raw[bv]):
        if is_ref and CAP_STEP is not None and s > CAP_STEP:
            continue
        vals = [v for v in raw[bv][s][field] if v is not None]
        if not vals:
            continue
        sk.append(s / 1000)
        ms.append(sum(vals) / len(vals))
        los.append(min(vals))
        his.append(max(vals))
    return sk, ms, los, his


for bv, (label, color, marker, jkey) in SERIES.items():
    # Left: whole-set fold spread fS-A (higher = more diverse)
    sk, fsa_m, fsa_lo, fsa_hi = aligned(bv, "fsa")
    axL.plot(
        sk,
        fsa_m,
        "-",
        marker=marker,
        color=color,
        linewidth=1.8,
        markersize=5,
        zorder=8,
        label=label,
    )
    axL.fill_between(sk, fsa_lo, fsa_hi, color=color, alpha=0.15, zorder=4)
    # Right: designable-subset pwTM (lower = more diverse)
    sk2, des_m, des_lo, des_hi = aligned(bv, "des")
    axR.plot(
        sk2,
        des_m,
        "-",
        marker=marker,
        color=color,
        linewidth=1.8,
        markersize=5,
        zorder=8,
        label=label,
    )
    axR.fill_between(sk2, des_lo, des_hi, color=color, alpha=0.15, zorder=4)

setup_axes(axL, title=r"(a) Whole set", ylabel=r"fS-A ($\uparrow$ = more diverse)")
log_step_axis(axL)
setup_axes(
    axR, title=r"(b) Designable subset", ylabel=r"pwTM ($\downarrow$ = more diverse)"
)
log_step_axis(axR)

# Single shared legend below the panels (common-legend convention).
handles, labels = axL.get_legend_handles_labels()
fig.tight_layout(rect=[0, 0.08, 1, 1])
leg = fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=len(labels),
    fontsize=9,
    frameon=True,
    framealpha=0.85,
    bbox_to_anchor=(0.5, 0.0),
)
leg.get_frame().set_linewidth(0.5)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")
