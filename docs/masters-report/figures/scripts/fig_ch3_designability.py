#!/usr/bin/env python3
"""Ch3 figure: designable vs non-designable generated backbones, proper Ca cartoons.
4 designable (2 helical + 2 beta-rich) over 4 non-designable, our own 1.3M PDB samples.
Designable = ESMFold refold within 2A (scRMSD); non-designable = refold fails.
Output: figures/fig_ch3_designability.png
"""

import os
import sys
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import use_report_style  # noqa: E402
from cartoon_util import draw_cartoon, ca_sse, SS_COLOR  # noqa: E402

use_report_style()
EVAL = "/home/sr2173/git/molecular-repa/eval_output"
OUTDIR = os.path.join(HERE, "..")
DIRS = [
    "inference_paper_inference_fid_60m_paper_sweep_repa_mpnn_l9_256_per_residue_step1300k_step_1300000__sde_n0.45__rep0",
    "inference_paper_inference_fid_60m_paper_sweep_repa_mpnn_l9_256_per_residue_step1300k_step_1300000__sde_n0.45__rep1",
    "inference_paper_inference_fid_60m_paper_sweep_baseline_256_bs24_2gpu_step1300k_step_1300000__sde_n0.45__rep1",
]


def load(dname):
    d = os.path.join(EVAL, dname)
    rows = list(csv.DictReader(open(os.path.join(d, "designability_index.csv"))))
    z = np.load(os.path.join(d, "ss_cache", "ss_fractions.npz"), allow_pickle=True)
    strand = {
        os.path.basename(str(p)): float(fr[1]) for p, fr in zip(z["paths"], z["fracs"])
    }
    out = []
    for r in rows:
        try:
            sc = float(r["scRMSD"])
        except (TypeError, ValueError):
            sc = np.nan
        out.append(
            dict(
                path=os.path.join(d, r["pdb_path"]),
                designable=r["designable"] == "True",
                scRMSD=sc,
                strand=strand.get(os.path.basename(r["pdb_path"]), np.nan),
                length=int(r["length"]),
                evaluated=r["evaluated"] == "True",
                plddt=(float(r["plddt"]) if r["plddt"] else np.nan),
            )
        )
    return out


S = []
for dn in DIRS:
    S += load(dn)
S = [x for x in S if x["length"] >= 120]
hel = sorted(
    [x for x in S if x["designable"] and x["strand"] < 0.08], key=lambda x: x["scRMSD"]
)[:2]
bet = sorted(
    [x for x in S if x["designable"] and x["strand"] >= 0.28], key=lambda x: x["scRMSD"]
)[:2]
bad = sorted(
    [
        x
        for x in S
        if (not x["designable"]) and x["evaluated"] and not np.isnan(x["plddt"])
    ],
    key=lambda x: x["plddt"],
)[:4]
des = hel + bet

fig = plt.figure(figsize=(9.4, 5.0))
for ri, (label, row) in enumerate([("Designable", des), ("Not designable", bad)]):
    for ci, s in enumerate(row):
        ax = fig.add_subplot(2, 4, ri * 4 + ci + 1, projection="3d")
        co, ss = ca_sse(s["path"])
        draw_cartoon(ax, co, ss)
        ax.set_title(
            rf"scRMSD $\,{s['scRMSD']:.1f}\;\mathrm{{\AA}}$", fontsize=9, pad=-2
        )
        if ci == 0:
            ax.text2D(
                -0.10,
                0.5,
                label,
                rotation=90,
                va="center",
                ha="center",
                transform=ax.transAxes,
                fontsize=12,
            )
fig.legend(
    [Line2D([0], [0], color=SS_COLOR[c], lw=5) for c in "abc"],
    ["helix", "sheet", "coil"],
    loc="lower center",
    ncol=3,
    fontsize=9,
    frameon=False,
)
fig.tight_layout(rect=[0.02, 0.05, 1, 1])
out = os.path.join(OUTDIR, "fig_ch3_designability.png")
fig.savefig(out, dpi=200)
print("wrote", out)
