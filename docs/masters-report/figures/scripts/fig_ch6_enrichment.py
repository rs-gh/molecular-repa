#!/usr/bin/env python3
"""Ch6 figure: beta-enrichment of the designable set on experimental (PDB) data.
Random designable backbones at 1.3M, baseline vs REPA-L9-MPNN, 6 each, as cartoons.
REPA adds beta-rich (gold) folds the baseline rarely produces (mean strand
fraction over designable L>=100: baseline 0.12 vs REPA 0.25).
Output: figures/fig_ch6_enrichment.png
"""

import os
import sys
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import use_report_style  # noqa: E402
from cartoon_util import draw_cartoon, ca_sse, SS_COLOR  # noqa: E402

use_report_style()
EVAL = "/home/sr2173/git/molecular-repa/eval_output"
OUTDIR = os.path.join(HERE, "..")
RNG = np.random.default_rng(2)
BASE = [
    "inference_paper_inference_fid_60m_paper_sweep_baseline_256_bs24_2gpu_step1300k_step_1300000__sde_n0.45__rep1",
    "inference_paper_inference_fid_60m_paper_sweep_baseline_256_bs24_2gpu_step1300k_step_1300000__sde_n0.45__rep2",
]
MPNN = [
    "inference_paper_inference_fid_60m_paper_sweep_repa_mpnn_l9_256_per_residue_step1300k_step_1300000__sde_n0.45__rep"
    + r
    for r in "012"
]


def load(dname):
    d = os.path.join(EVAL, dname)
    rows = list(csv.DictReader(open(os.path.join(d, "designability_index.csv"))))
    z = np.load(os.path.join(d, "ss_cache", "ss_fractions.npz"), allow_pickle=True)
    strand = {
        os.path.basename(str(p)): float(fr[1]) for p, fr in zip(z["paths"], z["fracs"])
    }
    return [
        dict(
            path=os.path.join(d, r["pdb_path"]),
            strand=strand.get(os.path.basename(r["pdb_path"]), np.nan),
            des=r["designable"] == "True",
            length=int(r["length"]),
        )
        for r in rows
    ]


def pool(dirs):
    out = []
    for dn in dirs:
        out += load(dn)
    return [x for x in out if x["des"] and x["length"] >= 100]


def pick(p, k=6):
    return [p[i] for i in RNG.permutation(len(p))[:k]]


base = pick(pool(BASE))
mpnn = pick(pool(MPNN))

fig = plt.figure(figsize=(9.4, 3.8))
for ri, (label, row) in enumerate([("Baseline", base), ("REPA\nL9-MPNN", mpnn)]):
    for ci, s in enumerate(row):
        ax = fig.add_subplot(2, 6, ri * 6 + ci + 1, projection="3d")
        co, ss = ca_sse(s["path"])
        draw_cartoon(ax, co, ss)
        ax.set_title(rf"$\beta{{=}}{s['strand']:.2f}$", fontsize=8, pad=-2)
        if ci == 0:
            ax.text2D(
                -0.12,
                0.5,
                label,
                rotation=90,
                va="center",
                ha="center",
                transform=ax.transAxes,
                fontsize=10,
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
out = os.path.join(OUTDIR, "fig_ch6_enrichment.png")
fig.savefig(out, dpi=200)
print("wrote", out)
