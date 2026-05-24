"""Pareto plot focused on the two trade-offs that matter: designability vs
secondary-structure preservation, and designability vs novelty.

Layout (2×2, one row per data regime):
    row 0 (PDB):  Des vs SS-JSD-PDB(des)    Des vs Novelty-PDB
    row 1 (AFDB): Des vs SS-JSD-AFDB(des)   Des vs Novelty-AFDB-SP

Each panel:
  - Light scatter for every checkpoint (one marker per run, connected by step)
  - Bold blue front: baseline-only Pareto envelope
  - Bold red front:  best-of-all-REPA-variants Pareto envelope
  - "Up-right wins": SS-JSD axis is inverted so right = lower JSD = closer to
    reference. Repa-dominates iff red front lies above-right of blue.

Output: ``figures/paper/n256_convergence/pareto_des_div_nov.{png,pdf}``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/paper"
FIG_OUT = ROOT / "figures/paper/n256_convergence"
FIG_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "PDB": RESULTS / "n256_convergence_pdb" / "sweep_results.jsonl",
    "AFDB": RESULTS / "n256_convergence_afdb" / "sweep_results.jsonl",
}

# Per-regime: (baseline prefixes, list of REPA prefixes). The aggregate REPA
# front is computed across ALL REPA prefixes combined — point being to ask
# "does *any* REPA setting Pareto-dominate baseline?", not which one.
BASELINES = {"PDB": ["baseline_256_bs24_2gpu"], "AFDB": ["baseline_afdb_256"]}
REPA = {
    "PDB": [
        "repa_l4_256_per_residue_bs24_2gpu",
        "repa_l9_256_per_residue_bs24_2gpu",
        "repa_mpnn_l4_256_per_residue",
        "repa_mpnn_l9_256_per_residue",
    ],
    "AFDB": [
        "repa_l4_afdb_256",
        "repa_l9_afdb_256",
        "repa_mpnn_l4_afdb_256",
        "repa_mpnn_l9_afdb_256",
    ],
}
# Per-run scatter styling — colors/markers consistent with the convergence plots.
RUN_STYLE = {
    # PDB
    "baseline_256_bs24_2gpu": ("tab:blue", "s"),
    "repa_l4_256_per_residue_bs24_2gpu": ("tab:red", "o"),
    "repa_l9_256_per_residue_bs24_2gpu": ("tab:green", "o"),
    "repa_mpnn_l4_256_per_residue": ("tab:red", "^"),
    "repa_mpnn_l9_256_per_residue": ("tab:green", "^"),
    # AFDB
    "baseline_afdb_256": ("tab:blue", "s"),
    "repa_l4_afdb_256": ("tab:red", "o"),
    "repa_l9_afdb_256": ("tab:green", "o"),
    "repa_mpnn_l4_afdb_256": ("tab:red", "^"),
    "repa_mpnn_l9_afdb_256": ("tab:green", "^"),
}

# (x_col_template, x_label, x_higher_better). y is always designability.
# {ds} placeholder gets replaced with the row's regime.
PANELS = [
    ("_res_ss_jsd_{ds_lower}_designable_2d", "SS 2D-JSD vs {ds} (designable)", False),
    (
        "_res_novelty_foldseek_{nov_ref}_rate",
        "Novelty vs {nov_ref_label} (fraction novel)",
        True,
    ),
]
# Map each regime to its matching novelty target DB.
NOV_REFS = {"PDB": ("pdb", "PDB"), "AFDB": ("afdb_swissprot", "AFDB-SP")}


def _humanize(v, _pos=None):
    av = abs(v)
    if av >= 1:
        return f"{v:g}"
    return f"{v:.3g}"


def load_jsonl(path: Path) -> List[Dict]:
    rows = [json.loads(line) for line in path.open()]
    d: Dict[tuple, Dict] = {}
    for r in rows:
        d[(r.get("run"), r.get("step"))] = r
    return [r for r in d.values() if r.get("error", "NONE") == "NONE"]


def points_for(rows, prefixes, xcol, ycol):
    """Return list of (step, x, y) for any row whose run starts with one of prefixes."""
    pts = []
    for r in rows:
        run = r.get("run", "")
        if not any(run.startswith(p) for p in prefixes):
            continue
        x, y, s = r.get(xcol), r.get(ycol), r.get("step")
        if x is None or y is None or s is None:
            continue
        pts.append((int(s), float(x), float(y), run))
    return pts


def pareto_upper(pts_xy):
    """Upper Pareto front: both higher-is-better."""
    if not pts_xy:
        return []
    sorted_pts = sorted(pts_xy, reverse=True)  # by x desc
    front, best_y = [], float("-inf")
    for x, y in sorted_pts:
        if y > best_y:
            front.append((x, y))
            best_y = y
    front.sort()
    return front


def main() -> None:
    fig, axes = plt.subplots(
        nrows=len(DATASETS),
        ncols=len(PANELS),
        figsize=(5.6 * len(PANELS), 4.4 * len(DATASETS)),
    )
    for row_i, (ds, path) in enumerate(DATASETS.items()):
        rows = load_jsonl(path)
        nov_ref, nov_label = NOV_REFS[ds]
        for col_i, (xcol_tmpl, xl_tmpl, x_higher_better) in enumerate(PANELS):
            xcol = xcol_tmpl.format(ds_lower=ds.lower(), nov_ref=nov_ref)
            xl = xl_tmpl.format(ds=ds, nov_ref_label=nov_label)
            ax = axes[row_i, col_i]
            ycol = "_res_designability_rate"

            # All scatter
            all_pts = []  # (x, y) pooled across baseline + REPA, used only for axis lims
            for prefix, label, _kind in [
                *[(p, p, "baseline") for p in BASELINES[ds]],
                *[(p, p, "repa") for p in REPA[ds]],
            ]:
                pts = points_for(rows, [prefix], xcol, ycol)
                if not pts:
                    continue
                color, marker = RUN_STYLE.get(prefix, ("tab:gray", "x"))
                # connect step-ordered
                pts.sort()
                xs = [p[1] for p in pts]
                ys = [p[2] for p in pts]
                ax.plot(xs, ys, linewidth=0.8, color=color, alpha=0.25)
                ax.scatter(
                    xs,
                    ys,
                    s=30,
                    color=color,
                    marker=marker,
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.7,
                    zorder=3,
                    label=prefix,
                )
                all_pts.extend(zip(xs, ys))

            # Build the two aggregate fronts (baseline-only vs all REPA combined)
            base_pts = points_for(rows, BASELINES[ds], xcol, ycol)
            repa_pts = points_for(rows, REPA[ds], xcol, ycol)

            # For x_higher_better=False, negate x for sweep then flip back.
            def front_for(pts):
                xy = [(p[1], p[2]) for p in pts]
                if not x_higher_better:
                    xy = [(-x, y) for x, y in xy]
                front = pareto_upper(xy)
                if not x_higher_better:
                    front = sorted((-x, y) for x, y in front)
                return front

            base_front = front_for(base_pts)
            repa_front = front_for(repa_pts)
            if len(base_front) > 1:
                fx, fy = zip(*base_front)
                ax.plot(
                    fx,
                    fy,
                    linewidth=2.6,
                    color="tab:blue",
                    alpha=0.95,
                    zorder=5,
                    label="Baseline front",
                )
            if len(repa_front) > 1:
                fx, fy = zip(*repa_front)
                ax.plot(
                    fx,
                    fy,
                    linewidth=2.6,
                    color="tab:red",
                    alpha=0.95,
                    zorder=5,
                    label="REPA front (best of variants)",
                )

            arrow = " ↓" if not x_higher_better else " ↑"
            ax.set_xlabel(xl + arrow)
            ax.set_ylabel("Designability rate ↑")
            ax.set_title(f"{ds} — Designability vs {xl.split(' vs ')[0]}")
            if not x_higher_better:
                ax.invert_xaxis()  # right = better
            ax.grid(False)
            ax.grid(True, axis="both", alpha=0.25)
            ax.xaxis.set_major_formatter(FuncFormatter(_humanize))
            ax.yaxis.set_major_formatter(FuncFormatter(_humanize))
            if row_i == 0 and col_i == 0:
                # Only one legend on the figure, top-left panel, with just the two front lines
                handles = [
                    plt.Line2D(
                        [], [], color="tab:blue", linewidth=2.6, label="Baseline front"
                    ),
                    plt.Line2D(
                        [],
                        [],
                        color="tab:red",
                        linewidth=2.6,
                        label="REPA front (any variant)",
                    ),
                ]
                ax.legend(
                    handles=handles, loc="lower left", fontsize=8, framealpha=0.85
                )

    fig.suptitle(
        "n=256 — Designability trade-offs (right + up = better)\n"
        "Red front above blue → REPA Pareto-dominates baseline on that trade-off.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = FIG_OUT / "pareto_des_div_nov.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
