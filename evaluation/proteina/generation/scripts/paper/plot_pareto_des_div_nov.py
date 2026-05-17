"""Protein-specific Pareto frontier — designability ↔ diversity ↔ novelty.

Reads the n256_convergence_{pdb,afdb} gen sweep_results and produces a 2x2
panel:
    row 0: PDB,  row 1: AFDB
    col 0: designability vs diversity   (higher-higher = top-right wins)
    col 1: designability vs novelty     (higher-higher = top-right wins)

Each checkpoint is one scatter point; lines connect same-run checkpoints in
step order. Per-method (baseline / each REPA variant) Pareto frontiers drawn
as bold step lines so REPA-dominates-baseline behaviour is visible at a glance.

Output: ``evaluation/proteina/generation/figures/paper/n256_convergence/pareto_des_div_nov.{png,pdf}``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # .../proteina/generation
RESULTS = ROOT / "results/paper"
FIG_OUT = ROOT / "figures/paper/n256_convergence"
FIG_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "PDB": RESULTS / "n256_convergence_pdb" / "sweep_results.jsonl",
    "AFDB": RESULTS / "n256_convergence_afdb" / "sweep_results.jsonl",
}

RUN_FAMILIES = {
    "PDB": [
        ("baseline_256_bs24_2gpu", "Baseline (PDB)", "tab:blue", "o"),
        ("repa_l4_256_per_residue_bs24_2gpu", "REPA L4 GearNet (PDB)", "tab:red", "s"),
        (
            "repa_l9_256_per_residue_bs24_2gpu",
            "REPA L9 GearNet (PDB)",
            "tab:orange",
            "s",
        ),
        ("repa_mpnn_l4_256_per_residue", "REPA L4 MPNN (PDB)", "tab:red", "^"),
    ],
    "AFDB": [
        ("baseline_afdb_256", "Baseline (AFDB)", "tab:blue", "o"),
        ("repa_l4_afdb_256", "REPA L4 GearNet (AFDB)", "tab:red", "s"),
        ("repa_l9_afdb_256", "REPA L9 GearNet (AFDB)", "tab:orange", "s"),
        ("repa_mpnn_l4_afdb_256", "REPA L4 MPNN (AFDB)", "tab:red", "^"),
        ("repa_mpnn_l9_afdb_256", "REPA L9 MPNN (AFDB)", "tab:orange", "^"),
    ],
}

# (x_metric, y_metric, x_label, y_label, x_higher_better)
# When x_higher_better=False (e.g. SS-JSD: lower is closer to reference) we
# negate the x value for plotting and Pareto-front computation so the "good"
# direction is always to the right and the same upper-Pareto sweep works.
PANELS = [
    (
        "_res_diversity_clusters_total",
        "_res_designability_rate",
        "Diversity (# clusters, designable subset)",
        "Designability rate",
        True,
    ),
    (
        "_res_novelty_foldseek_pdb_rate",
        "_res_designability_rate",
        "Novelty vs PDB (fraction novel)",
        "Designability rate",
        True,
    ),
    (
        "_res_novelty_foldseek_afdb_swissprot_rate",
        "_res_designability_rate",
        "Novelty vs AFDB-Swissprot (fraction novel)",
        "Designability rate",
        True,
    ),
    (
        "_res_ss_jsd_pdb_designable_2d",
        "_res_designability_rate",
        "SS 2D-JSD vs PDB (designable, lower=closer)",
        "Designability rate",
        False,
    ),
    (
        "_res_ss_jsd_afdb_designable_2d",
        "_res_designability_rate",
        "SS 2D-JSD vs AFDB (designable, lower=closer)",
        "Designability rate",
        False,
    ),
]


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.open()]
    dedup: Dict[tuple, Dict] = {}
    for r in rows:
        dedup[(r.get("run"), r.get("step"))] = r
    return [r for r in dedup.values() if r.get("error", "NONE") == "NONE"]


def extract_points(
    rows: List[Dict], run_prefix: str, xm: str, ym: str
) -> List[Tuple[int, float, float]]:
    pts = []
    for r in rows:
        run = r.get("run", "")
        if not run.startswith(run_prefix):
            continue
        x = r.get(xm)
        y = r.get(ym)
        s = r.get("step")
        if x is None or y is None or s is None:
            continue
        pts.append((int(s), float(x), float(y)))
    pts.sort()
    return pts


def pareto_front(pts: List[Tuple[int, float, float]]) -> List[Tuple[float, float]]:
    """Upper Pareto frontier (both axes higher = better)."""
    if not pts:
        return []
    # sort by x descending so we sweep right-to-left maintaining max y
    xs_ys = sorted([(p[1], p[2]) for p in pts], reverse=True)
    front = []
    best_y = float("-inf")
    for x, y in xs_ys:
        if y > best_y:
            front.append((x, y))
            best_y = y
    front.sort()  # left-to-right for plotting
    return front


def main() -> None:
    fig, axes = plt.subplots(
        nrows=len(DATASETS),
        ncols=len(PANELS),
        figsize=(5.0 * len(PANELS), 4.0 * len(DATASETS)),
        sharex=False,
    )
    for row_i, (ds_name, path) in enumerate(DATASETS.items()):
        rows = load_jsonl(path)
        for col_i, (xm, ym, xl, yl, x_higher_better) in enumerate(PANELS):
            ax = axes[row_i, col_i]
            for prefix, label, color, marker in RUN_FAMILIES[ds_name]:
                pts = extract_points(rows, prefix, xm, ym)
                if not pts:
                    continue
                xs = [p[1] for p in pts]
                ys = [p[2] for p in pts]
                sizes = [22 + 0.00007 * p[0] for p in pts]
                # connect by step order with a thin alpha line
                ax.plot(xs, ys, linewidth=0.8, color=color, alpha=0.4)
                ax.scatter(
                    xs,
                    ys,
                    s=sizes,
                    color=color,
                    marker=marker,
                    edgecolor="black",
                    linewidth=0.4,
                    label=label,
                    zorder=3,
                )
                # per-method Pareto front. When x is "lower=better", negate x
                # so pareto_front's higher-higher sweep still applies, then
                # flip back for plotting.
                if x_higher_better:
                    front = pareto_front(pts)
                else:
                    flipped = [(s, -x, y) for (s, x, y) in pts]
                    front_flipped = pareto_front(flipped)
                    front = [(-x, y) for (x, y) in front_flipped]
                    front.sort()
                if len(front) > 1:
                    fx, fy = zip(*front)
                    ax.plot(fx, fy, linewidth=2.0, color=color, alpha=0.8, zorder=2)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            if not x_higher_better:
                ax.invert_xaxis()  # visually right = better even for low-is-better axes
            ax.set_title(f"{ds_name}: {yl.split(' ')[0]} vs {xl.split(' ')[0]}")
            ax.grid(True, alpha=0.3)
            if col_i == 0:
                ax.legend(loc="best", fontsize=7)
    fig.suptitle(
        "n=256 — Pareto frontiers (point size ∝ training step; thick line per method)\n"
        "Top-right wins; REPA frontiers above-right of baseline = Pareto improvement.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = FIG_OUT / "pareto_des_div_nov.png"
    out_pdf = FIG_OUT / "pareto_des_div_nov.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
