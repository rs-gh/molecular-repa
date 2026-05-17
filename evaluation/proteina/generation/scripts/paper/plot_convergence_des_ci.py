"""Convergence des-family plot with Wilson 95% CIs on rate metrics.

Same panels as plot_convergence_des.py; error bars added on the designability
and novelty (foldseek) panels using each row's per-checkpoint denominator.
CIs are statistical-noise bands for "given these samples, where does the
true rate lie" — they do NOT include sample-from-model variance (would
require multi-seed re-eval).

Output: ``figures/paper/n256_convergence/convergence_des_ci.png``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

import sys

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[5]))
from evaluation.proteina.lib import pretrained_overlay  # noqa: E402

# Overlay pretrained Proteina (NGC 60M, ~1.3M steps) as a horizontal dashed
# reference line. Not step-comparable to our runs. Flip to False to hide.
SHOW_PRETRAINED = True


def _humanize(v, _pos=None):
    av = abs(v)
    if av >= 1e6:
        return f"{v / 1e6:g}M"
    if av >= 1e3:
        return f"{v / 1e3:g}K"
    if av >= 1:
        return f"{v:g}"
    return f"{v:.3g}"


def _style_axes(ax, log_y: bool = False) -> None:
    ax.grid(False)
    ax.grid(True, axis="y", alpha=0.3, which="both" if log_y else "major")
    ax.xaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0, 2.0, 4.0, 7.0), numticks=20)
    )
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=20))
    ax.xaxis.set_major_formatter(FuncFormatter(_humanize))
    ax.yaxis.set_major_formatter(FuncFormatter(_humanize))


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/paper"
FIG_OUT = ROOT / "figures/paper/n256_convergence"
FIG_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "PDB": RESULTS / "n256_convergence_pdb" / "sweep_results.jsonl",
    "AFDB": RESULTS / "n256_convergence_afdb" / "sweep_results.jsonl",
}

# Mirror plot_convergence_speedup.py — keep in sync by hand.
RUN_FAMILIES = {
    "PDB": [
        ("baseline_256_bs24_2gpu", "Baseline (PDB)", "tab:blue", "-", "s"),
        (
            "repa_l4_256_per_residue_bs24_2gpu",
            "REPA L4 GearNet (PDB)",
            "tab:red",
            "-",
            "o",
        ),
        (
            "repa_l9_256_per_residue_bs24_2gpu",
            "REPA L9 GearNet (PDB)",
            "tab:orange",
            "-",
            "o",
        ),
        ("repa_mpnn_l4_256_per_residue", "REPA L4 MPNN (PDB)", "tab:red", "--", "^"),
        ("repa_mpnn_l9_256_per_residue", "REPA L9 MPNN (PDB)", "tab:orange", "--", "^"),
        (
            "repa_l4_256_per_residue_random_bs24_2gpu",
            "REPA L4 GearNet-rand (PDB, ctrl)",
            "tab:gray",
            ":",
            "D",
        ),
    ],
    "AFDB": [
        ("baseline_afdb_256", "Baseline (AFDB)", "tab:blue", "-", "s"),
        ("repa_l4_afdb_256", "REPA L4 GearNet (AFDB)", "tab:red", "-", "o"),
        ("repa_l9_afdb_256", "REPA L9 GearNet (AFDB, partial)", "tab:orange", ":", "o"),
        ("repa_mpnn_l4_afdb_256", "REPA L4 MPNN (AFDB)", "tab:red", "--", "^"),
        ("repa_mpnn_l9_afdb_256", "REPA L9 MPNN (AFDB)", "tab:orange", "--", "^"),
    ],
}

# (metric, panel title, y-axis label, higher_is_better). metric is a column
# name (str) or (label, fn) for derived metrics (matches convergence_speedup).
METRICS = [
    ("_res_designability_rate", "Designability", "rate", True),
    # Raw cluster count; rate clusters/n_designable saturates at 1.0 for
    # small designable sets, so we keep raw counts despite the designability
    # confound. See plot_convergence_speedup.py for the same note.
    ("_res_diversity_clusters_total", "Diversity (clusters)", "# clusters (des)", True),
    # Continuous diversity: mean pairwise TM within designable pool. Doesn't
    # saturate at small n the way clusters/n does. Lower = more diverse.
    (
        "_res_diversity_pairwise_tm_mean",
        "Diversity (pairwise TM)",
        "mean pairwise TM (des)",
        False,
    ),
    (
        "_res_novelty_foldseek_pdb_rate",
        "Novelty vs PDB",
        "fraction novel (TM<0.5)",
        True,
    ),
    (
        "_res_novelty_foldseek_afdb_swissprot_rate",
        "Novelty vs AFDB-SP",
        "fraction novel (TM<0.5)",
        True,
    ),
    ("_res_ss_jsd_pdb_designable_2d", "SS 2D-JSD vs PDB (des)", "JSD", False),
    ("_res_ss_jsd_afdb_designable_2d", "SS 2D-JSD vs AFDB (des)", "JSD", False),
    (
        (
            "H/E",
            lambda r: (r.get("_res_ss_frac_H_designable") or 0) / e
            if (e := r.get("_res_ss_frac_E_designable") or 0) > 0
            else None,
        ),
        "H/E ratio (des)",
        "H/E",
        True,  # arrow suppressed in title for H/E
    ),
]


# Binomial-rate metric -> denominator column (used for Wilson CIs).
BINOMIAL_DENOM = {
    "_res_designability_rate": "_res_designability_n",
    "_res_novelty_foldseek_pdb_rate": "_res_novelty_foldseek_pdb_n",
    "_res_novelty_foldseek_afdb_swissprot_rate": "_res_novelty_foldseek_afdb_swissprot_n",
}
Z95 = 1.959963984540054


def wilson_ci(p: float, n: int, z: float = Z95):
    if n <= 0:
        return p, p
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def is_binomial(metric) -> bool:
    return isinstance(metric, str) and metric in BINOMIAL_DENOM


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.open()]
    dedup: Dict[tuple, Dict] = {}
    for r in rows:
        dedup[(r.get("run"), r.get("step"))] = r
    return list(dedup.values())


def extract_curve(rows, run_prefix, metric):
    """Returns (xs, ys, los, his); los/his are Wilson CI bounds for binomial
    metrics, else equal to y (zero-width)."""
    if isinstance(metric, tuple):
        _, fn = metric
        denom_col = None
    else:
        fn = lambda r, _m=metric: r.get(_m)  # noqa: E731
        denom_col = BINOMIAL_DENOM.get(metric)
    pts = []
    for r in rows:
        run = r.get("run", "")
        if not run.startswith(run_prefix):
            continue
        if "_step" not in run.split(run_prefix)[-1] and run != run_prefix:
            continue
        if r.get("error", "NONE") != "NONE":
            continue
        v = fn(r)
        s = r.get("step")
        if v is None or s is None:
            continue
        v = float(v)
        if denom_col is not None:
            n = r.get(denom_col)
            lo, hi = wilson_ci(v, int(n)) if n is not None else (v, v)
        else:
            lo, hi = v, v
        pts.append((int(s), v, lo, hi))
    pts.sort()
    if not pts:
        return [], [], [], []
    xs, ys, los, his = zip(*pts)
    return list(xs), list(ys), list(los), list(his)


def main() -> None:
    fig, axes = plt.subplots(
        nrows=len(DATASETS),
        ncols=len(METRICS),
        figsize=(3.8 * len(METRICS), 3.6 * len(DATASETS)),
        sharex=False,
    )
    for row_i, (ds_name, jsonl_path) in enumerate(DATASETS.items()):
        rows = load_jsonl(jsonl_path)
        for col_i, (metric, title, ylabel, higher_better) in enumerate(METRICS):
            ax = axes[row_i, col_i]
            for prefix, label, color, ls, marker in RUN_FAMILIES[ds_name]:
                xs, ys, los, his = extract_curve(rows, prefix, metric)
                if not xs:
                    continue
                ax.plot(
                    xs,
                    ys,
                    linewidth=1.4,
                    color=color,
                    linestyle=ls,
                    alpha=0.35,
                    marker="none",
                )
                if is_binomial(metric):
                    y_err_lo = [max(0.0, y - lo) for y, lo in zip(ys, los)]
                    y_err_hi = [max(0.0, hi - y) for y, hi in zip(ys, his)]
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=[y_err_lo, y_err_hi],
                        fmt="none",
                        ecolor=color,
                        elinewidth=1.0,
                        capsize=2.5,
                        capthick=0.8,
                        alpha=0.55,
                        zorder=2,
                    )
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    markersize=6,
                    markeredgewidth=0.8,
                    markeredgecolor="white",
                    linestyle="none",
                    color=color,
                    label=label,
                )
            ax.set_xscale("log")
            ax.set_xlabel("Training step")
            ax.set_ylabel(ylabel)
            if "H/E" in title:
                arrow = ""
            else:
                arrow = " ↑" if higher_better else " ↓"
            ax.set_title(f"{ds_name} — {title}{arrow}")
            _style_axes(ax)
            if SHOW_PRETRAINED and isinstance(metric, str):
                pre_val = pretrained_overlay.load_gen().get(metric)
                if pre_val is not None:
                    ax.axhline(
                        pre_val,
                        color=pretrained_overlay.PRETRAINED_COLOR,
                        linestyle="--",
                        linewidth=2.6,
                        alpha=0.9,
                        label=pretrained_overlay.PRETRAINED_LABEL
                        if col_i == 0
                        else None,
                        zorder=1,
                    )
            if col_i == 0:
                ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        "n=256 convergence — des/div/nov/SS (Wilson 95% CIs on rates)\n"
        "Error bars: statistical CI given the samples drawn; do NOT capture sample-from-model variance.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = FIG_OUT / "convergence_des_ci.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
