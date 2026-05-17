"""Convergence plot for designability-family metrics — proteina n=128.

Companion to ``plot_convergence_fid.py``. Covers designability, diversity,
novelty (vs PDB / AFDB-SP foldseek DBs), SS-JSD on the designable subset,
and the helix/sheet ratio. All metrics restricted to the designable subset
where applicable (paper App. F protocol).

Layout (2 rows × 7 cols, one row per dataset):
  Row 1 (PDB):  Des  Div  Nov-PDB  Nov-AFDB  SS-JSD-PDB  SS-JSD-AFDB  H/E
  Row 2 (AFDB): same

Same RUN_FAMILIES encoding (color = REPA layer depth, linestyle = data flavor,
marker = encoder family) and faded connecting lines as the other convergence
plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

import sys

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[5]))
from evaluation.proteina.lib import pretrained_overlay  # noqa: E402

# Overlay pretrained Proteina (NGC 60M, ~1.3M steps) as a horizontal dashed
# reference line where the pretrained model was evaluated. Not step-comparable.
# Flip to False to hide.
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
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(_humanize))
    ax.yaxis.set_major_formatter(FuncFormatter(_humanize))


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/paper"
FIG_OUT = ROOT / "figures/paper/n128_convergence"
FIG_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "PDB": RESULTS / "n128_convergence_pdb" / "sweep_results.jsonl",
    "AFDB": RESULTS / "n128_convergence_afdb" / "sweep_results.jsonl",
}

# Mirror plot_convergence_speedup.py — keep in sync by hand.
RUN_FAMILIES = {
    "PDB": [
        ("baseline_128_bs80", "Baseline (PDB)", "tab:blue", "-", "s"),
        (
            "repa_l4_128_bs80",
            "REPA L4 GearNet (PDB)",
            "tab:red",
            "-",
            "o",
        ),
        (
            "repa_l9_128_bs80",
            "REPA L9 GearNet (PDB)",
            "tab:orange",
            "-",
            "o",
        ),
        ("repa_mpnn_l4_128_bs80", "REPA L4 MPNN (PDB)", "tab:red", "--", "^"),
        ("repa_mpnn_l9_128_bs80_2gpu", "REPA L9 MPNN (PDB)", "tab:orange", "--", "^"),
        (
            "repa_l4_128_random",
            "REPA L4 GearNet-rand (PDB, ctrl)",
            "tab:gray",
            ":",
            "D",
        ),
    ],
    "AFDB": [
        ("baseline_afdb_128_bs80", "Baseline (AFDB)", "tab:blue", "-", "s"),
        ("repa_l4_afdb_128_bs80", "REPA L4 GearNet (AFDB)", "tab:red", "-", "o"),
        ("repa_mpnn_l4_afdb_128_bs80", "REPA L4 MPNN (AFDB)", "tab:red", "--", "^"),
        (
            "repa_mpnn_l9_afdb_128_bs80_2gpu",
            "REPA L9 MPNN (AFDB)",
            "tab:orange",
            "--",
            "^",
        ),
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


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.open()]
    dedup: Dict[tuple, Dict] = {}
    for r in rows:
        dedup[(r.get("run"), r.get("step"))] = r
    return list(dedup.values())


def extract_curve(rows, run_prefix, metric):
    if isinstance(metric, tuple):
        _, fn = metric
    else:
        fn = lambda r, _m=metric: r.get(_m)  # noqa: E731
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
        pts.append((int(s), float(v)))
    pts.sort()
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


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
                xs, ys = extract_curve(rows, prefix, metric)
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
                pre_val = pretrained_overlay.load_gen("n128").get(metric)
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
        "n=128 convergence — designability / diversity / novelty / SS metrics\n"
        "All restricted to the designable subset where applicable; baseline drawn first.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = FIG_OUT / "convergence_des.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
