"""Representation probe quality over training — n=256 convergence sweep.

Reads the consolidated ``pretrained_sweep_results.csv`` from the
n256_convergence_cath_if_dih_{pdb,afdb} sweeps. For each (run, step) we take
the BEST layer's probe value (max for higher-is-better metrics, min for MAE).

Panel layout: 2 rows (PDB, AFDB) × 6 cols
    CATH-C acc ↑   CATH-A acc ↑   CATH-T acc ↑   IF top-1 ↑   IF macro-F1 ↑   Dih MAE (total) ↓

Same styling as the gen-side convergence plots: per-encoder marker shape,
faded connecting lines, horizontal-only grid, human-friendly tick labels.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

import sys

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[5]))
from evaluation.proteina.lib import pretrained_overlay  # noqa: E402

# Overlay pretrained Proteina (NGC 60M, ~1.3M steps) as a horizontal dashed
# reference line on each probe panel. Not step-comparable. Flip to False to
# hide.
SHOW_PRETRAINED = True

# Map (probe_kind, metric_col) → pretrained_overlay rep key. cath panels are
# handled separately because they need the level (C/A/T) too.
_PRETRAINED_REP_KEY = {
    ("inverse_folding", "if_top1_acc"): "if_top1_acc",
    ("dihedral", "dih_mae_total_deg"): "dih_mae_total_deg",
}


def _humanize(v, _pos=None):
    av = abs(v)
    if av >= 1e6:
        return f"{v / 1e6:g}M"
    if av >= 1e3:
        return f"{v / 1e3:g}K"
    if av >= 1:
        return f"{v:g}"
    return f"{v:.3g}"


def _style_axes(ax):
    ax.grid(False)
    ax.grid(True, axis="y", alpha=0.3)
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
    "PDB": RESULTS
    / "n256_convergence_cath_if_dih_pdb"
    / "pretrained_sweep_results.csv",
    "AFDB": RESULTS
    / "n256_convergence_cath_if_dih_afdb"
    / "pretrained_sweep_results.csv",
}

# Mirror the gen-side plots (color, linestyle, marker).
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

# (probe_kind, filter_col, filter_val, metric_col, panel title, y label, higher_better)
PANELS = [
    ("cath", "cath_level", "C", "cath_accuracy", "CATH-C accuracy", "probe acc", True),
    ("cath", "cath_level", "A", "cath_accuracy", "CATH-A accuracy", "probe acc", True),
    ("cath", "cath_level", "T", "cath_accuracy", "CATH-T accuracy", "probe acc", True),
    ("inverse_folding", None, None, "if_top1_acc", "IF top-1 acc", "probe acc", True),
    ("inverse_folding", None, None, "if_macro_f1", "IF macro-F1", "macro F1", True),
    (
        "dihedral",
        None,
        None,
        "dih_mae_total_deg",
        "Dihedral MAE (total)",
        "MAE (deg)",
        False,
    ),
]


def load_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def best_per_ckpt(
    rows, run_prefix, probe_kind, filter_col, filter_val, metric_col, higher_better
):
    """Per step under run_prefix: best (max if higher_better else min) across layers."""
    init = float("-inf") if higher_better else float("inf")
    by_step: Dict[int, float] = defaultdict(lambda: init)
    for r in rows:
        if r.get("probe_kind") != probe_kind:
            continue
        if filter_col is not None and r.get(filter_col) != filter_val:
            continue
        run = r.get("run", "")
        if not run.startswith(run_prefix):
            continue
        v, s = fnum(r.get(metric_col)), fnum(r.get("step"))
        if v is None or s is None:
            continue
        s = int(s)
        if (higher_better and v > by_step[s]) or (not higher_better and v < by_step[s]):
            by_step[s] = v
    pts = sorted(by_step.items())
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def main() -> None:
    fig, axes = plt.subplots(
        nrows=len(DATASETS),
        ncols=len(PANELS),
        figsize=(3.8 * len(PANELS), 3.6 * len(DATASETS)),
        sharex=False,
    )
    for row_i, (ds_name, csv_path) in enumerate(DATASETS.items()):
        rows = load_csv(csv_path)
        for col_i, (probe_kind, fcol, fval, metric, title, ylabel, hb) in enumerate(
            PANELS
        ):
            ax = axes[row_i, col_i]
            for prefix, label, color, ls, marker in RUN_FAMILIES[ds_name]:
                xs, ys = best_per_ckpt(rows, prefix, probe_kind, fcol, fval, metric, hb)
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
            arrow = " ↑" if hb else " ↓"
            ax.set_title(f"{ds_name} — {title}{arrow}")
            _style_axes(ax)
            if SHOW_PRETRAINED:
                # Pick the right pretrained overlay key for this panel.
                if probe_kind == "cath" and fval in ("C", "A", "T"):
                    pre_key = f"cath_{fval}_top1"
                else:
                    pre_key = _PRETRAINED_REP_KEY.get((probe_kind, metric))
                pre_val = (
                    pretrained_overlay.load_rep(ds_name).get(pre_key)
                    if pre_key
                    else None
                )
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
        "n=256 convergence — representation probe quality vs training step\n"
        "Best layer per checkpoint (max for ↑ metrics, min for ↓); t=1.0; baseline drawn first.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = FIG_OUT / "repr_quality_over_training.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
