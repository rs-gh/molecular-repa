"""Representation quality over training — REPA paper Fig 3a analog.

Reads the consolidated ``pretrained_sweep_results.csv`` from the
n256_convergence_cath_if_dih_{pdb,afdb} sweeps and plots probe accuracy vs
training step. For each (run, step) we take the BEST layer's probe value
(peak-layer convention, matching the way the n128/n256 paper-struct tables
report ``best_layer`` for each ckpt).

Panel layout: 2 rows (PDB, AFDB) x 3 cols
    col 0: CATH-T accuracy (most discriminative level)
    col 1: CATH-C accuracy
    col 2: Inverse-folding top-1 accuracy

Baseline drawn first. REPA variants below. One line per run family.

Output: ``evaluation/proteina/representation/figures/paper/n256_convergence/repr_quality_over_training.{png,pdf}``
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # .../proteina/representation
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

RUN_FAMILIES = {
    "PDB": [
        ("baseline_256_bs24_2gpu", "Baseline (PDB)", "tab:blue", "-"),
        ("repa_l4_256_per_residue_bs24_2gpu", "REPA L4 GearNet (PDB)", "tab:red", "-"),
        (
            "repa_l9_256_per_residue_bs24_2gpu",
            "REPA L9 GearNet (PDB)",
            "tab:orange",
            "-",
        ),
        ("repa_mpnn_l4_256_per_residue", "REPA L4 MPNN (PDB)", "tab:red", "--"),
    ],
    "AFDB": [
        ("baseline_afdb_256", "Baseline (AFDB)", "tab:blue", "-"),
        ("repa_l4_afdb_256", "REPA L4 GearNet (AFDB)", "tab:red", "-"),
        ("repa_l9_afdb_256", "REPA L9 GearNet (AFDB, partial)", "tab:orange", ":"),
        ("repa_mpnn_l4_afdb_256", "REPA L4 MPNN (AFDB)", "tab:red", "--"),
        ("repa_mpnn_l9_afdb_256", "REPA L9 MPNN (AFDB)", "tab:orange", "--"),
    ],
}

# (probe_kind, filter_col, filter_val, metric_col, panel title, y label)
PANELS = [
    ("cath", "cath_level", "T", "cath_accuracy", "CATH-T accuracy", "probe acc"),
    ("cath", "cath_level", "C", "cath_accuracy", "CATH-C accuracy", "probe acc"),
    ("inverse_folding", None, None, "if_top1_acc", "IF top-1 acc", "probe acc"),
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
    rows: List[Dict],
    run_prefix: str,
    probe_kind: str,
    filter_col: str,
    filter_val: str,
    metric_col: str,
) -> Tuple[List[int], List[float]]:
    """For each step under run_prefix, pick the max metric across layers."""
    by_step: Dict[int, float] = defaultdict(lambda: float("-inf"))
    for r in rows:
        if r.get("probe_kind") != probe_kind:
            continue
        if filter_col is not None and r.get(filter_col) != filter_val:
            continue
        run = r.get("run", "")
        if not run.startswith(run_prefix):
            continue
        v = fnum(r.get(metric_col))
        s = fnum(r.get("step"))
        if v is None or s is None:
            continue
        s = int(s)
        if v > by_step[s]:
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
        figsize=(4.6 * len(PANELS), 3.8 * len(DATASETS)),
        sharex=False,
    )

    for row_i, (ds_name, csv_path) in enumerate(DATASETS.items()):
        rows = load_csv(csv_path)
        for col_i, (probe_kind, fcol, fval, metric, title, ylabel) in enumerate(PANELS):
            ax = axes[row_i, col_i]
            for prefix, label, color, ls in RUN_FAMILIES[ds_name]:
                xs, ys = best_per_ckpt(rows, prefix, probe_kind, fcol, fval, metric)
                if not xs:
                    continue
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=4,
                    linewidth=1.4,
                    color=color,
                    linestyle=ls,
                    label=label,
                )
            ax.set_xscale("log")
            ax.set_xlabel("Training step")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ds_name} — {title} (best layer)")
            ax.grid(True, alpha=0.3)
            if col_i == 0:
                ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        "n=256 convergence — representation probe accuracy vs training step\n"
        "Best layer per checkpoint; t=1.0; baseline drawn first.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = FIG_OUT / "repr_quality_over_training.png"
    out_pdf = FIG_OUT / "repr_quality_over_training.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
