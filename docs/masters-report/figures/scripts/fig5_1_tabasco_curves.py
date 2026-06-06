"""Fig 5.1 -- Tabasco validation curves: no early separation under REPA.

Three structural-quality axes (the columns of Table 5.2's quality block),
plotted per training epoch for the baseline and the two additive REPA
variants the chapter discusses:

  Validity            -- fraction of generated molecules that are valid
  Connectivity        -- fraction fully connected
  Atom-type dist.     -- agreement of the atom-type histogram with data

The point is that all three curves pile together from the first few epochs:
REPA neither accelerates nor improves the saturated axes. We crop to the common
epoch range (every series reaches epoch 14), so the comparison is strictly
apples-to-apples. FCD -- the one axis with headroom -- is deliberately ABSENT:
it is not logged over training (only the final-checkpoint value in Table 5.2
exists), which is exactly the caveat the prose makes.

(The earlier scaffold TODO said validity/connectivity/novelty; we use
atom-type distribution instead of novelty, to match Table 5.2's quality block.)

DATA NOTE: this reads the per-epoch *validation* metrics (100 molecules per
epoch, from the in-training WandB callbacks) in
  evaluation/tabasco/generation/results/geom/validation/validation_curves.csv
These are a DIFFERENT measurement from the 1000-molecule, final-checkpoint
*evaluation* numbers in Table 5.2, so absolute values will not match the table
row-for-row. The figure's claim is about trajectories (no separation), not
endpoints. Regenerate the CSV with
  evaluation/tabasco/generation/scripts/geom/compile_wandb_curves.py
(needs WandB); this script only re-plots from the CSV (no network).
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))
from style import setup_axes, use_report_style

use_report_style()

ROOT = "/home/sr2173/git/molecular-repa"
CSV = f"{ROOT}/evaluation/tabasco/generation/results/geom/validation/validation_curves.csv"
OUT = f"{ROOT}/docs/masters-report/figures/fig5_1_tabasco_curves.png"

# Map the CSV's model identifiers to the chapter's two additive variants.
# REPA-CheMeleon = the "fused" additive run (single shared projector over the
# concatenated coord/atom streams -- the construction described in S5.2).
# REPA-MACE = the additive MACE run.
SERIES = [
    # (csv_model, label, colour, marker)
    ("baseline", "Baseline", "#1f77b4", "s"),
    ("chemeleon_additive_fused", "REPA-CheMeleon", "#d62728", "o"),
    ("mace_additive", "REPA-MACE", "#2ca02c", "^"),
]

# Panels: the three structural-quality axes of Table 5.2's quality block.
PANELS = [
    ("val/validity", "Validity"),
    ("val/connectivity", "Connectivity"),
    ("val/atom_type_distribution", "Atom-type distribution"),
]

# Crop to the epochs every series reaches, so the comparison is strictly
# apples-to-apples and the figure needs no "baseline ran longer" caveat. MACE
# stops at epoch 14, so 14 is the common range (baseline runs to 32, dropped).
MAX_EPOCH = 14


def load():
    by = defaultdict(list)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            by[r["model"]].append(r)
    for m in by:
        by[m].sort(key=lambda r: int(r["epoch"]))
    return by


def main():
    by = load()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    for ax, (key, title) in zip(axes, PANELS):
        for csv_model, label, colour, marker in SERIES:
            rows = [r for r in by.get(csv_model, []) if int(r["epoch"]) <= MAX_EPOCH]
            xs = [int(r["epoch"]) for r in rows]
            ys = [float(r[key]) for r in rows]
            ax.plot(
                xs,
                ys,
                "-",
                marker=marker,
                color=colour,
                label=label,
                linewidth=1.6,
                markersize=4,
                alpha=0.95,
            )
        setup_axes(
            ax, title=title, xlabel="Epoch", ylabel="$\\uparrow$ higher is better"
        )
        ax.set_xlim(left=0)
        # Epoch is integer-valued; force whole-number ticks (no 2.5, 7.5, ...).
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # One shared legend below the row.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(SERIES),
        fontsize=9,
        frameon=True,
        framealpha=0.85,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
