"""Convergence plot for FID-family distributional metrics — proteina n=256.

Companion to ``plot_convergence_des.py``. Plots reference-distribution
metrics that aren't covered there: FID (overall coord-VAE feature dist),
fJSD (CATH fold-class JSD at C/A/T levels), and fS (CATH fold-class
Shannon entropy). All "lower-closer-to-reference" for fJSD; fS has no
single "good" direction but tracks coverage of fold classes.

Layout (2 rows × 7 cols, one row per dataset):
  Row 1 (PDB):  FID  fJSD_A  fJSD_C  fJSD_T  fS_C  fS_A  fS_T
  Row 2 (AFDB): FID  fJSD_A  fJSD_C  fJSD_T  fS_C  fS_A  fS_T

fS is dataset-agnostic at the column level but plotted within each data-regime
row so comparisons stay within {baseline vs REPA on PDB} or {... on AFDB}.

Reads the same two ``sweep_results.jsonl`` files and uses the same
RUN_FAMILIES color/linestyle/marker convention as plot_convergence_des.
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
# reference line on the FID panel(s). fJSD/fS aren't logged for the pretrained
# eval so those panels skip the overlay. Flip to False to hide.
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
FIG_OUT = ROOT / "figures/paper/n256_convergence/single_seed_42"
FIG_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "PDB": RESULTS / "n256_convergence_pdb" / "sweep_results.jsonl",
    "AFDB": RESULTS / "n256_convergence_afdb" / "sweep_results.jsonl",
}

# Mirror plot_convergence_des.py — same encoding, kept in sync by hand.
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
        ("repa_mpnn_l9_256_per_residue", "REPA L9 MPNN (PDB)", "tab:green", "--", "^"),
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
        ("repa_l9_afdb_256", "REPA L9 GearNet (AFDB, partial)", "tab:green", ":", "o"),
        ("repa_mpnn_l4_afdb_256", "REPA L4 MPNN (AFDB)", "tab:red", "--", "^"),
        ("repa_mpnn_l9_afdb_256", "REPA L9 MPNN (AFDB)", "tab:green", "--", "^"),
    ],
}

# FID + fJSD metrics; each is evaluated against BOTH the PDB and AFDB reference
# sets, so every dataset row gets 8 reference-conditional panels.
# Tuples: (suffix-in-column-name, panel-title-without-reference). The reference
# label ("PDB" / "AFDB") is added at plotting time.
DATASET_METRICS = [
    ("FID", "FID-1.1K"),
    ("fJSD_A", "fJSD (Architecture)"),
    ("fJSD_C", "fJSD (Class)"),
    ("fJSD_T", "fJSD (Topology)"),
]
REFERENCES = ["PDB", "AFDB"]

# Dataset-agnostic fold-class entropy panels.
FS_METRICS = [
    ("_res_fS_C", "fS (Class)"),
    ("_res_fS_A", "fS (Architecture)"),
    ("_res_fS_T", "fS (Topology)"),
]


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.open()]
    dedup: Dict[tuple, Dict] = {}
    for r in rows:
        dedup[(r.get("run"), r.get("step"))] = r
    return list(dedup.values())


def extract(rows, run_prefix, col):
    pts = []
    for r in rows:
        run = r.get("run", "")
        if not run.startswith(run_prefix):
            continue
        if "_step" not in run.split(run_prefix)[-1] and run != run_prefix:
            continue
        if r.get("error", "NONE") != "NONE":
            continue
        v = r.get(col)
        s = r.get("step")
        if v is None or s is None:
            continue
        pts.append((int(s), float(v)))
    pts.sort()
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def plot_curve(ax, xs, ys, color, ls, marker, label):
    ax.plot(xs, ys, linewidth=1.4, color=color, linestyle=ls, alpha=0.35, marker="none")
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


def main() -> None:
    # Each FID/fJSD metric is shown against both PDB and AFDB references, so the
    # dataset-conditional column count doubles.
    n_cols = len(DATASET_METRICS) * len(REFERENCES) + len(FS_METRICS)
    fig, axes = plt.subplots(
        nrows=len(DATASETS),
        ncols=n_cols,
        figsize=(3.8 * n_cols, 3.6 * len(DATASETS)),
        sharex=False,
    )

    for row_i, (ds_name, jsonl_path) in enumerate(DATASETS.items()):
        rows = load_jsonl(jsonl_path)
        # FID + fJSD columns × (PDB ref, AFDB ref)
        col_i = 0
        for suffix, title in DATASET_METRICS:
            for ref in REFERENCES:
                ax = axes[row_i, col_i]
                col = f"_res_{ref}_{suffix}"
                for prefix, label, color, ls, marker in RUN_FAMILIES[ds_name]:
                    xs, ys = extract(rows, prefix, col)
                    if not xs:
                        continue
                    plot_curve(ax, xs, ys, color, ls, marker, label)
                ax.set_xscale("log")
                ax.set_yscale(
                    "log"
                )  # FID/fJSD have wide dynamic range; log keeps late-regime separation visible
                ax.set_xlabel("Training step")
                ax.set_ylabel("value (lower = closer, log y)")
                ax.set_title(f"{ds_name} train — {title} vs {ref} ↓")
                _style_axes(ax, log_y=True)
                if col_i == 0:
                    ax.legend(loc="best", fontsize=7)
                col_i += 1
        # fS columns — same row's RUN_FAMILIES so comparisons stay within data regime
        for j, (col, title) in enumerate(FS_METRICS):
            ax = axes[row_i, col_i + j]
            for prefix, label, color, ls, marker in RUN_FAMILIES[ds_name]:
                xs, ys = extract(rows, prefix, col)
                if not xs:
                    continue
                plot_curve(ax, xs, ys, color, ls, marker, label)
            ax.set_xscale("log")
            ax.set_xlabel("Training step")
            ax.set_ylabel("entropy (higher = more coverage)")
            ax.set_title(f"{ds_name} — {title} ↑")
            _style_axes(ax)
            if SHOW_PRETRAINED:
                pre_val = pretrained_overlay.load_gen().get(col)
                if pre_val is not None:
                    ax.axhline(
                        pre_val,
                        color=pretrained_overlay.PRETRAINED_COLOR,
                        linestyle="--",
                        linewidth=2.6,
                        alpha=0.9,
                        zorder=1,
                    )

    fig.suptitle(
        "n=256 convergence — FID-family distributional metrics\n"
        "FID/fJSD: lower = closer to reference (log y). fS: fold-class entropy (linear, higher = more coverage).",
        fontsize=12,
    )
    # Figure-level legend at bottom; de-dup labels across axes.
    handles, labels = [], []
    seen = set()
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab in seen:
                continue
            seen.add(lab)
            handles.append(h)
            labels.append(lab)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        bbox_to_anchor=(0.5, -0.01),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    out_png = FIG_OUT / "convergence_fid.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
