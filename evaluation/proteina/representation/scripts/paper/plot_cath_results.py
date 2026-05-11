"""Plots for the paper-table CATH probe sweeps (paper_n128_cath / paper_n256_cath).

Reads ``pretrained_sweep_results.jsonl`` from one of the paper-CATH output
dirs and emits per-CATH-level layer-curve figures, plus one ablation-block
sub-figure per paper-table block (layer / encoder / bs+lr / λ / wd).

Usage:
  python evaluation/proteina/representation/scripts/paper/plot_cath_results.py \\
      --sweep paper_n128_cath
  python evaluation/proteina/representation/scripts/paper/plot_cath_results.py \\
      --sweep paper_n256_cath

Outputs land in ``figures/paper/n{128,256}_paper_cath/``:
  fig_layer_curves.png   — 3 panels (cath-C / A / T) × all runs as colored lines
  fig_block_<name>.png   — one figure per ablation block, 3 panels
  table_peak.csv         — peak accuracy per (run, level) for quick lookup
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
REPR_ROOT = HERE.parents[1]
RESULTS_ROOT = REPR_ROOT / "results"
FIG_ROOT = REPR_ROOT / "figures" / "paper"


# Per-paper-table ablation blocks. Each block has a list of `(run_key, label)`
# tuples. The first entry is the reference (baseline) for that block.
BLOCKS_N128: Dict[str, List[tuple]] = {
    "layer": [
        ("baseline_128_bs80_step200k", "baseline"),
        ("repa_l0_128_bs80_step200k", "REPA L0"),
        ("repa_l4_128_bs80_step200k", "REPA L4"),
        ("repa_l9_128_bs80_step200k", "REPA L9"),
    ],
    "layer_per_residue_step400k": [
        ("baseline_128_bs24_step400k", "baseline (bs24, 400k)"),
        ("repa_l0_128_per_residue_step400k", "REPA L0"),
        ("repa_l4_128_per_residue_step400k", "REPA L4"),
        ("repa_l9_128_per_residue_step400k", "REPA L9"),
    ],
    "encoder": [
        ("baseline_128_bs80_step200k", "baseline"),
        ("repa_l4_128_bs80_step200k", "L4 / CA-GearNet"),
        ("repa_l4_128_random_step200k", "L4 / random init"),
        ("repa_l4_128_pw_structure_step100k", "L4 / PW-Structure"),
        ("repa_l4_128_pw_torsional_step100k", "L4 / PW-Torsional"),
        ("repa_mpnn_l4_128_bs80_step200k", "L4 / ProteinMPNN"),
        ("repa_esm_l4_128_step200k", "L4 / ESM2"),
    ],
    "bs_lr": [
        ("baseline_128_bs24_step200k", "baseline bs24 200k"),
        ("baseline_128_bs24_step400k", "baseline bs24 400k"),
        ("baseline_128_bs80_step200k", "baseline bs80 200k"),
        ("baseline_128_bs80_lr3x_step200k", "baseline bs80 lr3x"),
        ("repa_l4_128_bs24_step200k", "REPA L4 bs24 200k"),
        ("repa_l4_128_bs24_step400k", "REPA L4 bs24 400k"),
        ("repa_l4_128_bs80_step200k", "REPA L4 bs80 200k"),
        ("repa_l4_128_bs80_lr3x_steplast", "REPA L4 bs80 lr3x"),
    ],
    "lambda_wd": [
        ("repa_l4_128_bs80_step200k", "REPA L4 λ=0.5 wd_def"),
        ("repa_l4_128_bs80_lambda2_steplast", "REPA L4 λ=2.0"),
        ("repa_l4_128_bs80_wd1e2_step200k", "REPA L4 wd=1e-2"),
    ],
    "pretrained_vs_ours": [
        ("baseline_128_bs80_step200k", "ours (10L) baseline"),
        ("repa_l4_128_bs80_step200k", "ours (10L) REPA L4"),
        ("pretrained_dfs_60m", "NGC pretrained (12L)"),
    ],
}

BLOCKS_N256: Dict[str, List[tuple]] = {
    "layer": [
        ("baseline_256_ep21", "baseline"),
        ("repa_l0_256_ep26", "REPA L0"),
        ("repa_l4_256_ep22", "REPA L4"),
        ("repa_l9_256_ep25", "REPA L9"),
    ],
    "encoder": [
        ("baseline_256_ep21", "baseline"),
        ("repa_l4_256_ep22", "L4 / CA-GearNet"),
        ("repa_l4_256_random_ep17", "L4 / random init"),
        ("repa_mpnn_l4_256_per_residue_step300k", "L4 / ProteinMPNN"),
        ("repa_esm_l9_t30_256_steplast", "L9 / ESM2 (≠L4)"),
    ],
    "dataset": [
        ("baseline_256_ep21", "PDB baseline"),
        ("baseline_afdb_256_ep20", "AFDB baseline"),
        ("repa_l4_256_ep22", "PDB REPA L4"),
        ("repa_l4_afdb_256_ep20", "AFDB REPA L4"),
    ],
    "averaging": [
        ("repa_l0_256_ep26", "L0 per_residue"),
        ("repa_l0_256_per_sample_steplast", "L0 per_sample"),
        ("repa_l4_256_ep22", "L4 per_residue"),
        ("repa_l4_256_per_sample_step400k", "L4 per_sample"),
        ("repa_l9_256_ep25", "L9 per_residue"),
        ("repa_l9_256_per_sample_steplast", "L9 per_sample"),
    ],
    "lambda": [
        ("repa_l4_256_ep13_step300k", "λ=0.5 @ 300k"),
        ("repa_l4_256_per_residue_lambda1_step300k", "λ=1.0 @ 300k"),
        ("repa_l4_256_per_residue_lambda2_step200k", "λ=2.0 @ 200k"),
    ],
    "step_extension": [
        ("repa_l4_256_ep13_step300k", "REPA L4 @ 300k"),
        ("repa_l4_256_ep22", "REPA L4 @ 400k"),
        ("repa_l4_256_ep31_step500k", "REPA L4 @ 500k"),
    ],
    "pretrained_vs_ours": [
        ("baseline_256_ep21", "ours (10L) baseline"),
        ("repa_l4_256_ep22", "ours (10L) REPA L4"),
        ("pretrained_dfs_60m", "NGC pretrained (12L)"),
    ],
}


# A 7-color palette that holds up at 7+ curves; black for non-REPA references.
_PALETTE = [
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#bcbd22",
]


def _results_dir(sweep: str) -> Path:
    # Both sweeps now follow the post-refactor convention:
    # ``results/paper/<sweep_short>_paper_cath/cath/``. Probe-from-cache jobs
    # write to ``output_dir/cath/`` (probe_kind subdir), so the JSONL/CSV live
    # one level deeper than the YAML ``output_dir``.
    if sweep == "paper_n128_cath":
        return RESULTS_ROOT / "paper" / "n128_paper_cath" / "cath"
    if sweep == "paper_n256_cath":
        return RESULTS_ROOT / "paper" / "n256_paper_cath" / "cath"
    raise ValueError(f"unknown sweep {sweep!r}")


def _fig_dir(sweep: str) -> Path:
    # Figures always go to figures/paper/<sweep_short>_paper_cath/cath/, regardless
    # of where the results JSONL currently lives. The trailing /cath/ matches the
    # lowest-level probe-type split (contact/ vs cath/) used throughout figures/.
    short = sweep.replace("paper_", "").replace("_cath", "")  # n128 / n256
    return FIG_ROOT / f"{short}_paper_cath" / "cath"


def load_results(sweep: str) -> pd.DataFrame:
    jsonl = _results_dir(sweep) / "pretrained_sweep_results.jsonl"
    if not jsonl.exists():
        raise SystemExit(f"results not found: {jsonl}")
    rows = []
    with open(jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r.get("probe_kind") == "cath" and "cath_accuracy" in r:
                rows.append(r)
    if not rows:
        raise SystemExit(f"no valid CATH rows in {jsonl}")
    df = pd.DataFrame(rows)
    return df


def peak_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(run, level) peak accuracy + layer where it occurs."""
    idx = df.groupby(["run", "cath_level"])["cath_accuracy"].idxmax()
    peak = df.loc[idx, ["run", "cath_level", "layer", "cath_accuracy", "cath_macro_f1"]]
    return peak.pivot(
        index="run",
        columns="cath_level",
        values=["layer", "cath_accuracy", "cath_macro_f1"],
    )


def plot_layer_curves(df: pd.DataFrame, outpath: Path, title: str) -> None:
    """3-panel figure: cath-C / A / T accuracy vs layer, all runs overlaid."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    runs = sorted(df["run"].unique())
    colors = {r: _PALETTE[i % len(_PALETTE)] for i, r in enumerate(runs)}
    for ax, level in zip(axes, ["C", "A", "T"]):
        sub = df[df.cath_level == level]
        for run in runs:
            r = sub[sub["run"] == run].sort_values("layer")
            if r.empty:
                continue
            ax.plot(
                r["layer"],
                r["cath_accuracy"],
                marker="o",
                color=colors[run],
                label=run,
                linewidth=1.5,
                markersize=4,
            )
        ax.set_title(f"CATH-{level} accuracy vs layer", fontsize=11)
        ax.set_xlabel("transformer layer")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    axes[-1].legend(fontsize=6, loc="lower right", ncol=2)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


def plot_block(
    df: pd.DataFrame,
    block_name: str,
    run_labels: List[tuple],
    outpath: Path,
    sweep_title: str,
) -> None:
    """3-panel figure for one ablation block: each run gets a fixed color+label."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    colors = {run: _PALETTE[i % len(_PALETTE)] for i, (run, _) in enumerate(run_labels)}
    for ax, level in zip(axes, ["C", "A", "T"]):
        sub = df[df.cath_level == level]
        for run, label in run_labels:
            r = sub[sub["run"] == run].sort_values("layer")
            if r.empty:
                continue
            ax.plot(
                r["layer"],
                r["cath_accuracy"],
                marker="o",
                color=colors[run],
                label=label,
                linewidth=1.5,
                markersize=5,
            )
        ax.set_title(f"CATH-{level}", fontsize=11)
        ax.set_xlabel("transformer layer")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{sweep_title} — {block_name} ablation", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sweep", required=True, choices=["paper_n128_cath", "paper_n256_cath"]
    )
    args = ap.parse_args()

    df = load_results(args.sweep)
    blocks = BLOCKS_N128 if args.sweep == "paper_n128_cath" else BLOCKS_N256
    fig_dir = _fig_dir(args.sweep)
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(df)
    n_ckpts = df[["run", "step"]].drop_duplicates().shape[0]
    print(f"[{args.sweep}] {n_rows} rows from {n_ckpts} ckpts")

    # Peak table.
    peak = peak_table(df)
    peak_csv = fig_dir / "table_peak.csv"
    peak.to_csv(peak_csv)
    print(f"  wrote {peak_csv.name}")

    # Headline layer curves over all runs.
    plot_layer_curves(
        df,
        fig_dir / "fig_layer_curves.png",
        title=f"{args.sweep}: per-layer CATH accuracy",
    )

    # Per-block figures (skip blocks where >half the runs are missing).
    for block_name, run_labels in blocks.items():
        present = sum(1 for run, _ in run_labels if (df["run"] == run).any())
        if present < max(2, len(run_labels) // 2):
            print(
                f"  skip block {block_name} ({present}/{len(run_labels)} runs present)"
            )
            continue
        plot_block(
            df,
            block_name,
            run_labels,
            fig_dir / f"fig_block_{block_name}.png",
            sweep_title=args.sweep,
        )


if __name__ == "__main__":
    main()
