"""Side-by-side PDB vs AFDB inverse-folding probe plots for n=256 paper sweeps.

Loads consolidated JSONL from both ``paper_n256_struct`` (PDB labels) and
``paper_n256_struct_afdb`` (AFDB-Swissprot real sequence labels) and emits
2×2 grid figures per ablation block:

    row 0 (top)    PDB probe    top-1 accuracy | macro-F1
    row 1 (bottom) AFDB probe   top-1 accuracy | macro-F1

Same color per run across rows so any drop / lift between probe datasets is
visible by eye. Outputs land under
``figures/paper/n256_paper_struct_pdb_vs_afdb/inverse_folding/``.

Why inverse folding only: AFDB-Swissprot ``residue_type`` is the real
UniProt sequence (measured), while AFDB coords/dihedrals/distances are AF2
predictions (synthetic). IF is the only AFDB structural probe with
unambiguously real targets, so it's the only cross-dataset comparison that
isolates rep-quality from input-distribution noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
REPR_ROOT = HERE.parents[1]
RESULTS_ROOT = REPR_ROOT / "results"
FIG_ROOT = REPR_ROOT / "figures" / "paper"

sys.path.insert(0, str(HERE))
from plot_struct_probes import (  # noqa: E402
    BLOCKS_N128,
    BLOCKS_N256,
    _PALETTE,
    _draw_baseline_curves,
    _draw_trivial_baselines,
    _run_step,
    _split_baselines,
    pretty_run_label,
)


def _paths_for_size(size: int) -> Tuple[Path, Path, Path, Dict]:
    if size == 128:
        return (
            RESULTS_ROOT
            / "paper"
            / "n128_paper_struct"
            / "pretrained_sweep_results.jsonl",
            RESULTS_ROOT
            / "paper"
            / "n128_paper_struct_afdb"
            / "pretrained_sweep_results.jsonl",
            FIG_ROOT / "n128" / "if",
            BLOCKS_N128,
        )
    if size == 256:
        return (
            RESULTS_ROOT
            / "paper"
            / "n256_paper_struct"
            / "pretrained_sweep_results.jsonl",
            RESULTS_ROOT
            / "paper"
            / "n256_paper_struct_afdb"
            / "pretrained_sweep_results.jsonl",
            FIG_ROOT / "n256" / "if",
            BLOCKS_N256,
        )
    raise ValueError(f"unsupported --size {size}; choose 128 or 256")


# IF gets two metric panels per probe-dataset row: top-1 accuracy and macro-F1.
# y-limits tightened to the actual operating range — IF top-1 on n=256 frozen
# flow-matching reps at t=1.0 saturates around 0.15-0.25 (20-way chance ~5%,
# majority-class floor ~9-10%), so a 0-1 axis flattens all curves into a strip.
METRICS: List[Tuple[str, str, Tuple[float, float]]] = [
    ("if_top1_acc", "top-1 accuracy", (0.0, 0.30)),
    ("if_macro_f1", "macro-F1", (0.0, 0.20)),
]


def _load_sweep_shards(results_dir: Path) -> pd.DataFrame:
    """Union all per-run + baseline shards + legacy single jsonl; IF rows only."""
    rows: List[dict] = []
    for jsonl in sorted(results_dir.glob("pretrained_sweep_results*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                r = json.loads(line)
                if r.get("probe_kind") == "inverse_folding" and "if_top1_acc" in r:
                    rows.append(r)
    if not rows:
        raise SystemExit(f"no inverse_folding rows found in {results_dir}")
    df = pd.DataFrame(rows)
    key = [c for c in ["run", "step", "layer", "t"] if c in df.columns]
    df = df.drop_duplicates(subset=key, keep="last")
    return df


def _plot_row(
    axes,  # iterable of 2 axes (top-1 | macro-F1)
    df: pd.DataFrame,
    run_labels: List[Tuple[str, str]],
    colors: Dict[str, str],
    row_title: str,
    show_legend: bool,
) -> None:
    ckpt_df, baseline_df = _split_baselines(df)
    for ax, (metric, ylabel, ylim) in zip(axes, METRICS):
        for run, label in run_labels:
            r = ckpt_df[ckpt_df["run"] == run].sort_values("layer")
            if r.empty or metric not in r.columns:
                continue
            ax.plot(
                r["layer"],
                r[metric],
                marker="o",
                color=colors[run],
                label=pretty_run_label(
                    run,
                    step=_run_step(ckpt_df, run),
                    display=label,
                    allow_missing_step=True,
                ),
                linewidth=1.5,
                markersize=5,
            )
        # Trivial baselines (knn_dist / random_gauss / seq_onehot) — horizontal
        # lines from negative-layer rows. Baseline-curve plotting for
        # untrained_proteina + trained_noise piggybacks on the metric name.
        _draw_trivial_baselines(ax, baseline_df, probe="inverse_folding")
        _draw_baseline_curves(ax, baseline_df, probe="inverse_folding")
        ax.set_title(f"{row_title} — {ylabel}", fontsize=10)
        ax.set_xlabel("transformer layer")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_ylim(*ylim)
        if show_legend:
            ax.legend(fontsize=6, loc="lower right")


def plot_block_pdb_vs_afdb(
    df_pdb: pd.DataFrame,
    df_afdb: pd.DataFrame,
    block_name: str,
    run_labels: List[Tuple[str, str]],
    outpath: Path,
    sweep_label: str = "paper_n256_struct",
) -> None:
    colors = {run: _PALETTE[i % len(_PALETTE)] for i, (run, _) in enumerate(run_labels)}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey="col")
    _plot_row(axes[0], df_pdb, run_labels, colors, "PDB probe", show_legend=False)
    _plot_row(axes[1], df_afdb, run_labels, colors, "AFDB probe", show_legend=True)
    fig.suptitle(
        f"{sweep_label} — {block_name} ablation (inverse folding, PDB vs AFDB probe labels)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


def plot_layer_curves_pdb_vs_afdb(
    df_pdb: pd.DataFrame,
    df_afdb: pd.DataFrame,
    outpath: Path,
    sweep_label: str = "paper_n256_struct",
) -> None:
    runs = sorted(set(df_pdb["run"]) | set(df_afdb["run"]))
    run_labels = [(r, r) for r in runs]
    colors = {r: _PALETTE[i % len(_PALETTE)] for i, r in enumerate(runs)}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey="col")
    _plot_row(axes[0], df_pdb, run_labels, colors, "PDB probe", show_legend=False)
    _plot_row(axes[1], df_afdb, run_labels, colors, "AFDB probe", show_legend=True)
    axes[1, -1].legend(fontsize=5, loc="lower right", ncol=2)
    fig.suptitle(
        f"{sweep_label}: per-layer inverse-folding accuracy (all runs, PDB vs AFDB probe labels)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256, choices=[128, 256])
    args = ap.parse_args()

    pdb_jsonl, afdb_jsonl, fig_dir, blocks = _paths_for_size(args.size)
    df_pdb = _load_sweep_shards(pdb_jsonl.parent)
    df_afdb = _load_sweep_shards(afdb_jsonl.parent)
    print(f"[PDB n={args.size}]  {len(df_pdb)} rows, {df_pdb['run'].nunique()} runs")
    print(f"[AFDB n={args.size}] {len(df_afdb)} rows, {df_afdb['run'].nunique()} runs")

    fig_dir.mkdir(parents=True, exist_ok=True)
    sweep_label = f"paper_n{args.size}_struct"
    plot_layer_curves_pdb_vs_afdb(
        df_pdb, df_afdb, fig_dir / "fig_layer_curves.png", sweep_label=sweep_label
    )

    for block_name, run_labels in blocks.items():
        present_pdb = sum(1 for run, _ in run_labels if (df_pdb["run"] == run).any())
        present_afdb = sum(1 for run, _ in run_labels if (df_afdb["run"] == run).any())
        thresh = max(2, len(run_labels) // 2)
        if present_pdb < thresh or present_afdb < thresh:
            print(
                f"  skip block {block_name} (PDB={present_pdb}/{len(run_labels)}, AFDB={present_afdb}/{len(run_labels)})"
            )
            continue
        plot_block_pdb_vs_afdb(
            df_pdb,
            df_afdb,
            block_name,
            run_labels,
            fig_dir / f"fig_block_{block_name}.png",
            sweep_label=sweep_label,
        )


if __name__ == "__main__":
    main()
