"""Side-by-side PDB vs AFDB CATH-probe plots for n=256 paper sweeps.

Loads the consolidated JSONL from both ``paper_n256_cath`` (PDB labels) and
``paper_n256_cath_afdb`` (Gene3D-derived AFDB labels) and emits 2×3 grid
figures per ablation block:

    row 0 (top)    PDB probe    CATH-C | CATH-A | CATH-T
    row 1 (bottom) AFDB probe   CATH-C | CATH-A | CATH-T

Same color per run across rows so any drop / lift between probe datasets
is visible by eye. Outputs land under
``figures/paper/n256_paper_cath_pdb_vs_afdb/cath/``.
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

# Reuse the same block defs + palette + baseline-drawing helpers as the
# single-sweep plotter — keeps colors and run ordering consistent.
sys.path.insert(0, str(HERE))
from plot_cath_results import (  # noqa: E402
    BLOCKS_N128,
    BLOCKS_N256,
    _PALETTE,
    _draw_baselines,
    _split_baselines,
    _run_step,
)
from evaluation.proteina.lib.plot_labels import (  # noqa: E402
    block_label_plan,
    compose_legend_label,
    compose_title_suffix,
)


def _paths_for_size(size: int) -> Tuple[Path, Path, Path, Dict]:
    if size == 128:
        return (
            RESULTS_ROOT
            / "paper"
            / "n128_paper_cath"
            / "pretrained_sweep_results.jsonl",
            RESULTS_ROOT
            / "paper"
            / "n128_paper_cath_afdb"
            / "pretrained_sweep_results.jsonl",
            FIG_ROOT / "n128" / "cath",
            BLOCKS_N128,
        )
    if size == 256:
        return (
            RESULTS_ROOT
            / "paper"
            / "n256_paper_cath"
            / "pretrained_sweep_results.jsonl",
            RESULTS_ROOT
            / "paper"
            / "n256_paper_cath_afdb"
            / "pretrained_sweep_results.jsonl",
            FIG_ROOT / "n256" / "cath",
            BLOCKS_N256,
        )
    raise ValueError(f"unsupported --size {size}; choose 128 or 256")


def _load_sweep_shards(results_dir: Path) -> pd.DataFrame:
    """Load all per-run shards + the legacy single jsonl (whichever exists).

    The array launcher writes per-run shards (pretrained_sweep_results.<run>.jsonl);
    older single-task runs write the unsharded pretrained_sweep_results.jsonl.
    We union both — duplicates dropped by (run, step, layer, cath_level, t).
    """
    rows: List[dict] = []
    for jsonl in sorted(results_dir.glob("pretrained_sweep_results*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                r = json.loads(line)
                if r.get("probe_kind") == "cath" and "cath_accuracy" in r:
                    rows.append(r)
    if not rows:
        raise SystemExit(f"no CATH rows found in {results_dir}")
    df = pd.DataFrame(rows)
    # Drop duplicates if a row exists in both the shard and the legacy file.
    key = ["run", "step", "layer", "cath_level", "t"]
    df = df.drop_duplicates(subset=[c for c in key if c in df.columns], keep="last")
    return df


def _shared_palette(run_labels: List[Tuple[str, str]]) -> Dict[str, str]:
    return {run: _PALETTE[i % len(_PALETTE)] for i, (run, _) in enumerate(run_labels)}


def _plot_row(
    axes,  # iterable of 3 axes for C / A / T
    df: pd.DataFrame,
    run_labels: List[Tuple[str, str]],
    colors: Dict[str, str],
    row_title: str,
    show_legend: bool,
) -> None:
    ckpt_df, baseline_df = _split_baselines(df)
    runs_in_block = [run for run, _ in run_labels]
    _, varying = block_label_plan(runs_in_block)
    for ax, level in zip(axes, ["C", "A", "T"]):
        sub = ckpt_df[ckpt_df.cath_level == level]
        for run, variant in run_labels:
            r = sub[sub["run"] == run].sort_values("layer")
            if r.empty:
                continue
            ax.plot(
                r["layer"],
                r["cath_accuracy"],
                marker="o",
                color=colors[run],
                label=compose_legend_label(
                    run,
                    step=_run_step(sub, run),
                    variant_tag=variant,
                    varying_fields=varying,
                ),
                linewidth=1.5,
                markersize=5,
            )
        _draw_baselines(ax, baseline_df, level)
        ax.set_title(f"{row_title} — CATH-{level}", fontsize=10)
        ax.set_xlabel("transformer layer")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        if show_legend:
            ax.legend(fontsize=6, loc="lower right")


def plot_block_pdb_vs_afdb(
    df_pdb: pd.DataFrame,
    df_afdb: pd.DataFrame,
    block_name: str,
    run_labels: List[Tuple[str, str]],
    outpath: Path,
    sweep_label: str = "paper_n256_cath",
) -> None:
    colors = _shared_palette(run_labels)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey="col")
    _plot_row(
        axes[0], df_pdb, run_labels, colors, row_title="PDB probe", show_legend=False
    )
    _plot_row(
        axes[1], df_afdb, run_labels, colors, row_title="AFDB probe", show_legend=True
    )
    title_suffix = compose_title_suffix([run for run, _ in run_labels])
    fig.suptitle(
        f"{sweep_label} — {block_name} ablation{title_suffix} (PDB vs AFDB probe labels)",
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
    sweep_label: str = "paper_n256_cath",
) -> None:
    """All-runs overlay; top row PDB probe, bottom row AFDB probe."""
    runs = sorted(set(df_pdb["run"]) | set(df_afdb["run"]))
    # variant_tag=None — legend label comes from RunMeta via compose_legend_label.
    run_labels = [(r, None) for r in runs]
    colors = {r: _PALETTE[i % len(_PALETTE)] for i, r in enumerate(runs)}
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey="col")
    _plot_row(
        axes[0], df_pdb, run_labels, colors, row_title="PDB probe", show_legend=False
    )
    _plot_row(
        axes[1], df_afdb, run_labels, colors, row_title="AFDB probe", show_legend=True
    )
    axes[1, -1].legend(fontsize=5, loc="lower right", ncol=2)
    fig.suptitle(
        f"{sweep_label}: per-layer CATH accuracy (all runs, PDB vs AFDB probe labels)",
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

    n_runs_pdb = df_pdb["run"].nunique()
    n_runs_afdb = df_afdb["run"].nunique()
    print(f"[PDB n={args.size}]  {len(df_pdb)} rows, {n_runs_pdb} runs")
    print(f"[AFDB n={args.size}] {len(df_afdb)} rows, {n_runs_afdb} runs")

    fig_dir.mkdir(parents=True, exist_ok=True)
    sweep_label = f"paper_n{args.size}_cath"

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
