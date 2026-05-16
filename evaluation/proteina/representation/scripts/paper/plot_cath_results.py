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

from evaluation.proteina.lib.plot_labels import pretty_run_label


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


# Baseline rendering. Matches lib/sources.py sentinel codes and the
# untrained_proteina sentinel range (layer = -(trunk_layer + 10), so
# trunk_layer = -layer - 10 inverts it).
_LAYER_RANDOM_GAUSS = -2
_LAYER_SEQ_ONEHOT = -3
_UNTRAINED_LAYER_MIN = -19
_UNTRAINED_LAYER_MAX = -10
_TRAINED_NOISE_LAYER_MIN = -29
_TRAINED_NOISE_LAYER_MAX = -20

_BASELINE_COLORS = {
    "random_gauss": "#7f7f7f",
    "seq_onehot": "#8c564b",
    "untrained_proteina": "#e377c2",
    "trained_noise": "#17becf",
}


def _split_baselines(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (checkpoint_rows, baseline_rows). Baselines have negative layer."""
    is_baseline = df["layer"] < 0
    return df[~is_baseline].copy(), df[is_baseline].copy()


def _draw_baselines(ax, baselines: pd.DataFrame, level: str) -> None:
    """Overlay random_gauss + seq_onehot as horizontal lines, untrained_proteina
    as a curve at trunk-layer index. CATH level filter applied here."""
    sub = baselines[baselines.cath_level == level]
    if sub.empty:
        return

    for name, sentinel in [
        ("random_gauss", _LAYER_RANDOM_GAUSS),
        ("seq_onehot", _LAYER_SEQ_ONEHOT),
    ]:
        row = sub[sub["layer"] == sentinel]
        if row.empty:
            continue
        v = float(row["cath_accuracy"].iloc[0])
        ax.axhline(
            v,
            linestyle="--",
            linewidth=0.9,
            color=_BASELINE_COLORS[name],
            label=name,
            alpha=0.75,
        )

    for run_name, lmin, lmax, offset, marker in [
        ("untrained_proteina", _UNTRAINED_LAYER_MIN, _UNTRAINED_LAYER_MAX, 10, "s"),
        ("trained_noise", _TRAINED_NOISE_LAYER_MIN, _TRAINED_NOISE_LAYER_MAX, 20, "^"),
    ]:
        curve = sub[
            (sub["run"] == run_name) & (sub["layer"] >= lmin) & (sub["layer"] <= lmax)
        ].copy()
        if curve.empty:
            continue
        curve["trunk_layer"] = -curve["layer"] - offset
        curve = curve.sort_values("trunk_layer")
        ax.plot(
            curve["trunk_layer"],
            curve["cath_accuracy"],
            "--" + marker,
            color=_BASELINE_COLORS[run_name],
            label=run_name,
            linewidth=1.2,
            markersize=3.5,
            alpha=0.85,
        )


def _results_dir(sweep: str) -> Path:
    # Post-flatten (2026-05-16): no more probe_kind subdir for single-probe
    # sweeps. JSONL/CSV live directly under ``results/paper/<sweep_short>_paper_cath/``.
    if sweep == "paper_n128_cath":
        return RESULTS_ROOT / "paper" / "n128_paper_cath"
    if sweep == "paper_n256_cath":
        return RESULTS_ROOT / "paper" / "n256_paper_cath"
    raise ValueError(f"unknown sweep {sweep!r}")


def _fig_dir(sweep: str) -> Path:
    # Post-reorg (2026-05-16): probe-kind dirs live under set-level parents.
    # figures/paper/<set>/cath/ holds the 2-row PDB+AFDB plots (from
    # plot_cath_pdb_vs_afdb.py) and the per-run CSV tables this script emits.
    # PNG generation in this script is disabled (single-source plots are the
    # top row of the 2-row plots).
    short = sweep.replace("paper_", "").replace("_cath", "")  # n128 / n256
    return FIG_ROOT / short / "cath"


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


def _run_step(df: pd.DataFrame, run: str) -> int | None:
    """Pull the step recorded for ``run`` in this df, or None if missing.

    Eval rows for one run share one ckpt, hence one step. The data row's
    ``step`` is the authoritative source — the run id suffix (``_steplast``)
    can mean different actual steps across suites.
    """
    if "step" not in df.columns:
        return None
    steps = df[df["run"] == run]["step"].dropna().unique()
    if len(steps) == 0:
        return None
    return int(steps[0])


def peak_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(run, level) peak accuracy + layer where it occurs.

    Restricted to real transformer layers (layer >= 0). Baselines have
    negative sentinel layers and live in a separate companion CSV.
    """
    df = df[df["layer"] >= 0]
    idx = df.groupby(["run", "cath_level"])["cath_accuracy"].idxmax()
    peak = df.loc[idx, ["run", "cath_level", "layer", "cath_accuracy", "cath_macro_f1"]]
    return peak.pivot(
        index="run",
        columns="cath_level",
        values=["layer", "cath_accuracy", "cath_macro_f1"],
    )


def plot_layer_curves(df: pd.DataFrame, outpath: Path, title: str) -> None:
    """3-panel figure: cath-C / A / T accuracy vs layer, all runs overlaid."""
    ckpt_df, baseline_df = _split_baselines(df)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    runs = sorted(ckpt_df["run"].unique())
    colors = {r: _PALETTE[i % len(_PALETTE)] for i, r in enumerate(runs)}
    for ax, level in zip(axes, ["C", "A", "T"]):
        sub = ckpt_df[ckpt_df.cath_level == level]
        for run in runs:
            r = sub[sub["run"] == run].sort_values("layer")
            if r.empty:
                continue
            ax.plot(
                r["layer"],
                r["cath_accuracy"],
                marker="o",
                color=colors[run],
                label=pretty_run_label(
                    run, step=_run_step(sub, run), allow_missing_step=True
                ),
                linewidth=1.5,
                markersize=4,
            )
        _draw_baselines(ax, baseline_df, level)
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
    _, baseline_df = _split_baselines(df)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    colors = {run: _PALETTE[i % len(_PALETTE)] for i, (run, _) in enumerate(run_labels)}
    for ax, level in zip(axes, ["C", "A", "T"]):
        sub = df[(df.cath_level == level) & (df["layer"] >= 0)]
        for run, label in run_labels:
            r = sub[sub["run"] == run].sort_values("layer")
            if r.empty:
                continue
            ax.plot(
                r["layer"],
                r["cath_accuracy"],
                marker="o",
                color=colors[run],
                label=pretty_run_label(
                    run,
                    step=_run_step(sub, run),
                    display=label,
                    allow_missing_step=True,
                ),
                linewidth=1.5,
                markersize=5,
            )
        _draw_baselines(ax, baseline_df, level)
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

    # Baseline summary (random_gauss / seq_onehot scalars + untrained per layer).
    _, baseline_df = _split_baselines(df)
    if not baseline_df.empty:
        base_csv = fig_dir / "table_baselines.csv"
        baseline_df[
            [
                "run",
                "layer",
                "cath_level",
                "cath_accuracy",
                "cath_macro_f1",
                "cath_n_classes",
            ]
        ].sort_values(["run", "cath_level", "layer"]).to_csv(base_csv, index=False)
        print(f"  wrote {base_csv.name}")

    # PNG generation disabled 2026-05-16: single-source cath plots removed
    # because their data is the top row of plot_cath_pdb_vs_afdb.py. This
    # script now only emits the CSV tables above. Re-enable by uncommenting
    # the plot_layer_curves and plot_block calls (and set fig_dir back to
    # a dedicated single-source location to avoid clashing with the
    # comparison plots).


if __name__ == "__main__":
    main()
