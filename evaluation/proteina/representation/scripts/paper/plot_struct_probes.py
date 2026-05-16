"""Plots for the residue/pair-level structural probes (paper_n{128,256}_struct).

Reads ``pretrained_sweep_results.jsonl`` shards from the struct sweeps and emits
per-probe layer-curve figures + per-ablation-block sub-figures for each of the
three new probes:

  inverse_folding  per-residue 20-way AA classification (top-1, macro-F1)
  dihedral         per-residue (φ, ψ) regression (mean angular error in deg)
  distance         pair Cα-Cα distance regression (Huber-MAE in Å, bucketed)

Headline metrics by probe (chosen for the layer-curves panels):

  inverse_folding → if_top1_acc     ↑ better, baselines: knn_dist, random_gauss
  dihedral        → dih_mae_total_deg ↓ better, baselines: local_frame, random_gauss
  distance        → dist_mae_total    ↓ better, baselines: seqsep_pair, random_gauss

Trivial-geometric baselines render as dashed horizontal lines because each is a
single sentinel-layer scalar, not a per-trunk-layer curve. ``untrained_proteina``
and ``trained_noise`` render as dashed curves because they emit one row per
sentinel layer (10 each, mapped back to trunk-layer index).

Usage:
  python evaluation/proteina/representation/scripts/paper/plot_struct_probes.py \\
      --sweep paper_n128_struct
  python evaluation/proteina/representation/scripts/paper/plot_struct_probes.py \\
      --sweep paper_n256_struct

Outputs land in ``figures/paper/n{128,256}_paper_struct/``:
  fig_layer_curves_<probe>.png   — all runs vs layer for one probe kind
  fig_block_<probe>_<name>.png   — one figure per (probe_kind, ablation block)
  table_peak_<probe>.csv         — peak metric per run (best layer)
  table_baselines.csv            — baseline scalars + sentinel-layer breakdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from evaluation.proteina.lib.plot_labels import pretty_run_label


HERE = Path(__file__).resolve().parent
REPR_ROOT = HERE.parents[1]
RESULTS_ROOT = REPR_ROOT / "results"
FIG_ROOT = REPR_ROOT / "figures" / "paper"


# ─── Probe-axis spec ───────────────────────────────────────────────────────
# For each probe kind: the headline metric column, axis label, whether
# higher is better, optional y-limits, and which sentinel-layer baselines
# to overlay (with a label).

_LAYER_RANDOM_GAUSS = -2
_LAYER_SEQ_ONEHOT = -3
_LAYER_KNN_DIST = -200
_LAYER_LOCAL_FRAME = -201
_LAYER_SEQSEP_PAIR = -202
_UNTRAINED_LAYER_MIN, _UNTRAINED_LAYER_MAX, _UNTRAINED_OFFSET = -19, -10, 10
_TRAINED_NOISE_LAYER_MIN, _TRAINED_NOISE_LAYER_MAX, _TRAINED_NOISE_OFFSET = -29, -20, 20


PROBE_SPEC: Dict[str, Dict] = {
    "inverse_folding": {
        "metric": "if_top1_acc",
        "ylabel": "top-1 accuracy",
        "higher_is_better": True,
        "ylim": (0.0, 1.0),
        # (run, sentinel_layer, label, color)
        "trivial_baselines": [
            ("knn_dist", _LAYER_KNN_DIST, "knn_dist (8NN local geom)", "#9467bd"),
            ("random_gauss", _LAYER_RANDOM_GAUSS, "random_gauss", "#7f7f7f"),
            ("seq_onehot", _LAYER_SEQ_ONEHOT, "seq_onehot (oracle, =1)", "#8c564b"),
        ],
        "title_short": "Inverse folding",
    },
    "dihedral": {
        "metric": "dih_mae_total_deg",
        "ylabel": "mean angular error (°)",
        "higher_is_better": False,
        "ylim": (0.0, 100.0),
        "trivial_baselines": [
            ("local_frame", _LAYER_LOCAL_FRAME, "local_frame (analytic)", "#9467bd"),
            ("random_gauss", _LAYER_RANDOM_GAUSS, "random_gauss", "#7f7f7f"),
        ],
        "title_short": "Backbone dihedral (φ, ψ)",
    },
    "distance": {
        "metric": "dist_mae_total",
        "ylabel": "MAE (Å)",
        "higher_is_better": False,
        "ylim": (0.0, 12.0),
        "trivial_baselines": [
            ("seqsep_pair", _LAYER_SEQSEP_PAIR, "seqsep_pair (chain prior)", "#9467bd"),
            ("random_gauss", _LAYER_RANDOM_GAUSS, "random_gauss", "#7f7f7f"),
        ],
        "title_short": "Pair Cα–Cα distance",
    },
}


# ─── Ablation blocks (mirrors plot_cath_results.py + extends with new runs) ─

BLOCKS_N128: Dict[str, List[Tuple[str, str]]] = {
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
        ("repa_l4_128_bs80_lambda025_step200k", "REPA L4 λ=0.25"),
        ("repa_l4_128_bs80_lambda1_step200k", "REPA L4 λ=1.0"),
        ("repa_l4_128_bs80_lambda2_steplast", "REPA L4 λ=2.0"),
        ("repa_l4_128_bs80_wd1e2_step200k", "REPA L4 wd=1e-2"),
    ],
    "afdb": [
        ("baseline_128_bs80_step200k", "PDB baseline 200k"),
        ("baseline_afdb_128_bs80_step200k", "AFDB baseline 200k"),
        ("baseline_afdb_128_bs80_step400k", "AFDB baseline 400k"),
        ("repa_l4_128_bs80_step200k", "PDB REPA L4 200k"),
        ("repa_l4_afdb_128_bs80_step200k", "AFDB REPA L4 200k"),
        ("repa_mpnn_l4_128_bs80_step200k", "PDB REPA MPNN L4 200k"),
        ("repa_mpnn_l4_afdb_128_bs80_step200k", "AFDB REPA MPNN L4 200k"),
        ("repa_mpnn_l4_afdb_128_bs80_step400k", "AFDB REPA MPNN L4 400k"),
    ],
    "pretrained_vs_ours": [
        ("baseline_128_bs80_step200k", "ours (10L) baseline"),
        ("repa_l4_128_bs80_step200k", "ours (10L) REPA L4"),
        ("pretrained_dfs_60m", "NGC pretrained (12L)"),
    ],
}

BLOCKS_N256: Dict[str, List[Tuple[str, str]]] = {
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
        ("baseline_256_ep21", "PDB baseline ep21"),
        ("baseline_afdb_256_ep20", "AFDB baseline ep20"),
        ("baseline_afdb_256_step400k", "AFDB baseline 400k"),
        ("baseline_afdb_256_step700k", "AFDB baseline 700k"),
        ("baseline_afdb_256_step900k", "AFDB baseline 900k"),
        ("repa_l4_256_ep22", "PDB REPA L4 ep22"),
        ("repa_l4_afdb_256_ep20", "AFDB REPA L4 ep20"),
        ("repa_l4_afdb_256_step400k", "AFDB REPA L4 400k"),
        ("repa_l4_afdb_256_step700k", "AFDB REPA L4 700k"),
        ("repa_mpnn_l4_afdb_256_step400k", "AFDB REPA MPNN L4 400k"),
        ("repa_mpnn_l9_afdb_256_step400k", "AFDB REPA MPNN L9 400k"),
        ("repa_mpnn_l9_afdb_256_step700k", "AFDB REPA MPNN L9 700k"),
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
        ("repa_l4_256_per_residue_lambda1_step200k", "λ=1.0 @ 200k"),
        ("repa_l4_256_per_residue_lambda1_step300k", "λ=1.0 @ 300k"),
        ("repa_l4_256_per_residue_lambda2_step200k", "λ=2.0 @ 200k"),
        ("repa_l4_256_per_residue_lambda2_step300k", "λ=2.0 @ 300k"),
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


_PALETTE = [
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#bcbd22",
    "#e377c2",
    "#7f7f7f",
    "#aec7e8",
    "#ffbb78",
]

_BASELINE_CURVE_COLORS = {
    "untrained_proteina": "#e377c2",
    "trained_noise": "#17becf",
}


# ─── IO + helpers ──────────────────────────────────────────────────────────


def _split_baselines(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Negative `layer` values are sentinel codes for baselines; non-negative are real trunk layers."""
    is_baseline = df["layer"] < 0
    return df[~is_baseline].copy(), df[is_baseline].copy()


def _results_dir(sweep: str) -> Path:
    if sweep == "paper_n128_struct":
        return RESULTS_ROOT / "paper" / "n128_paper_struct"
    if sweep == "paper_n256_struct":
        return RESULTS_ROOT / "paper" / "n256_paper_struct"
    raise ValueError(f"unknown sweep {sweep!r}")


def _fig_dir(sweep: str) -> Path:
    short = sweep.replace("paper_", "").replace("_struct", "")  # n128 / n256
    return FIG_ROOT / f"{short}_paper_struct"


def load_results(sweep: str) -> pd.DataFrame:
    """Concatenate every shard JSONL in the sweep dir; one row per probe-fit.

    Same shard-glob behaviour as ``consolidate()`` in pretrain_probe_sweep.py.
    Filtered to the three struct probe kinds; cath/contact rows are silently
    dropped because this script doesn't render them.
    """
    rdir = _results_dir(sweep)
    jsonl_paths = sorted(rdir.glob("pretrained_sweep_results*.jsonl"))
    if not jsonl_paths:
        raise SystemExit(f"no result shards found in {rdir}")
    rows = []
    for p in jsonl_paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in r:
                    continue
                if r.get("probe_kind") in PROBE_SPEC:
                    rows.append(r)
    if not rows:
        raise SystemExit(f"no struct-probe rows in {rdir} shards")
    df = pd.DataFrame(rows)
    # Dedup by row identity tuple; keep last (matches consolidate() semantics).
    df = df.drop_duplicates(
        subset=["run", "step", "layer", "t", "probe_kind"], keep="last"
    )
    return df


# ─── Baseline overlays ─────────────────────────────────────────────────────


def _draw_trivial_baselines(ax, baselines: pd.DataFrame, probe: str) -> None:
    """Horizontal dashed lines at each trivial-geometric baseline's metric value."""
    spec = PROBE_SPEC[probe]
    metric = spec["metric"]
    sub = baselines[baselines.probe_kind == probe]
    if sub.empty:
        return
    for run, sentinel, label, color in spec["trivial_baselines"]:
        row = sub[(sub["run"] == run) & (sub["layer"] == sentinel)]
        if row.empty:
            continue
        v = float(row[metric].iloc[0])
        ax.axhline(
            v,
            linestyle="--",
            linewidth=1.0,
            color=color,
            label=label,
            alpha=0.85,
        )


def _draw_baseline_curves(ax, baselines: pd.DataFrame, probe: str) -> None:
    """``untrained_proteina`` + ``trained_noise`` as dashed curves over trunk layers."""
    spec = PROBE_SPEC[probe]
    metric = spec["metric"]
    sub = baselines[baselines.probe_kind == probe]
    if sub.empty:
        return
    for run_name, lmin, lmax, offset, marker in [
        (
            "untrained_proteina",
            _UNTRAINED_LAYER_MIN,
            _UNTRAINED_LAYER_MAX,
            _UNTRAINED_OFFSET,
            "s",
        ),
        (
            "trained_noise",
            _TRAINED_NOISE_LAYER_MIN,
            _TRAINED_NOISE_LAYER_MAX,
            _TRAINED_NOISE_OFFSET,
            "^",
        ),
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
            curve[metric],
            "--" + marker,
            color=_BASELINE_CURVE_COLORS[run_name],
            label=run_name,
            linewidth=1.2,
            markersize=3.5,
            alpha=0.85,
        )


# ─── Plotters ──────────────────────────────────────────────────────────────


def _run_step(df: pd.DataFrame, run: str) -> int | None:
    """Pull the (single) step recorded for ``run`` in this df, or None.

    Eval rows for one run share one ckpt, hence one step. The data row's
    ``step`` is the authoritative source — the run id suffix (``_steplast``)
    is ambiguous across suites.
    """
    if "step" not in df.columns:
        return None
    steps = df[df["run"] == run]["step"].dropna().unique()
    if len(steps) == 0:
        return None
    return int(steps[0])


def plot_layer_curves_one_probe(
    df: pd.DataFrame,
    probe: str,
    outpath: Path,
    title: str,
) -> None:
    """All checkpoints' headline-metric vs layer for one probe kind, single panel."""
    spec = PROBE_SPEC[probe]
    metric = spec["metric"]
    ckpt_df, baseline_df = _split_baselines(df[df.probe_kind == probe])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    runs = sorted(ckpt_df["run"].unique())
    colors = {r: _PALETTE[i % len(_PALETTE)] for i, r in enumerate(runs)}
    for run in runs:
        r = ckpt_df[ckpt_df["run"] == run].sort_values("layer")
        if r.empty:
            continue
        ax.plot(
            r["layer"],
            r[metric],
            marker="o",
            color=colors[run],
            label=pretty_run_label(
                run, step=_run_step(ckpt_df, run), allow_missing_step=True
            ),
            linewidth=1.5,
            markersize=4,
        )
    _draw_trivial_baselines(ax, baseline_df, probe)
    _draw_baseline_curves(ax, baseline_df, probe)
    ax.set_title(f"{spec['title_short']}: {metric} vs layer", fontsize=11)
    ax.set_xlabel("transformer layer")
    ax.set_ylabel(spec["ylabel"])
    ax.grid(alpha=0.3)
    if spec["ylim"] is not None:
        ax.set_ylim(*spec["ylim"])
    ax.legend(fontsize=6, loc="best", ncol=2)
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


def plot_block_one_probe(
    df: pd.DataFrame,
    probe: str,
    block_name: str,
    run_labels: List[Tuple[str, str]],
    outpath: Path,
    sweep_title: str,
) -> None:
    """One ablation block, one probe — labelled curves vs trunk layer + baselines."""
    spec = PROBE_SPEC[probe]
    metric = spec["metric"]
    sub = df[(df.probe_kind == probe) & (df["layer"] >= 0)]
    _, baseline_df = _split_baselines(df[df.probe_kind == probe])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = {run: _PALETTE[i % len(_PALETTE)] for i, (run, _) in enumerate(run_labels)}
    for run, label in run_labels:
        r = sub[sub["run"] == run].sort_values("layer")
        if r.empty:
            continue
        ax.plot(
            r["layer"],
            r[metric],
            marker="o",
            color=colors[run],
            label=pretty_run_label(
                run, step=_run_step(sub, run), display=label, allow_missing_step=True
            ),
            linewidth=1.5,
            markersize=5,
        )
    _draw_trivial_baselines(ax, baseline_df, probe)
    _draw_baseline_curves(ax, baseline_df, probe)
    ax.set_title(f"{spec['title_short']} — {block_name}", fontsize=11)
    ax.set_xlabel("transformer layer")
    ax.set_ylabel(spec["ylabel"])
    ax.grid(alpha=0.3)
    if spec["ylim"] is not None:
        ax.set_ylim(*spec["ylim"])
    ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"{sweep_title}: {probe} — {block_name} ablation", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


def plot_distance_buckets(df: pd.DataFrame, outpath: Path, title: str) -> None:
    """Distance probe: 3-panel showing MAE per sequence-separation bucket
    (short |i-j|<6 / medium 6-24 / long ≥24) — illuminates where the gain
    (or lack thereof) actually is.
    """
    spec = PROBE_SPEC["distance"]
    ckpt_df, baseline_df = _split_baselines(df[df.probe_kind == "distance"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    runs = sorted(ckpt_df["run"].unique())
    colors = {r: _PALETTE[i % len(_PALETTE)] for i, r in enumerate(runs)}
    bucket_specs = [
        ("dist_mae_short", "short |i−j|<6"),
        ("dist_mae_medium", "medium 6≤|i−j|<24"),
        ("dist_mae_long", "long |i−j|≥24"),
    ]
    for ax, (col, label) in zip(axes, bucket_specs):
        for run in runs:
            r = ckpt_df[ckpt_df["run"] == run].sort_values("layer")
            if r.empty:
                continue
            ax.plot(
                r["layer"],
                r[col],
                marker="o",
                color=colors[run],
                label=pretty_run_label(
                    run, step=_run_step(ckpt_df, run), allow_missing_step=True
                ),
                linewidth=1.2,
                markersize=3.5,
            )
        # Trivial baselines (seqsep_pair + random_gauss): horizontal lines at
        # this bucket's value, to keep the comparison apples-to-apples.
        for run, sentinel, lbl, color in spec["trivial_baselines"]:
            row = baseline_df[
                (baseline_df["run"] == run) & (baseline_df["layer"] == sentinel)
            ]
            if row.empty:
                continue
            v = float(row[col].iloc[0])
            ax.axhline(v, linestyle="--", linewidth=1.0, color=color, alpha=0.85)
        ax.set_title(f"distance MAE — {label}", fontsize=11)
        ax.set_xlabel("transformer layer")
        ax.set_ylabel("MAE (Å)")
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=6, loc="best", ncol=2)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


def peak_table_one_probe(df: pd.DataFrame, probe: str) -> pd.DataFrame:
    """Best (run, layer) per probe — using max for higher-better, min otherwise."""
    spec = PROBE_SPEC[probe]
    metric = spec["metric"]
    sub = df[(df.probe_kind == probe) & (df["layer"] >= 0)].copy()
    if sub.empty:
        return pd.DataFrame()
    if spec["higher_is_better"]:
        idx = sub.groupby("run")[metric].idxmax()
    else:
        idx = sub.groupby("run")[metric].idxmin()
    out = sub.loc[idx, ["run", "layer", metric]].sort_values(
        metric, ascending=not spec["higher_is_better"]
    )
    out = out.set_index("run")
    return out


# ─── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sweep", required=True, choices=["paper_n128_struct", "paper_n256_struct"]
    )
    args = ap.parse_args()

    df = load_results(args.sweep)
    blocks = BLOCKS_N128 if args.sweep == "paper_n128_struct" else BLOCKS_N256
    fig_dir = _fig_dir(args.sweep)
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(df)
    n_ckpts = df[["run", "step"]].drop_duplicates().shape[0]
    print(f"[{args.sweep}] {n_rows} rows from {n_ckpts} unique (run, step)")

    # Baseline summary CSV.
    _, baseline_df = _split_baselines(df)
    if not baseline_df.empty:
        cols = ["run", "layer", "probe_kind"] + [
            c
            for c in (
                "if_top1_acc",
                "if_macro_f1",
                "dih_mae_phi_deg",
                "dih_mae_psi_deg",
                "dih_mae_total_deg",
                "dist_mae_total",
                "dist_mae_short",
                "dist_mae_medium",
                "dist_mae_long",
            )
            if c in baseline_df.columns
        ]
        baseline_df[cols].sort_values(["probe_kind", "run", "layer"]).to_csv(
            fig_dir / "table_baselines.csv",
            index=False,
        )
        print("  wrote table_baselines.csv")

    for probe in PROBE_SPEC:
        # Peak table.
        peak = peak_table_one_probe(df, probe)
        if not peak.empty:
            peak.to_csv(fig_dir / f"table_peak_{probe}.csv")
            print(f"  wrote table_peak_{probe}.csv")

        # Headline layer curves over all runs.
        plot_layer_curves_one_probe(
            df,
            probe,
            fig_dir / f"fig_layer_curves_{probe}.png",
            title=f"{args.sweep}: {probe}",
        )

        # Per-block figures (skip blocks where >half the runs are missing).
        for block_name, run_labels in blocks.items():
            present = sum(
                1
                for run, _ in run_labels
                if ((df["run"] == run) & (df["probe_kind"] == probe)).any()
            )
            if present < max(2, len(run_labels) // 2):
                print(
                    f"  skip block {probe}/{block_name} ({present}/{len(run_labels)} runs)"
                )
                continue
            plot_block_one_probe(
                df,
                probe,
                block_name,
                run_labels,
                fig_dir / f"fig_block_{probe}_{block_name}.png",
                sweep_title=args.sweep,
            )

    # Distance-only: per-bucket panel for short/medium/long sequence separation.
    plot_distance_buckets(
        df,
        fig_dir / "fig_distance_buckets.png",
        title=f"{args.sweep}: distance MAE by sequence-separation bucket",
    )


if __name__ == "__main__":
    main()
