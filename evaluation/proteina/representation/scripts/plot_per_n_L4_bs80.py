"""Per-layer probe plots for the n=128 L4 bs=80 sweep.

Reads results/n128_L4_bs80_val/sweep_results.csv and produces a 2 x 3 grid:
    rows = step bucket (mid ~8M / end ~16M samples)
    cols = probe timestep t in {0.5, 0.75, 1.0}
    x    = transformer layer (0..9 for our 10-layer 60M model)
    y    = p_at_L_5_linear (P@L/5, linear probe)
    one curve per run (4 students), plus horizontal reference lines for
    the frozen baselines that the sweep harness writes for free
    (gearnet, seq_onehot, random_gauss, distance_only).

A second figure with the same layout but y = cath_acc is also written.

Mirrors plot_per_n.py but groups by step bucket instead of model size and
restricts the family set to the four bs=80 students.

Renamed 2026-05-06 from plot_per_n_bs80_sweep.py -> plot_per_n_L4_bs80.py
to make the layer-4-only scope explicit (cf. plot_per_n.py, which covers the
L0/L4/L9 layer ablation across n=128/256/512).

Usage:
    python evaluation/proteina/representation/scripts/plot_per_n_L4_bs80.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "n128_L4_bs80_val"
FIG_DIR = HERE.parent / "figures"

# Per-checkpoint colour, matches the gen-side bs80 plot for consistency
RUN_COLORS = {
    "baseline_128_bs80": "#4A90D9",
    "repa_l4_128_bs80": "#E74C3C",
    "repa_l4_128_bs80_lr3x": "#F39C12",
    "repa_l4_128_random": "#55A868",
}
RUN_LABELS = {
    "baseline_128_bs80": "Baseline",
    "repa_l4_128_bs80": "REPA L4",
    "repa_l4_128_bs80_lr3x": "REPA L4 (lr3x)",
    "repa_l4_128_random": "REPA L4 (random)",
}

# Frozen-encoder reference rows are encoded with negative `layer` sentinels by
# the sweep harness; mirror plot_per_n.py.
REF_SENTINELS = {
    -1: "gearnet",
    -2: "random_gauss",
    -3: "seq_onehot",
    -4: "distance_only",
}
REF_COLORS = {
    "gearnet": "black",
    "seq_onehot": "#8c564b",
    "random_gauss": "#7f7f7f",
    "distance_only": "#17becf",
}

MID_STEP = 100000
BUCKET_LABEL = {
    "mid": "8M samples (step 100K)",
    "end": "~16M samples (step 200K / lr3x last)",
}
TS = [0.50, 0.75, 1.00]


def _load() -> pd.DataFrame:
    csv = RESULTS / "sweep_results.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No sweep results at {csv}")
    df = pd.read_csv(csv)
    df = df[df["error"].isna() | (df["error"] == "")].copy()
    for c in ["layer", "step", "t", "p_at_L_5_linear", "p_at_L_5_mlp", "cath_acc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["layer", "t"])
    return df


def _bucket(step: float) -> str:
    return "mid" if int(step) == MID_STEP else "end"


def _ref_values(df: pd.DataFrame, metric: str) -> dict:
    out = {}
    for sentinel, name in REF_SENTINELS.items():
        sub = df[df["layer"] == sentinel]
        if not sub.empty:
            v = float(sub[metric].mean())
            if v == v:
                out[name] = v
    return out


def _panel(ax, df: pd.DataFrame, bucket: str, t: float, metric: str) -> None:
    sub = df[(df["layer"] >= 0) & (df["t"].round(2) == round(t, 2))].copy()
    sub = sub[sub["step"].apply(_bucket) == bucket]
    for run, color in RUN_COLORS.items():
        cur = sub[sub["run"] == run].sort_values("layer")
        if cur.empty:
            continue
        ax.plot(
            cur["layer"],
            cur[metric],
            "-o",
            color=color,
            label=RUN_LABELS[run],
            linewidth=1.6,
            markersize=3.5,
        )

    for name, v in _ref_values(df, metric).items():
        ax.axhline(
            v,
            linestyle="--",
            linewidth=0.9,
            color=REF_COLORS[name],
            label=name,
            alpha=0.7,
        )

    ax.grid(True, alpha=0.3)


def _plot_grid(df: pd.DataFrame, metric: str, ylabel: str, out_name: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    for row, bucket in enumerate(["mid", "end"]):
        for col, t in enumerate(TS):
            ax = axes[row, col]
            _panel(ax, df, bucket, t, metric)
            if row == 0:
                ax.set_title(f"t = {t:.2f}", fontsize=11, fontweight="bold")
            if row == 1:
                ax.set_xlabel("Transformer layer")
        axes[row, 0].set_ylabel(
            f"{BUCKET_LABEL[bucket]}\n{ylabel}", fontsize=10, fontweight="bold"
        )

    handles, labels, seen = [], [], set()
    for ax_row in axes:
        for ax in ax_row:
            for h, lbl in zip(*ax.get_legend_handles_labels()):
                if lbl in seen:
                    continue
                seen.add(lbl)
                handles.append(h)
                labels.append(lbl)
    fig.legend(
        handles, labels, loc="center right", fontsize=9, bbox_to_anchor=(1.0, 0.5)
    )

    fig.suptitle(
        f"n=128 bs=80 sweep - {ylabel} (layerwise, per timestep)",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 0.88, 0.98])
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    df = _load()
    _plot_grid(
        df, "p_at_L_5_linear", "P@L/5 (linear)", "n128_L4_bs80_per_layer_PaL5.png"
    )
    _plot_grid(df, "cath_acc", "CATH accuracy", "n128_L4_bs80_per_layer_cath.png")
