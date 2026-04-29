"""Plots for the pretrained-probe contact sweep.

Reads ``pretrained_sweep_results.csv`` and emits figures into
``evaluation/proteina/representation/figures/pretrained_probe/``.

Figures produced
----------------
fig_layerwise_p_at_L.png
    1 x 3 grid (n=128, 256, 512), y = P@L, x = transformer layer.
    One curve per run family + pretrained_dfs_60m reference.

fig_layerwise_p_at_L_2.png
    Same layout, y = P@L/2.

fig_layerwise_p_at_L_5.png
    Same layout, y = P@L/5.

Usage
-----
  python evaluation/proteina/representation/scripts/plot_pretrained_probe.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "results" / "pretrained_probe" / "pretrained_sweep_results.csv"
FIG_DIR = HERE.parent / "figures" / "pretrained_probe"

# -- colour scheme matches plot_per_n.py ---------------------------------------
FAMILY_COLORS = {
    "baseline": "#1f77b4",  # blue
    "repa_l0": "#ff7f0e",  # orange
    "repa_l4": "#d62728",  # red
    "repa_l9": "#2ca02c",  # green
    "esm_repa_l0": "#ffbb78",  # light orange
    "esm_repa_l4": "#ff9896",  # light red
    "esm_repa_l9": "#98df8a",  # light green
    "pretrained_dfs_60m": "#9467bd",  # purple - reference ceiling
}

# -- run-name -> (n_bucket, family) ---------------------------------------------
# "baseline" alone is the 400k n=512 checkpoint; "baseline_512_sm" is 500k.
# We keep both in the CSV but only plot the later one for n=512.
RUN_META: dict[str, tuple[str, str]] = {
    "baseline": ("n512", "baseline"),  # 400k; shadowed by baseline_512_sm below
    "baseline_128": ("n128", "baseline"),
    "baseline_256": ("n256", "baseline"),
    "baseline_512_sm": ("n512", "baseline"),  # 500k - preferred n=512 baseline
    "repa_l0_128": ("n128", "repa_l0"),
    "repa_l4_128": ("n128", "repa_l4"),
    "repa_l9_128": ("n128", "repa_l9"),
    "repa_l0_256": ("n256", "repa_l0"),
    "repa_l4_256": ("n256", "repa_l4"),
    "repa_l9_256": ("n256", "repa_l9"),
    "repa_l0_512_sm": ("n512", "repa_l0"),
    "repa_l4_512_sm": ("n512", "repa_l4"),
    "repa_l9_512_sm": ("n512", "repa_l9"),
    "esm_repa_l0_128": ("n128", "esm_repa_l0"),
    "esm_repa_l4_128": ("n128", "esm_repa_l4"),
    "esm_repa_l9_128": ("n128", "esm_repa_l9"),
}
PRETRAINED_RUN = "pretrained_dfs_60m"  # shown on every panel as reference

# When two runs map to the same (n_bucket, family) prefer the later step.
# "baseline" (400k) vs "baseline_512_sm" (500k) -> keep 500k only for n512.
PREFERRED_RUN: dict[tuple[str, str], str] = {
    ("n512", "baseline"): "baseline_512_sm",
}

N_LABELS = {"n128": "n = 128", "n256": "n = 256", "n512": "n = 512"}
N_ORDER = ["n128", "n256", "n512"]

FAMILY_LABEL = {
    "baseline": "baseline",
    "repa_l0": "REPA L0 (GearNet)",
    "repa_l4": "REPA L4 (GearNet)",
    "repa_l9": "REPA L9 (GearNet)",
    "esm_repa_l0": "REPA L0 (ESM2)",
    "esm_repa_l4": "REPA L4 (ESM2)",
    "esm_repa_l9": "REPA L9 (ESM2)",
    "pretrained_dfs_60m": "pretrained DFS-60M (12L)",
}

METRIC_LABELS = {
    "p_at_L": "P@L",
    "p_at_L_2": "P@L/2",
    "p_at_L_5": "P@L/5",
}


def _load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    for c in ["p_at_L", "p_at_L_2", "p_at_L_5", "layer", "step"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["layer", "p_at_L"])
    return df


def _panel(ax: plt.Axes, df: pd.DataFrame, n_bucket: str, metric: str) -> None:
    """Draw one panel for a given n_bucket and metric."""

    # Runs that belong to this n_bucket
    n_runs = {run: family for run, (nb, family) in RUN_META.items() if nb == n_bucket}

    # When two runs map to the same family, keep only the preferred one
    plotted_families: set[str] = set()
    for run, family in sorted(n_runs.items(), key=lambda kv: kv[0]):
        preferred = PREFERRED_RUN.get((n_bucket, family))
        if preferred is not None and run != preferred:
            continue
        if family in plotted_families:
            continue

        sub = df[df["run"] == run].sort_values("layer")
        if sub.empty:
            continue

        color = FAMILY_COLORS.get(family, "gray")
        ax.plot(
            sub["layer"],
            sub[metric],
            "-o",
            color=color,
            label=FAMILY_LABEL.get(family, family),
            linewidth=1.8,
            markersize=4,
        )
        plotted_families.add(family)

    # Pretrained DFS-60M reference (12 transformer layers, different depth)
    ref = df[df["run"] == PRETRAINED_RUN].sort_values("layer")
    if not ref.empty:
        ax.plot(
            ref["layer"],
            ref[metric],
            "--^",
            color=FAMILY_COLORS[PRETRAINED_RUN],
            label=FAMILY_LABEL[PRETRAINED_RUN],
            linewidth=1.5,
            markersize=4,
            alpha=0.85,
        )

    ax.set_xlabel("Transformer layer")
    ax.grid(True, alpha=0.3)


def _plot_grid(metric: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    df = _load()

    for col, n_bucket in enumerate(N_ORDER):
        ax = axes[col]
        _panel(ax, df, n_bucket, metric)
        ax.set_title(N_LABELS[n_bucket], fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)

    # De-duplicated shared legend below the plots
    handles, labels = [], []
    seen: set[str] = set()
    for ax in axes:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in seen:
                seen.add(lbl)
                handles.append(h)
                labels.append(lbl)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
    )
    fig.suptitle(
        f"Contact probe {METRIC_LABELS[metric]} - layerwise by model size (t = 1.0)",
        fontsize=13,
        fontweight="bold",
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    for metric in ["p_at_L", "p_at_L_2", "p_at_L_5"]:
        _plot_grid(metric, FIG_DIR / f"fig_layerwise_{metric}.png")


if __name__ == "__main__":
    main()
