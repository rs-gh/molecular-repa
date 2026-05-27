"""Plot per-residue and per-protein CKNNA matrices as layer-wise curves.

Renders two figures:
    figures/cknna_n10k_per_residue.png
    figures/cknna_n10k_per_protein.png

Each is a 1x3 grid (one panel per encoder column), with one line per model
row coloured/markered per project convention:
  blue = baseline, red = L4, green = L9
  circle = CA-GearNet target, triangle = ProteinMPNN target

Run:
    source .venv/bin/activate
    python evaluation/proteina/alignment/scripts/plot_cknna.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ALIGN_ROOT = HERE.parent
RESULTS = ALIGN_ROOT / "results"
FIG_DIR = RESULTS / "figures"

MODEL_ROWS = [
    "baseline",
    "repa_gearnet_l4",
    "repa_gearnet_l9",
    "repa_mpnn_l4",
    "repa_mpnn_l9",
]
ENCODER_COLS = ["gearnet", "mpnn", "esm2"]
MODEL_DISPLAY = {
    "baseline": "Baseline",
    "repa_gearnet_l4": "REPA-GearNet (L4)",
    "repa_gearnet_l9": "REPA-GearNet (L9)",
    "repa_mpnn_l4": "REPA-MPNN (L4)",
    "repa_mpnn_l9": "REPA-MPNN (L9)",
}
ENCODER_DISPLAY = {
    "gearnet": "CA-GearNet",
    "mpnn": "ProteinMPNN",
    "esm2": "ESM-2",
}
COLORS = {
    "baseline": "#1f77b4",
    "repa_gearnet_l4": "#d62728",
    "repa_gearnet_l9": "#2ca02c",
    "repa_mpnn_l4": "#d62728",
    "repa_mpnn_l9": "#2ca02c",
}
MARKERS = {
    "baseline": "o",
    "repa_gearnet_l4": "o",
    "repa_gearnet_l9": "o",
    "repa_mpnn_l4": "^",
    "repa_mpnn_l9": "^",
}


def _plot(mode: str) -> None:
    matrix_path = RESULTS / f"cknna_matrix_{mode}.jsonl"
    out_path = FIG_DIR / f"cknna_n10k_{mode}.png"
    if not matrix_path.exists():
        print(f"  [skip] {matrix_path} not found")
        return

    rows = []
    with open(matrix_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} cells from {matrix_path.name}")
    if not rows:
        return

    by_panel: dict = defaultdict(list)
    for r in rows:
        by_panel[(r["encoder"], r["model"])].append(
            (r["layer"], r["cknna"], r["lo5"], r["hi95"])
        )
    for k in by_panel:
        by_panel[k].sort(key=lambda x: x[0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, enc in zip(axes, ENCODER_COLS):
        for model in MODEL_ROWS:
            data = by_panel.get((enc, model))
            if not data:
                continue
            xs = [d[0] for d in data]
            ys = [d[1] for d in data]
            lo = [d[2] for d in data]
            hi = [d[3] for d in data]
            ax.plot(
                xs,
                ys,
                marker=MARKERS[model],
                label=MODEL_DISPLAY[model],
                color=COLORS[model],
                lw=1.8,
                markersize=6,
            )
            ax.fill_between(xs, lo, hi, alpha=0.18, color=COLORS[model], lw=0)
        for inj in (4, 9):
            ax.axvline(inj, color="black", lw=0.6, ls=":", alpha=0.45)
        ax.set_title(f"vs {ENCODER_DISPLAY[enc]}", fontsize=11)
        ax.set_xlabel("Proteina layer index")
        ax.grid(alpha=0.25)
        ax.set_xticks(range(10))
    axes[0].set_ylabel("CKNNA (k=10)")
    axes[0].legend(loc="best", fontsize=9, frameon=False)
    mode_pretty = (
        "per-residue" if mode == "per_residue" else "per-protein (mean-pooled)"
    )
    fig.suptitle(
        f"Per-layer CKNNA ({mode_pretty}): proteina hidden states "
        f"(n=256 PDB, t=1.0 clean, step=1000k, N=10,000) vs frozen encoders",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    for mode in ("per_protein", "per_residue"):
        _plot(mode)


if __name__ == "__main__":
    main()
