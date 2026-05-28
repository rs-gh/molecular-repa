"""Plot the denoised cos_sim trajectories (mean ± std band) for the four
REPA runs, to test whether cos_sim plateaus differently across encoders.

Conclusion (see h1_cossim_denoised.csv): it does NOT. All four runs show a
monotone, decelerating rise. Absolute levels differ (not comparable across
encoder/layer), but the shapes are the same — and the PDB-L9-GN run that
shows the sharp 700K T-D cliff has the MOST persistent late rise, not an
early plateau. So the "REPA saturates early → cliff" mechanism is not
supported by cos_sim.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[5]
CSV = (
    REPO_ROOT / "evaluation/proteina/generation/results/variance/h1_cossim_denoised.csv"
)
OUT = (
    REPO_ROOT
    / "evaluation/proteina/generation/figures/paper/n256_sampler_ablation/h1_cossim_denoised.png"
)

LABELS = {
    "PDB_L9_GN": ("PDB L9-GearNet (has 700K cliff)", "C3"),
    "AFDB_L4_GN": ("AFDB L4-GearNet (no cliff)", "C0"),
    "AFDB_L9_GN": ("AFDB L9-GearNet (no cliff)", "C2"),
    "AFDB_MPNN_L9": ("AFDB L9-MPNN (no cliff, α-shift)", "C1"),
}


def main():
    data = defaultdict(lambda: {"x": [], "m": [], "s": []})
    for r in csv.DictReader(open(CSV)):
        d = data[r["run"]]
        d["x"].append(int(r["target"]) / 1000)
        d["m"].append(float(r["cos_sim_mean"]))
        d["s"].append(float(r["cos_sim_std"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute trajectories with ±1 std bands.
    for run, (lab, c) in LABELS.items():
        if run not in data:
            continue
        d = data[run]
        x, m, s = d["x"], d["m"], d["s"]
        ax1.plot(x, m, "-o", color=c, label=lab, markersize=4)
        ax1.fill_between(
            x,
            [a - b for a, b in zip(m, s)],
            [a + b for a, b in zip(m, s)],
            color=c,
            alpha=0.12,
        )
    ax1.axvline(
        700, color="k", linestyle=":", linewidth=0.8, label="700K (cliff for PDB-L9-GN)"
    )
    ax1.set_xlabel("training step (K)")
    ax1.set_ylabel("cos_sim (aligned layer), windowed mean ±1σ")
    ax1.set_title("Absolute cos_sim — levels not comparable across encoders")
    ax1.legend(fontsize=8)

    # Right: normalized to each run's 100K value, to compare SHAPES.
    for run, (lab, c) in LABELS.items():
        if run not in data:
            continue
        d = data[run]
        x, m = d["x"], d["m"]
        base = m[0]
        ax2.plot(x, [v - base for v in m], "-o", color=c, label=lab, markersize=4)
    ax2.axvline(700, color="k", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("training step (K)")
    ax2.set_ylabel("cos_sim − cos_sim(100K)")
    ax2.set_title(
        "Shape (zeroed at 100K): all monotone-decelerating;\nPDB-L9-GN rises MOST late, no early plateau"
    )
    ax2.legend(fontsize=8)

    fig.suptitle(
        "Denoised cos_sim trajectories (±10K-step windows, n≈20k/point)", y=1.02
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
