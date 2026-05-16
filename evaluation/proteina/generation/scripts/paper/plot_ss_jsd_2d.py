"""Plot 3-bin vs 2D-binned SS JSD per run.

Reads `ss_jsd_2d_comparison.tsv` (output of compute_ss_jsd_2d.py) and produces
a scatter for n=128 and n=256 showing where each run lands on the
(3-bin JSD, 2D-bin JSD) plane. Off-diagonal points are runs whose centroid
matched the reference but whose distribution shape did not — exactly the
mode-collapse signal the centroid-only metric was missing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from evaluation.proteina.lib.plot_labels import pretty_run_label

ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = ROOT / "figures/paper"
TSV = FIG_ROOT / "ss_jsd_2d_comparison.tsv"


def to_float(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def load_rows():
    with TSV.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def plot_one(rows, n_label: str, out: Path):
    pts = [r for r in rows if r["n"] == n_label]
    if not pts:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    panels = [
        (axes[0], "ss_jsd_pdb_3bin", "ss_jsd_pdb_2d", "PDB ref"),
        (axes[1], "ss_jsd_afdb_3bin", "ss_jsd_afdb_2d", "AFDB ref"),
    ]
    for ax, k3, k2, title in panels:
        x_vals, y_vals, labels = [], [], []
        for p in pts:
            x = to_float(p[k3])
            y = to_float(p[k2])
            if x is None or y is None:
                continue
            x_vals.append(x)
            y_vals.append(y)
            # Step is encoded in the run id suffix (e.g. _step200k); pretty_run_label
            # parses it. For _steplast / _epNN runs we fall back to the raw id —
            # ambiguity already flagged in lineage audit.
            labels.append(pretty_run_label(p["run"], allow_missing_step=True))

        ax.scatter(
            x_vals,
            y_vals,
            s=70,
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            zorder=3,
        )
        for x, y, lbl in zip(x_vals, y_vals, labels):
            ax.annotate(
                lbl,
                (x, y),
                fontsize=6.5,
                xytext=(4, 3),
                textcoords="offset points",
                alpha=0.85,
                zorder=4,
            )

        if x_vals and y_vals:
            lo = 0.0
            hi = max(max(x_vals), max(y_vals)) * 1.05
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4)
            ax.text(
                hi * 0.98,
                hi * 0.98,
                "y = x",
                fontsize=8,
                ha="right",
                va="top",
                style="italic",
                color="dimgray",
            )

        ax.set_xlabel(f"3-bin JSD vs {title} (centroid only)")
        ax.set_ylabel(f"2D-bin JSD vs {title} (5x5 on f_H × f_E)")
        ax.set_title(f"n={n_label} — SS JSD: 3-bin vs 2D ({title})")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"n={n_label} — Distribution-shape sensitivity. "
        "Distance above the diagonal = how much shape disagrees with reference even when the mean matches.",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    rows = load_rows()
    plot_one(rows, "128", FIG_ROOT / "n128_paper/n128_ss_jsd_2d.png")
    plot_one(rows, "256", FIG_ROOT / "n256_paper/n256_ss_jsd_2d.png")


if __name__ == "__main__":
    main()
