"""At matched designability, does REPA reduce SS 2D-JSD?

Two-panel scatter (PDB / AFDB) with designability on x and SS-JSD on y. Each
checkpoint is one dot; baseline blue, all REPA variants pooled into red. A
binned-mean trend line per group makes "REPA below baseline at this
designability" readable at a glance — if the red trend lies under blue across
all bins, REPA strictly improves SS preservation conditional on designability.

Output: ``figures/paper/n128_convergence/ssjsd_vs_des.{png,pdf}``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/paper"
FIG_OUT = ROOT / "figures/paper/n128_convergence"
FIG_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "PDB": RESULTS / "n128_convergence_pdb" / "sweep_results.jsonl",
    "AFDB": RESULTS / "n128_convergence_afdb" / "sweep_results.jsonl",
}
SS_COL = {
    "PDB": "_res_ss_jsd_pdb_designable_2d",
    "AFDB": "_res_ss_jsd_afdb_designable_2d",
}
BASELINES = {"PDB": ["baseline_128_bs80"], "AFDB": ["baseline_afdb_128_bs80"]}
REPA = {
    "PDB": [
        "repa_l4_128_bs80",
        "repa_l9_128_bs80",
        "repa_mpnn_l4_128_bs80",
        "repa_mpnn_l9_128_bs80_2gpu",
    ],
    "AFDB": [
        "repa_l4_afdb_128_bs80",
        "repa_mpnn_l4_afdb_128_bs80",
        "repa_mpnn_l9_afdb_128_bs80_2gpu",
    ],
}


def load_jsonl(path: Path):
    rows = [json.loads(line) for line in path.open()]
    d = {}
    for r in rows:
        d[(r.get("run"), r.get("step"))] = r
    return [r for r in d.values() if r.get("error", "NONE") == "NONE"]


def pts_for(rows, prefixes, xcol, ycol):
    out = []
    for r in rows:
        run = r.get("run", "")
        if not any(run.startswith(p) for p in prefixes):
            continue
        x, y = r.get(xcol), r.get(ycol)
        if x is None or y is None:
            continue
        out.append((float(x), float(y)))
    return out


def binned_mean(pts, n_bins=5, x_range=(0.0, 1.0)):
    """Mean y per equal-width bin in x. Returns (bin_centers, means, counts)."""
    if not pts:
        return np.array([]), np.array([]), np.array([])
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    edges = np.linspace(*x_range, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = (xs >= edges[i]) & (
            xs < edges[i + 1] if i < n_bins - 1 else xs <= edges[i + 1]
        )
        if m.any():
            means[i] = ys[m].mean()
            counts[i] = int(m.sum())
    return centers, means, counts


def _humanize(v, _pos=None):
    av = abs(v)
    if av >= 1:
        return f"{v:g}"
    return f"{v:.3g}"


def main() -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.0 * len(DATASETS), 4.6))
    for col_i, (ds, path) in enumerate(DATASETS.items()):
        rows = load_jsonl(path)
        ycol = SS_COL[ds]
        xcol = "_res_designability_rate"
        base = pts_for(rows, BASELINES[ds], xcol, ycol)
        repa = pts_for(rows, REPA[ds], xcol, ycol)
        # _res_designability_rate is 0–1 → show as 0–100 %.
        base = [(x * 100.0, y) for x, y in base]
        repa = [(x * 100.0, y) for x, y in repa]
        ax = axes[col_i]

        if base:
            bx, by = zip(*base)
            ax.scatter(
                bx,
                by,
                color="tab:blue",
                marker="s",
                s=40,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.55,
                zorder=3,
                label=f"Baseline (n={len(base)})",
            )
        if repa:
            rx, ry = zip(*repa)
            ax.scatter(
                rx,
                ry,
                color="tab:red",
                marker="o",
                s=30,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.55,
                zorder=3,
                label=f"REPA (all variants, n={len(repa)})",
            )

        ax.set_xlabel("Designability (%) ↑")
        ax.set_ylabel(f"SS 2D-JSD vs {ds} (designable) ↓")
        ax.invert_yaxis()  # up = better
        ax.set_title(f"{ds} — does REPA preserve SS at matched designability?")
        ax.grid(False)
        ax.grid(True, axis="both", alpha=0.25)
        ax.xaxis.set_major_formatter(FuncFormatter(_humanize))
        ax.yaxis.set_major_formatter(FuncFormatter(_humanize))
        ax.legend(loc="best", fontsize=8, framealpha=0.85)

    fig.suptitle(
        "n=128 — SS 2D-JSD conditional on designability\n"
        "Red points below blue at a given designability ⇒ REPA preserves SS better at that rate.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = FIG_OUT / "ssjsd_vs_des.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
