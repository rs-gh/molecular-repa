"""Helix/sheet (H/E) ratio over training — companion to convergence plots.

Reads ``sweep_results.jsonl`` from the n128_convergence_{pdb,afdb} sweeps and
plots H/E ratio of the *designable* generated structures vs training step,
one curve per run.

Why this metric: many flow-matching protein generators collapse toward
all-α (high H) outputs under designability pressure; tracking H/E reveals
secondary-structure preference drift over training. Restricting to the
designable subset isolates "what kind of proteins are we actually getting"
from noisy/unfolded outputs.

Output: ``figures/paper/n128_convergence/helix_sheet_ratio.{png,pdf}``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


def _humanize(v, _pos=None):
    av = abs(v)
    if av >= 1e6:
        return f"{v / 1e6:g}M"
    if av >= 1e3:
        return f"{v / 1e3:g}K"
    if av >= 1:
        return f"{v:g}"
    return f"{v:.3g}"


ROOT = Path(__file__).resolve().parents[2]  # .../proteina/generation
RESULTS = ROOT / "results/paper"
FIG_OUT = ROOT / "figures/paper/n128_convergence"
FIG_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "PDB": RESULTS / "n128_convergence_pdb" / "sweep_results.jsonl",
    "AFDB": RESULTS / "n128_convergence_afdb" / "sweep_results.jsonl",
}

RUN_FAMILIES = {
    "PDB": [
        ("baseline_128_bs80", "Baseline (PDB)", "tab:blue", "-"),
        ("repa_l4_128_bs80", "REPA L4 GearNet (PDB)", "tab:red", "-"),
        (
            "repa_l9_128_bs80",
            "REPA L9 GearNet (PDB)",
            "tab:orange",
            "-",
        ),
        ("repa_mpnn_l4_128_bs80", "REPA L4 MPNN (PDB)", "tab:red", "--"),
        ("repa_mpnn_l9_128_bs80_2gpu", "REPA L9 MPNN (PDB)", "tab:green", "--"),
        (
            "repa_l4_128_random",
            "REPA L4 random-GN (PDB)",
            "tab:gray",
            ":",
        ),
    ],
    "AFDB": [
        ("baseline_afdb_128_bs80", "Baseline (AFDB)", "tab:blue", "-"),
        ("repa_l4_afdb_128_bs80", "REPA L4 GearNet (AFDB)", "tab:red", "-"),
        ("repa_mpnn_l4_afdb_128_bs80", "REPA L4 MPNN (AFDB)", "tab:red", "--"),
        ("repa_mpnn_l9_afdb_128_bs80_2gpu", "REPA L9 MPNN (AFDB)", "tab:green", "--"),
    ],
}


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.open()]
    dedup: Dict[tuple, Dict] = {}
    for r in rows:
        dedup[(r.get("run"), r.get("step"))] = r
    return [r for r in dedup.values() if r.get("error", "NONE") == "NONE"]


def extract_ratio(
    rows: List[Dict], run_prefix: str, designable: bool = True
) -> Tuple[List[int], List[float]]:
    """Pull (step, H/E_ratio). designable=True uses the designable-subset SS
    fractions; H/E = ss_frac_H / max(ss_frac_E, 1e-6)."""
    h_col = "_res_ss_frac_H_designable" if designable else "_res_ss_frac_H"
    e_col = "_res_ss_frac_E_designable" if designable else "_res_ss_frac_E"
    pts = []
    for r in rows:
        run = r.get("run", "")
        if not run.startswith(run_prefix):
            continue
        h = r.get(h_col)
        e = r.get(e_col)
        s = r.get("step")
        if h is None or e is None or s is None:
            continue
        if e <= 0:
            continue
        pts.append((int(s), float(h) / float(e)))
    pts.sort()
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def main() -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 4.5), sharey=False)
    for ax, (ds_name, path) in zip(axes, DATASETS.items()):
        rows = load_jsonl(path)
        for prefix, label, color, ls in RUN_FAMILIES[ds_name]:
            xs, ys = extract_ratio(rows, prefix, designable=True)
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=4,
                linewidth=1.4,
                color=color,
                linestyle=ls,
                label=label,
            )
        ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.4, linestyle=":")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(
            LogLocator(base=10.0, subs=(1.0, 2.0, 4.0, 7.0), numticks=20)
        )
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(_humanize))
        ax.yaxis.set_major_formatter(FuncFormatter(_humanize))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Training step")
        ax.set_ylabel("H/E ratio (designable)")
        ax.invert_yaxis()  # closer to 1 = better; flip log axis so 1 sits nearer the top
        ax.set_title(f"{ds_name}: helix/sheet ratio over training")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    fig.suptitle(
        "n=128 — Helix/Sheet ratio of designable generations vs training step\n"
        "Dotted horizontal line = 1 (equal). Higher = more α-helical bias.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = FIG_OUT / "helix_sheet_ratio.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
