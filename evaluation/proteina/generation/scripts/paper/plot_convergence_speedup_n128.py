"""n=128 convergence speedup — companion to plot_convergence_speedup.py.

Uses the existing n128_paper_afdb sweep data (not a dedicated n128 convergence
sweep — there isn't one). Coverage is partial:
  baseline:           step ∈ {200k, 400k, 600k, 800k, 1000k, 1200k}
  REPA L4 (GearNet):  step ∈ {200k, 600k}              (only 2 points)
  REPA L4 (MPNN):     step ∈ {200k, 400k, 600k, 800k, 1000k}
  REPA L9 (any):      none on AFDB n=128 — omitted
  PDB n=128 step-curve: not available — single-panel AFDB only

Output: ``figures/paper/n128_convergence/convergence_speedup.{png,pdf}``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/paper"
FIG_OUT = ROOT / "figures/paper/n128_convergence"
FIG_OUT.mkdir(parents=True, exist_ok=True)

JSONL = RESULTS / "n128_paper_afdb" / "sweep_results.jsonl"

RUN_FAMILIES = [
    ("baseline_afdb_128_bs80", "Baseline (AFDB n=128)", "tab:blue", "-"),
    ("repa_l4_afdb_128_bs80", "REPA L4 GearNet (AFDB n=128, partial)", "tab:red", ":"),
    ("repa_mpnn_l4_afdb_128_bs80", "REPA L4 MPNN (AFDB n=128)", "tab:red", "--"),
]

METRICS = [
    ("_res_designability_rate", "Designability", "rate", True),
    (
        "_res_diversity_clusters_total",
        "Diversity",
        "# clusters (designable subset)",
        True,
    ),
    (
        "_res_novelty_foldseek_pdb_rate",
        "Novelty vs PDB",
        "fraction novel (TM<0.5)",
        True,
    ),
    (
        "_res_novelty_foldseek_afdb_swissprot_rate",
        "Novelty vs AFDB-SP",
        "fraction novel (TM<0.5)",
        True,
    ),
    ("_res_PDB_FID", "FID (vs PDB)", "FID-50K", False),
    (
        "_res_ss_jsd_pdb_designable_2d",
        "SS 2D-JSD vs PDB (des)",
        "JSD (lower = closer)",
        False,
    ),
    (
        "_res_ss_jsd_afdb_designable_2d",
        "SS 2D-JSD vs AFDB (des)",
        "JSD (lower = closer)",
        False,
    ),
]


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.open()]
    dedup: Dict[tuple, Dict] = {}
    for r in rows:
        dedup[(r.get("run"), r.get("step"))] = r
    return list(dedup.values())


def extract(rows, prefix, metric):
    pts = []
    for r in rows:
        run = r.get("run", "")
        if not run.startswith(prefix):
            continue
        if r.get("error", "NONE") != "NONE":
            continue
        v = r.get(metric)
        s = r.get("step")
        if v is None or s is None:
            continue
        pts.append((int(s), float(v)))
    pts.sort()
    return ([p[0] for p in pts], [p[1] for p in pts])


def main() -> None:
    rows = load_jsonl(JSONL)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.0))
    for ax, (metric, title, ylabel, higher) in zip(axes, METRICS):
        for prefix, label, color, ls in RUN_FAMILIES:
            xs, ys = extract(rows, prefix, metric)
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
        ax.set_xscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"AFDB n=128 — {title}")
        ax.grid(True, alpha=0.3)
        if not higher:
            ax.invert_yaxis()
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle(
        "n=128 convergence (AFDB) — generation metrics vs training step\n"
        "Partial data: PDB n=128 step-curve not available; L9 variants not trained on AFDB n=128.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = FIG_OUT / "convergence_speedup.png"
    out_pdf = FIG_OUT / "convergence_speedup.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
