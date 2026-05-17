"""Convergence speedup plot — REPA paper Fig 1 analog for proteina.

Reads ``sweep_results.jsonl`` from the n256_convergence_{pdb,afdb} sweeps and
emits a 2 (datasets) x 4 (metrics) panel showing each metric vs training step
with one line per run. The four metrics are designability, diversity (cluster
count), novelty (foldseek vs matching DB), and FID (PDB / AFDB).

One curve per run; baseline drawn first (per `feedback_baseline_first.md`),
REPA variants in red/orange. Latest-by-(run, step) dedup so retry rows don't
double-count and ERR rows (designability=None) drop out cleanly.

Output: ``evaluation/proteina/generation/figures/paper/n256_convergence/convergence_speedup.{png,pdf}``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # .../proteina/generation
RESULTS = ROOT / "results/paper"
FIG_OUT = ROOT / "figures/paper/n256_convergence"
FIG_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "PDB": RESULTS / "n256_convergence_pdb" / "sweep_results.jsonl",
    "AFDB": RESULTS / "n256_convergence_afdb" / "sweep_results.jsonl",
}

# Run families per dataset — match RUN_SCHEDULES labels (without step suffix).
# baseline first, REPA variants after (per feedback_baseline_first.md).
RUN_FAMILIES = {
    "PDB": [
        ("baseline_256_bs24_2gpu", "Baseline (PDB)", "tab:blue", "-"),
        ("repa_l4_256_per_residue_bs24_2gpu", "REPA L4 GearNet (PDB)", "tab:red", "-"),
        (
            "repa_l9_256_per_residue_bs24_2gpu",
            "REPA L9 GearNet (PDB)",
            "tab:orange",
            "-",
        ),
        ("repa_mpnn_l4_256_per_residue", "REPA L4 MPNN (PDB)", "tab:red", "--"),
        # PDB L9 MPNN not trained yet
    ],
    "AFDB": [
        ("baseline_afdb_256", "Baseline (AFDB)", "tab:blue", "-"),
        ("repa_l4_afdb_256", "REPA L4 GearNet (AFDB)", "tab:red", "-"),
        ("repa_l9_afdb_256", "REPA L9 GearNet (AFDB, partial)", "tab:orange", ":"),
        ("repa_mpnn_l4_afdb_256", "REPA L4 MPNN (AFDB)", "tab:red", "--"),
        ("repa_mpnn_l9_afdb_256", "REPA L9 MPNN (AFDB)", "tab:orange", "--"),
    ],
}

# Metric column → (panel title, y-axis label, higher_is_better)
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
    # SS-JSD on the (H,E)/(H,C) joint distribution restricted to the designable
    # subset — captures whether the surviving designable structures match the
    # reference SS distribution; lower = closer to natural proteins.
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
    """Load jsonl, deduping by (run, step) keeping the latest row."""
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.open()]
    dedup: Dict[tuple, Dict] = {}
    for r in rows:
        dedup[(r.get("run"), r.get("step"))] = r
    return list(dedup.values())


def extract_curve(rows: List[Dict], run_prefix: str, metric: str) -> tuple:
    """Pull (steps, values) for one run family, dropping rows with errors or
    missing metric. Sorted by step ascending."""
    pts = []
    for r in rows:
        run = r.get("run", "")
        if not run.startswith(run_prefix):
            continue
        # ensure it's a step-suffixed label of the family (not e.g. _per_sample)
        if "_step" not in run.split(run_prefix)[-1] and run != run_prefix:
            continue
        if r.get("error", "NONE") != "NONE":
            continue
        v = r.get(metric)
        s = r.get("step")
        if v is None or s is None:
            continue
        pts.append((int(s), float(v)))
    pts.sort()
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def main() -> None:
    fig, axes = plt.subplots(
        nrows=len(DATASETS),
        ncols=len(METRICS),
        figsize=(4.2 * len(METRICS), 3.6 * len(DATASETS)),
        sharex=False,
    )

    for row_i, (ds_name, jsonl_path) in enumerate(DATASETS.items()):
        rows = load_jsonl(jsonl_path)
        for col_i, (metric, title, ylabel, higher_better) in enumerate(METRICS):
            ax = axes[row_i, col_i]
            for prefix, label, color, linestyle in RUN_FAMILIES[ds_name]:
                xs, ys = extract_curve(rows, prefix, metric)
                if not xs:
                    continue
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=4,
                    linewidth=1.4,
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )
            ax.set_xscale("log")
            ax.set_xlabel("Training step")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ds_name} — {title}")
            ax.grid(True, alpha=0.3)
            if not higher_better:
                ax.invert_yaxis()  # lower-is-better metrics flipped for visual consistency
            if col_i == 0:
                ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        "n=256 convergence — generation metrics vs training step\n"
        "Higher is better unless axis inverted; baseline drawn first.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = FIG_OUT / "convergence_speedup.png"
    out_pdf = FIG_OUT / "convergence_speedup.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
