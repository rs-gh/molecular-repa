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
from matplotlib.ticker import FuncFormatter, LogLocator

import sys

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[5]))  # repo root
from evaluation.proteina.lib import pretrained_overlay  # noqa: E402

# Overlay pretrained Proteina (NGC 60M, ~1.3M steps) as a horizontal dashed
# reference line. Not step-comparable to our runs (different schedule/data) —
# flip to False to hide.
SHOW_PRETRAINED = True


def _humanize(v, _pos=None):
    """Format a number as 50K / 1M / 0.42 — readable for both step counts and metric values."""
    av = abs(v)
    if av >= 1e6:
        return f"{v / 1e6:g}M"
    if av >= 1e3:
        return f"{v / 1e3:g}K"
    if av >= 1:
        return f"{v:g}"
    return f"{v:.3g}"


def _style_axes(ax, log_y: bool = False) -> None:
    """Horizontal-only grid, human-friendly tick labels, denser x ticks on log axis."""
    ax.grid(False)
    ax.grid(True, axis="y", alpha=0.3, which="both" if log_y else "major")
    # Denser major ticks on the log x-axis: powers of 10 plus 2/4/7 within each decade
    # (matches the sweep step grid: 100k/200k/400k/700k/1M/1.3M/1.6M).
    ax.xaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0, 2.0, 4.0, 7.0), numticks=20)
    )
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=20))
    ax.xaxis.set_major_formatter(FuncFormatter(_humanize))
    ax.yaxis.set_major_formatter(FuncFormatter(_humanize))


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
# Visual encoding:
#   color    → REPA layer depth (baseline=blue, L4=red, L9=orange)
#   linestyle→ same as color group (kept for B/W readability)
#   marker   → encoder family (baseline=square, GearNet=circle, MPNN=triangle)
RUN_FAMILIES = {
    "PDB": [
        ("baseline_256_bs24_2gpu", "Baseline (PDB)", "tab:blue", "-", "s"),
        (
            "repa_l4_256_per_residue_bs24_2gpu",
            "REPA L4 GearNet (PDB)",
            "tab:red",
            "-",
            "o",
        ),
        (
            "repa_l9_256_per_residue_bs24_2gpu",
            "REPA L9 GearNet (PDB)",
            "tab:orange",
            "-",
            "o",
        ),
        ("repa_mpnn_l4_256_per_residue", "REPA L4 MPNN (PDB)", "tab:red", "--", "^"),
        ("repa_mpnn_l9_256_per_residue", "REPA L9 MPNN (PDB)", "tab:orange", "--", "^"),
        # random-init GearNet control: gray dotted diamond — same depth as L4 GearNet
        # but encoder weights are random; should regress toward baseline
        (
            "repa_l4_256_per_residue_random_bs24_2gpu",
            "REPA L4 GearNet-rand (PDB, ctrl)",
            "tab:gray",
            ":",
            "D",
        ),
    ],
    "AFDB": [
        ("baseline_afdb_256", "Baseline (AFDB)", "tab:blue", "-", "s"),
        ("repa_l4_afdb_256", "REPA L4 GearNet (AFDB)", "tab:red", "-", "o"),
        ("repa_l9_afdb_256", "REPA L9 GearNet (AFDB, partial)", "tab:orange", ":", "o"),
        ("repa_mpnn_l4_afdb_256", "REPA L4 MPNN (AFDB)", "tab:red", "--", "^"),
        ("repa_mpnn_l9_afdb_256", "REPA L9 MPNN (AFDB)", "tab:orange", "--", "^"),
    ],
}

# Metric column → (panel title, y-axis label, higher_is_better)
METRICS = [
    ("_res_designability_rate", "Designability", "rate", True),
    # Raw cluster count (sum across length bins, designable subset). Note:
    # the rate clusters/n_designable saturates at 1.0 for small n_designable
    # (each rare designable structure trivially gets its own fold), so we
    # stick with raw counts here despite the designability confound.
    (
        "_res_diversity_clusters_total",
        "Diversity (clusters)",
        "# clusters (designable subset)",
        True,
    ),
    # Continuous diversity: mean pairwise TM within designable pool. Doesn't
    # saturate at small n the way clusters/n does. Lower = more diverse.
    (
        "_res_diversity_pairwise_tm_mean",
        "Diversity (pairwise TM)",
        "mean pairwise TM (des)",
        False,
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
    # Derived metric: helix/sheet ratio (designable subset). Computed as
    # ss_frac_H / ss_frac_E with a small guard against zero-sheet rows.
    # Passing a callable instead of a column name routes through the same
    # extract_curve plumbing; "higher_better" is False but H/E has no natural
    # "good" direction — we just leave the axis upright so 1.0 stays readable.
    (
        (
            "H/E",
            lambda r: (r.get("_res_ss_frac_H_designable") or 0) / e
            if (e := r.get("_res_ss_frac_E_designable") or 0) > 0
            else None,
        ),
        "H/E ratio (des)",
        "H/E",
        True,
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


def extract_curve(rows: List[Dict], run_prefix: str, metric) -> tuple:
    """Pull (steps, values) for one run family, dropping rows with errors or
    missing metric. Sorted by step ascending.

    ``metric`` may be a column name (str) for direct lookup, or a tuple
    ``(label, fn)`` where fn(row) computes a derived value (e.g. H/E ratio)."""
    if isinstance(metric, tuple):
        _, fn = metric
    else:
        fn = lambda r, _m=metric: r.get(_m)  # noqa: E731
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
        v = fn(r)
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
            for prefix, label, color, linestyle, marker in RUN_FAMILIES[ds_name]:
                xs, ys = extract_curve(rows, prefix, metric)
                if not xs:
                    continue
                # connecting line at reduced opacity so dense regions read as
                # markers, not as a tangle of overlapping strokes
                ax.plot(
                    xs,
                    ys,
                    linewidth=1.4,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.35,
                    marker="none",
                )
                # markers at full opacity carry the legend entry
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    markersize=6,
                    markeredgewidth=0.8,
                    markeredgecolor="white",
                    linestyle="none",
                    color=color,
                    label=label,
                )
            ax.set_xscale("log")
            ax.set_xlabel("Training step")
            ax.set_ylabel(ylabel)
            # H/E ratio has no natural "better" direction — skip arrow.
            if "H/E" in title:
                arrow = ""
            else:
                arrow = " ↑" if higher_better else " ↓"
            ax.set_title(f"{ds_name} — {title}{arrow}")
            _style_axes(ax)
            # Optional pretrained Proteina horizontal reference line. Only
            # plotted for metrics it has data for (FID, designability).
            if SHOW_PRETRAINED and isinstance(metric, str):
                pre_val = pretrained_overlay.load_gen().get(metric)
                if pre_val is not None:
                    ax.axhline(
                        pre_val,
                        color=pretrained_overlay.PRETRAINED_COLOR,
                        linestyle="--",
                        linewidth=1.4,
                        alpha=0.85,
                        label=pretrained_overlay.PRETRAINED_LABEL
                        if col_i == 0
                        else None,
                        zorder=1,
                    )
            if col_i == 0:
                ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        "n=256 convergence — generation metrics vs training step\n"
        "Direction (higher vs lower better) varies by panel — see ylabel; baseline drawn first.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = FIG_OUT / "convergence_speedup.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
