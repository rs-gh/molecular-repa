"""Sampler-ablation convergence plot — proteina n=256.

Sweep at ``results/variance/n256_sampler_ablation/sweep_results.jsonl`` evaluates
each (run, ckpt) under three sampling configurations:
  - ode           (deterministic ODE, vf mode)
  - sde_n0.0      (SC mode, sc_scale_noise=0.0  — denoising-only SDE)
  - sde_n1.0      (SC mode, sc_scale_noise=1.0  — full-temperature SDE)

Layout (2 rows × ncols, one row per trained model):
  Row 1: baseline_256_bs24_2gpu
  Row 2: repa_l9_256_per_residue_bs24_2gpu

Columns mirror plot_convergence_fid.py + plot_convergence_des.py so the same
metric panels are reused: FID / fJSD / fS (PDB+AFDB ref) then Des / Div / Nov /
SS / H/E. Three curves per panel (one per sampler tag); the trained-model
identity is encoded by row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

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


def _style_axes(ax, log_y: bool = False) -> None:
    ax.grid(False)
    ax.grid(True, axis="y", alpha=0.3, which="both" if log_y else "major")
    ax.xaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0, 2.0, 4.0, 7.0), numticks=20)
    )
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=20))
    ax.xaxis.set_major_formatter(FuncFormatter(_humanize))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(_humanize))
    ax.yaxis.set_minor_formatter(NullFormatter())


ROOT = Path(__file__).resolve().parents[2]

# Per-dataset configuration. Selected via --dataset CLI flag (default: pdb).
# Each entry pins the ablation jsonl, the γ=0.45 overlay (multi-rep convergence
# sweep), the two model runs to compare head-to-head, and the fig output dir.
# Color/marker follow the report-wide convention: baseline=blue/square,
# L4=red, L9=green; GearNet=circle, MPNN=triangle.
DATASET_CONFIGS = {
    "pdb": {
        "jsonl": ROOT / "results/variance/n256_sampler_ablation/sweep_results.jsonl",
        "jsonl_paper": ROOT
        / "results/paper/n256_convergence_pdb/sweep_results.clean.jsonl",
        "fig_out": ROOT / "figures/paper/n256_sampler_ablation/pdb",
        "models": [
            ("baseline_256_bs24_2gpu", "Baseline", "tab:blue", "s"),
            (
                "repa_l9_256_per_residue_bs24_2gpu",
                "REPA L9 (GearNet)",
                "tab:green",
                "o",
            ),
        ],
        "label": "PDB",
    },
    "afdb": {
        "jsonl": ROOT
        / "results/variance/n256_afdb_sampler_ablation/sweep_results.jsonl",
        "jsonl_paper": ROOT
        / "results/paper/n256_convergence_afdb/sweep_results.clean.jsonl",
        "fig_out": ROOT / "figures/paper/n256_sampler_ablation/afdb",
        "models": [
            ("baseline_afdb_256", "Baseline (AFDB)", "tab:blue", "s"),
            ("repa_l4_afdb_256", "REPA L4 GearNet (AFDB)", "tab:red", "o"),
        ],
        "label": "AFDB",
    },
}

# Module-level placeholders filled in main() based on CLI dataset choice.
JSONL = None
JSONL_PAPER = None
FIG_OUT = None
MODEL_RUNS = None
DATASET_LABEL = None

# One row per sampler config. Ordered by injected noise, least → most.
SAMPLERS = [
    ("ode", "ODE  (vf, deterministic)"),
    ("sde_n0.0", "SDE  γ=0.0 (denoise-only)"),
    ("sde_n0.35", "SDE  γ=0.35"),
    ("sde_n0.45", "SDE  γ=0.45 (paper default)"),
    ("sde_n0.5", "SDE  γ=0.5"),
    ("sde_n1.0", "SDE  γ=1.0 (full temp.)"),
]

# FID-family columns: (dataset, suffix, title, log_y, higher_better)
# Mirrors plot_convergence_fid.DATASET_METRICS + FS_METRICS.
FID_METRICS = [
    ("PDB", "FID", "PDB FID-50K", True, False),
    ("PDB", "fJSD_C", "PDB fJSD (Class)", True, False),
    ("PDB", "fJSD_A", "PDB fJSD (Arch)", True, False),
    ("PDB", "fJSD_T", "PDB fJSD (Topology)", True, False),
    ("AFDB", "FID", "AFDB FID-50K", True, False),
    ("AFDB", "fJSD_C", "AFDB fJSD (Class)", True, False),
]
FS_METRICS = [
    ("_res_fS_C", "fS (Class)", False, True),
    ("_res_fS_A", "fS (Arch)", False, True),
    ("_res_fS_T", "fS (Topology)", False, True),
]


# Designability-family (only keep those that exist for at least some rows;
# n=250 sweeps include div/nov/ss columns). Same as plot_convergence_des.METRICS.
def _he_ratio(r):
    h = r.get("_res_ss_frac_H_designable") or 0
    e = r.get("_res_ss_frac_E_designable") or 0
    return h / e if e > 0 else None


DES_METRICS = [
    ("_res_designability_rate", "Designability", "%", False, True),
    ("_res_scRMSD_mean", "scRMSD (mean)", "Å", False, False),
    ("_res_scRMSD_median", "scRMSD (median)", "Å", False, False),
    ("_res_plddt_mean", "pLDDT (mean)", "pLDDT", False, True),
    (
        "_res_tm_score_self_mean",
        "Self-consistency TM",
        "mean TM(struct, MPNN→ESMFold)",
        False,
        True,
    ),
    (
        "_res_diversity_pairwise_tm_mean",
        "Diversity (pwTM)",
        "mean pairwise TM (des)",
        False,
        False,
    ),
    (
        "_res_diversity_clusters_total",
        "Diversity (#clusters)",
        "# clusters (des)",
        False,
        True,
    ),
    (
        "_res_novelty_foldseek_pdb_rate",
        "Novelty vs PDB",
        "% novel (TM<0.5)",
        False,
        True,
    ),
    (
        "_res_novelty_foldseek_pdb_max_tm_mean",
        "max-TM vs PDB",
        "mean max-TM (lower=more novel)",
        False,
        False,
    ),
    (
        "_res_novelty_foldseek_afdb_swissprot_rate",
        "Novelty vs AFDB-SP",
        "% novel (TM<0.5)",
        False,
        True,
    ),
    (
        "_res_novelty_foldseek_afdb_swissprot_max_tm_mean",
        "max-TM vs AFDB-SP",
        "mean max-TM (lower=more novel)",
        False,
        False,
    ),
    ("_res_ss_jsd_pdb_designable_2d", "SS 2D-JSD vs PDB", "JSD", False, False),
    ("_res_ss_jsd_afdb_designable_2d", "SS 2D-JSD vs AFDB", "JSD", False, False),
    (("H/E", _he_ratio), "H/E ratio (des)", "H/E (log y)", True, True),
]


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.open()]


def load_paper_overlay(path: Path) -> List[Dict]:
    """Pull γ=0.45 rows from convergence sweep, tag them as a synthetic sampler.

    Same (run-prefix, step) keys as the ablation sweep; differs only in sampler.
    """
    if not path.exists():
        return []
    out = []
    for line in path.open():
        r = json.loads(line)
        if r.get("sc_scale_noise") != 0.45 or r.get("sampling_mode") != "sc":
            continue
        r = dict(r)
        r["sampler_tag"] = "sde_n0.45"
        out.append(r)
    return out


def extract(rows, run_prefix, sampler_tag, metric):
    """Per (run-prefix, sampler_tag, step), aggregate reps → (steps, means,
    mins, maxs). Single-rep configs collapse to mn==mx so the band fill is
    empty and plot_band renders just the line+markers."""
    if isinstance(metric, tuple) and callable(metric[1]):
        fn = metric[1]
    else:
        fn = lambda r, _m=metric: r.get(_m)  # noqa: E731
    by_step: Dict[int, List[float]] = {}
    for r in rows:
        run = r.get("run", "")
        if not run.startswith(run_prefix):
            continue
        # require either exact match or a "_step" suffix, mirroring the
        # convention used by the other convergence scripts
        if run != run_prefix and "_step" not in run.split(run_prefix)[-1]:
            continue
        if r.get("sampler_tag") != sampler_tag:
            continue
        v = fn(r)
        s = r.get("step")
        if v is None or s is None:
            continue
        by_step.setdefault(int(s), []).append(float(v))
    steps = sorted(by_step)
    if not steps:
        return [], [], [], []
    means = [sum(by_step[s]) / len(by_step[s]) for s in steps]
    mins = [min(by_step[s]) for s in steps]
    maxs = [max(by_step[s]) for s in steps]
    return steps, means, mins, maxs


def plot_band(ax, xs, means, mins, maxs, color, marker, label):
    """Mean line + min/max envelope. Mirrors _multi_seed_utils.plot_band:
    multi-rep → faded line + min/max fill; single-rep ≥3 pts → opaque line,
    no fill; <3 pts → markers only."""
    if not xs:
        return
    has_variance = any(mn != mx for mn, mx in zip(mins, maxs))
    too_few_for_trend = len(xs) < 3
    if too_few_for_trend:
        ax.plot(
            xs,
            means,
            marker=marker,
            markersize=9,
            markeredgewidth=1.0,
            markeredgecolor="white",
            linestyle="none",
            color=color,
            label=label,
        )
        if has_variance:
            ax.fill_between(xs, mins, maxs, color=color, alpha=0.18, linewidth=0)
        return
    line_alpha = 0.45 if has_variance else 0.85
    line_width = 1.4 if has_variance else 1.7
    ax.plot(
        xs,
        means,
        linewidth=line_width,
        color=color,
        linestyle="-",
        alpha=line_alpha,
        marker="none",
    )
    ax.fill_between(xs, mins, maxs, color=color, alpha=0.18, linewidth=0)
    ax.plot(
        xs,
        means,
        marker=marker,
        markersize=6,
        markeredgewidth=0.8,
        markeredgecolor="white",
        linestyle="none",
        color=color,
        label=label,
    )


def extract_des_rate(rows, run_prefix, sampler_tag):
    """Per training step, mean designability rate across reps. Used to
    annotate SS panels so the reader can spot small-N noise (esp. baseline at
    γ=1.0). Rate handles the mixed-N case (250 vs 100 samples per rep)."""
    by_step: Dict[int, List[float]] = {}
    for r in rows:
        run = r.get("run", "")
        if not run.startswith(run_prefix):
            continue
        if run != run_prefix and "_step" not in run.split(run_prefix)[-1]:
            continue
        if r.get("sampler_tag") != sampler_tag:
            continue
        rate = r.get("_res_designability_rate")
        s = r.get("step")
        if rate is None or s is None:
            continue
        by_step.setdefault(int(s), []).append(float(rate))
    return {s: sum(v) / len(v) for s, v in by_step.items()}


def _plot_figure(rows, columns, suptitle, out_name, annotate_n=False):
    ncols = len(columns)
    fig, axes = plt.subplots(
        nrows=len(SAMPLERS),
        ncols=ncols,
        figsize=(3.8 * ncols, 3.2 * len(SAMPLERS)),
        sharex=False,
        squeeze=False,
    )
    for row_i, (sampler_tag, sampler_label) in enumerate(SAMPLERS):
        for col_i, (metric, title, log_y, hib) in enumerate(columns):
            ax = axes[row_i, col_i]
            for run_prefix, model_label, color, marker in MODEL_RUNS:
                xs, means, mins, maxs = extract(rows, run_prefix, sampler_tag, metric)
                if not xs:
                    continue
                # Rate columns are 0–1 fractions → render as 0–100 %.
                if isinstance(metric, str) and metric.endswith("_rate"):
                    means = [v * 100.0 for v in means]
                    mins = [v * 100.0 for v in mins]
                    maxs = [v * 100.0 for v in maxs]
                plot_band(ax, xs, means, mins, maxs, color, marker, model_label)
                if annotate_n:
                    rate_by_step = extract_des_rate(rows, run_prefix, sampler_tag)
                    for x, y in zip(xs, means):
                        rate = rate_by_step.get(x)
                        if rate is None:
                            continue
                        ax.annotate(
                            f"{rate * 100:.1f}%",
                            (x, y),
                            xytext=(4, 4),
                            textcoords="offset points",
                            fontsize=6,
                            color=color,
                            alpha=0.85,
                        )
            ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")
            if row_i == len(SAMPLERS) - 1:
                ax.set_xlabel("Training step")
            arrow = "" if "H/E" in title else (" ↑" if hib else " ↓")
            if row_i == 0:
                ax.set_title(f"{title}{arrow}", fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f"{sampler_label}", fontsize=10)
            if "H/E" in title or not hib:
                ax.invert_yaxis()  # up = better
            _style_axes(ax, log_y=log_y)
    fig.suptitle(suptitle, fontsize=12)
    handles, labels = [], []
    seen = set()
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab in seen:
                continue
            seen.add(lab)
            handles.append(h)
            labels.append(lab)
    if labels:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 6),
            bbox_to_anchor=(0.5, -0.01),
            fontsize=9,
            frameon=False,
        )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_png = FIG_OUT / out_name
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS), default="pdb")
    args = p.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    global JSONL, JSONL_PAPER, FIG_OUT, MODEL_RUNS, DATASET_LABEL
    JSONL = cfg["jsonl"]
    JSONL_PAPER = cfg["jsonl_paper"]
    FIG_OUT = cfg["fig_out"]
    MODEL_RUNS = cfg["models"]
    DATASET_LABEL = cfg["label"]
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(JSONL) + load_paper_overlay(JSONL_PAPER)

    fid_cols = [
        (f"_res_{ds}_{suffix}", title, log_y, hib)
        for ds, suffix, title, log_y, hib in FID_METRICS
    ]
    fid_cols += [(col, title, log_y, hib) for col, title, log_y, hib in FS_METRICS]

    all_des_cols = [
        (metric, title, log_y, hib) for metric, title, _yl, log_y, hib in DES_METRICS
    ]
    # Split into two figures so per-panel width stays readable.
    des_quality_titles = {
        "Designability",
        "scRMSD (mean)",
        "scRMSD (median)",
        "pLDDT (mean)",
        "Self-consistency TM",
        "Diversity (pwTM)",
        "Diversity (#clusters)",
    }
    des_quality = [c for c in all_des_cols if c[1] in des_quality_titles]
    des_dist = [c for c in all_des_cols if c[1] not in des_quality_titles]

    _plot_figure(
        rows,
        fid_cols,
        suptitle=(
            f"n=256 {DATASET_LABEL} sampler ablation — FID-family distributional metrics across training steps\n"
            "Rows = sampler config. Baseline (blue) vs REPA (red/green) head-to-head per regime. γ=0.45 plotted as mean+min/max envelope; other γ are single seed 42."
        ),
        out_name="sampler_ablation_fid.png",
    )
    _plot_figure(
        rows,
        des_quality,
        suptitle=(
            f"n=256 {DATASET_LABEL} sampler ablation — designability / sample-quality / diversity (designable subset)\n"
            "Rows = sampler config. Baseline (blue) vs REPA (red/green) head-to-head per regime. γ=0.45 plotted as mean+min/max envelope; other γ are single seed 42."
        ),
        out_name="sampler_ablation_des_quality.png",
    )
    _plot_figure(
        rows,
        des_dist,
        suptitle=(
            f"n=256 {DATASET_LABEL} sampler ablation — novelty / SS-distribution match (designable subset)\n"
            "Rows = sampler config. Baseline (blue) vs REPA (red/green) head-to-head per regime. γ=0.45 plotted as mean+min/max envelope; other γ are single seed 42."
        ),
        out_name="sampler_ablation_des_dist.png",
    )

    # Focused secondary-structure view: H frac, E frac, H/E ratio, SS-JSD vs PDB
    # and AFDB. Same head-to-head layout, narrower so SS trends are easier to
    # read across sampler regimes.
    ss_cols = [
        ("_res_ss_frac_H_designable", "Helix fraction (des)", False, True),
        ("_res_ss_frac_E_designable", "Sheet fraction (des)", False, True),
        (("H/E", _he_ratio), "H/E ratio (des)", True, True),
        ("_res_ss_jsd_pdb_designable_2d", "SS 2D-JSD vs PDB", False, False),
        ("_res_ss_jsd_afdb_designable_2d", "SS 2D-JSD vs AFDB", False, False),
    ]
    _plot_figure(
        rows,
        ss_cols,
        suptitle=(
            f"n=256 {DATASET_LABEL} sampler ablation — secondary-structure focus (designable subset)\n"
            "Does REPA shift SS diversity (H frac, E frac, H/E, SS-2D-JSD) under different sampling regimes? Rows = sampler. γ=0.45 plotted as mean+min/max over 3 reps (seeds 42/1042/2042); other γ are single seed 42. Inline % is the designability rate at that ckpt (small % → SS computed over few samples → noisy)."
        ),
        out_name="sampler_ablation_ss.png",
        annotate_n=True,
    )


if __name__ == "__main__":
    main()
