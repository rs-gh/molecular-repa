"""Training-dynamics convergence plots for n=256 ablation blocks.

Sister script to plot_n128_training_dynamics.py. Same convention: 2x4
metric grid per block, x = training step (log), up = better (lower-is-better
y-axes inverted), humanized tick labels, multi-step legs render as
line+markers, single-step legs render as markers only.

Output:
    figures/paper/n256_training_dynamics_ablations/{block}.png

Block list reflects the n=256 ablation structure documented in
`results/paper/ablation_blocks.yaml`: encoder + layer are combined, and
PDB / AFDB are separate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[2]
RESULTS = ROOT / "results" / "paper"
FIG_OUT = ROOT / "figures" / "paper" / "n256_training_dynamics_ablations"
FIG_OUT.mkdir(parents=True, exist_ok=True)


ABLATIONS: Dict[str, Dict] = {
    "pdb_bs": {
        "title": "n=256 PDB — Batch size ablation (REPA L4 GearNet, bs ∈ {24, 80} × 2gpu)",
        "runs": [
            ("baseline_256_bs24_2gpu", "Baseline (bs24)", "tab:blue", "-", "s"),
            ("repa_l4_256_per_residue_bs24_2gpu", "L4 bs24", "#E74C3C", "-", "o"),
            ("repa_l4_256_per_residue_bs80_2gpu", "L4 bs80", "#8B0000", "-", "o"),
        ],
    },
    "pdb_lambda": {
        "title": "n=256 PDB — λ ablation (REPA L4 GearNet, bs=24, λ ∈ {0.25, 0.5, 1.0, 2.0})",
        "runs": [
            ("baseline_256_bs24_2gpu", "Baseline", "tab:blue", "-", "s"),
            ("repa_l4_256_per_residue_lambda025", "λ=0.25", "#FBB4AE", "-", "o"),
            ("repa_l4_256_per_residue_bs24_2gpu", "λ=0.5 (ref)", "#E74C3C", "-", "o"),
            ("repa_l4_256_per_residue_lambda1", "λ=1.0", "#C44893", "-", "o"),
            ("repa_l4_256_per_residue_lambda2", "λ=2.0", "#8B0000", "-", "o"),
        ],
    },
    "pdb_averaging": {
        "title": "n=256 PDB — Averaging ablation (per_residue vs per_sample, L0/L4/L9)",
        "runs": [
            ("baseline_256_bs24_2gpu", "Baseline", "tab:blue", "-", "s"),
            ("repa_l0_256_ep26", "L0 per_residue", "#55A868", "-", "o"),
            ("repa_l0_256_per_sample", "L0 per_sample", "#55A868", "--", "^"),
            ("repa_l4_256_ep22", "L4 per_residue", "#E74C3C", "-", "o"),
            ("repa_l4_256_per_sample", "L4 per_sample", "#E74C3C", "--", "^"),
            ("repa_l9_256_ep25", "L9 per_residue", "#F39C12", "-", "o"),
            ("repa_l9_256_per_sample", "L9 per_sample", "#F39C12", "--", "^"),
        ],
    },
}

# Effective batch size per run prefix. Used to convert training step → samples
# seen for the bs ablation block (see plot_n128_training_dynamics.py for
# rationale). n=256 bs ablation: bs24_2gpu = 24, bs80_2gpu = 80.
EFF_BS: Dict[str, int] = {
    "baseline_256_bs24_2gpu": 24,
    "repa_l4_256_per_residue_bs24_2gpu": 24,
    "repa_l4_256_per_residue_bs80_2gpu": 80,
}

METRICS: List[Tuple[str, str, bool]] = [
    ("_res_PDB_FID", "PDB FID-1.1K ↓", True),
    ("_res_PDB_fJSD_T", "PDB fJSD T ↓", True),
    ("_res_AFDB_FID", "AFDB FID-1.1K ↓", True),
    ("_res_AFDB_fJSD_T", "AFDB fJSD T ↓", True),
    ("_res_designability_rate", "Designability (%) ↑", False),
    ("_res_scRMSD_mean", "scRMSD mean (Å) ↓", True),
    ("_res_diversity_clusters_total", "Diversity (clusters) ↑", False),
    ("_res_novelty_foldseek_pdb_max_tm_mean", "Foldseek max-TM PDB ↓", True),
]


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
    ax.grid(True, axis="y", alpha=0.3, which="both" if log_y else "major")
    ax.xaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0, 2.0, 4.0, 7.0), numticks=20)
    )
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(_humanize))
    if log_y:
        ax.yaxis.set_major_locator(
            LogLocator(base=10.0, subs=(1.0, 2.0, 4.0, 7.0), numticks=20)
        )
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=20))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(_humanize))


def load_all_rows() -> List[Dict]:
    """Union all n256 paper + convergence jsonls, dedup by (run, step)."""
    sources = list(RESULTS.glob("n256_paper_*/sweep_results.jsonl")) + list(
        RESULTS.glob("n256_convergence_*/sweep_results.jsonl")
    )
    dedup: Dict[Tuple[str, int], Dict] = {}
    for path in sources:
        if not path.exists():
            continue
        for line in path.open():
            r = json.loads(line)
            if r.get("error", "NONE") != "NONE":
                continue
            key = (r.get("run"), r.get("step"))
            if key[0] is None or key[1] is None:
                continue
            dedup.setdefault(key, r)
    return list(dedup.values())


def extract(
    rows: List[Dict], run_prefix: str, col: str
) -> Tuple[List[int], List[float]]:
    """Return sorted (xs, ys) for every row whose run starts with prefix."""
    pts = []
    for r in rows:
        run = r.get("run", "")
        if run == run_prefix:
            ok = True
        elif run.startswith(run_prefix + "_step"):
            ok = True
        else:
            ok = False
        if not ok:
            continue
        v = r.get(col)
        s = r.get("step")
        if v is None or s is None:
            continue
        if isinstance(v, float) and v != v:
            continue
        pts.append((int(s), float(v)))
    pts.sort()
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def plot_curve(ax, xs, ys, color, ls, marker, label):
    if len(xs) >= 2:
        ax.plot(xs, ys, linewidth=1.2, color=color, linestyle=ls, alpha=0.45)
    ax.plot(
        xs,
        ys,
        marker=marker,
        markersize=7,
        markeredgewidth=0.8,
        markeredgecolor="white",
        linestyle="none",
        color=color,
        label=label,
    )


def plot_block(block_id: str, rows: List[Dict], x_mode: str = "step") -> Path:
    """Render one ablation block. x_mode in {"step", "samples"}."""
    cfg = ABLATIONS[block_id]
    n_cols = 4
    n_rows = (len(METRICS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.0 * n_cols, 3.4 * n_rows), sharex=False
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (col, title, lower_better) in enumerate(METRICS):
        ax = axes_flat[i]
        anything = False
        for prefix, label, color, ls, marker in cfg["runs"]:
            xs, ys = extract(rows, prefix, col)
            if not xs:
                continue
            if x_mode == "samples":
                bs = EFF_BS.get(prefix)
                if bs is None:
                    continue
                xs = [s * bs for s in xs]
            # Rate columns are 0–1 fractions → render as 0–100 %.
            if col.endswith("_rate"):
                ys = [y * 100.0 for y in ys]
            plot_curve(ax, xs, ys, color, ls, marker, label)
            anything = True
        ax.set_xscale("log")
        if "FID" in title or "fJSD" in title or "scRMSD" in title:
            ax.set_yscale("log")
            _style_axes(ax, log_y=True)
        else:
            _style_axes(ax, log_y=False)
        if lower_better:
            ax.invert_yaxis()
        ax.set_xlabel("Samples seen" if x_mode == "samples" else "Training step")
        ax.set_title(title)
        if not anything:
            ax.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="gray",
                fontsize=11,
            )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if not handles:
        for ax in axes_flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 6),
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
            fontsize=10,
        )

    suffix = " (vs samples seen)" if x_mode == "samples" else ""
    fig.suptitle(cfg["title"] + suffix, fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    fname = f"{block_id}_by_samples.png" if x_mode == "samples" else f"{block_id}.png"
    out = FIG_OUT / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


SAMPLES_BLOCKS = {"pdb_bs"}


def main() -> None:
    rows = load_all_rows()
    print(
        f"Loaded {len(rows)} unique (run, step) rows across n256 paper + convergence dirs."
    )
    for block_id in ABLATIONS:
        out = plot_block(block_id, rows, x_mode="step")
        print(f"Wrote {out}")
        if block_id in SAMPLES_BLOCKS:
            out = plot_block(block_id, rows, x_mode="samples")
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
