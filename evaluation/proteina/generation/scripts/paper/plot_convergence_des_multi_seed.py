"""Multi-seed designability/diversity/novelty/SS convergence plot — PDB + AFDB.

Multi-seed companion to `plot_convergence_des.py` / `plot_convergence_des_n128.py`.
Reads `sweep_results.clean.jsonl`, shows mean + min/max envelope across reps.
Single-rep legacy rows appear as bare markers without a shaded band.

See `plot_convergence_fid_multi_seed.py` for the rep-coverage notes.
"""

from __future__ import annotations

from pathlib import Path
import sys

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[5]))

from evaluation.proteina.generation.scripts.paper._multi_seed_utils import (  # noqa: E402
    plt,
    style_axes,
    load_jsonl_with_reps,
    extract_bands,
    plot_band,
    RATE_KEYS,
    _rate_to_pct,
)
from evaluation.proteina.generation.scripts.paper.plot_convergence_fid_multi_seed import (  # noqa: E402
    RUN_FAMILIES_N128,
    RUN_FAMILIES_N256,
)
from evaluation.proteina.lib import pretrained_overlay  # noqa: E402

SHOW_PRETRAINED = True

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/paper"

CONFIGS = [
    (
        128,
        "n128_convergence/multi_seed",
        [
            ("PDB", "n128_convergence_pdb", RUN_FAMILIES_N128["PDB"]),
            ("AFDB", "n128_convergence_afdb", RUN_FAMILIES_N128["AFDB"]),
        ],
    ),
    (
        256,
        "n256_convergence/multi_seed",
        [
            ("PDB", "n256_convergence_pdb", RUN_FAMILIES_N256["PDB"]),
            ("AFDB", "n256_convergence_afdb", RUN_FAMILIES_N256["AFDB"]),
        ],
    ),
]

METRICS = [
    ("_res_designability_rate", "Designability", "%", True),
    ("_res_diversity_clusters_total", "Diversity (clusters)", "# clusters (des)", True),
    (
        "_res_diversity_pairwise_tm_mean",
        "Diversity (pairwise TM)",
        "mean pairwise TM (des)",
        False,
    ),
    (
        "_res_novelty_foldseek_pdb_rate",
        "Novelty vs PDB",
        "% novel (TM<0.5)",
        True,
    ),
    (
        "_res_novelty_foldseek_afdb_swissprot_rate",
        "Novelty vs AFDB-SP",
        "% novel (TM<0.5)",
        True,
    ),
    ("_res_ss_jsd_pdb_designable_2d", "SS 2D-JSD vs PDB (des)", "JSD", False),
    ("_res_ss_jsd_afdb_designable_2d", "SS 2D-JSD vs AFDB (des)", "JSD", False),
    (
        (
            "H/E",
            lambda r: (r.get("_res_ss_frac_H_designable") or 0) / e
            if (e := r.get("_res_ss_frac_E_designable") or 0) > 0
            else None,
        ),
        # Log y: at early steps E≈0 (pure-helix warmup) makes the ratio span
        # ~0.3-250; linear y would crush the post-warmup regime into a sliver.
        "H/E ratio (des)",
        "H/E (log y)",
        True,
    ),
    # Decomposed view of the same SS story: H and E plotted separately so the
    # near-zero-E warmup that dominates the ratio panel doesn't hide them.
    ("_res_ss_frac_H_designable", "Helix fraction (des)", "frac H", True),
    ("_res_ss_frac_E_designable", "Sheet fraction (des)", "frac E", True),
]
_LOG_Y_TITLES = {"H/E ratio (des)"}


def plot_one(n, fig_subdir, datasets):
    fig_out = ROOT / "figures/paper" / fig_subdir
    fig_out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=len(METRICS),
        figsize=(3.8 * len(METRICS), 3.6 * len(datasets)),
        sharex=False,
        squeeze=False,
    )

    for row_i, (ds_name, results_subdir, families) in enumerate(datasets):
        jsonl = RESULTS / results_subdir / "sweep_results.clean.jsonl"
        rows = load_jsonl_with_reps(jsonl)
        if not rows:
            for col_i in range(len(METRICS)):
                axes[row_i, col_i].set_visible(False)
            continue

        for col_i, (metric, title, ylabel, higher) in enumerate(METRICS):
            ax = axes[row_i, col_i]
            for prefix, label, color, ls, marker in families:
                xs, mu, lo, hi, _ = extract_bands(rows, prefix, metric)
                mu = _rate_to_pct(mu, metric)
                lo = _rate_to_pct(lo, metric)
                hi = _rate_to_pct(hi, metric)
                plot_band(ax, xs, mu, lo, hi, color, ls, marker, label)
            ax.set_xscale("log")
            log_y = title in _LOG_Y_TITLES
            if log_y:
                ax.set_yscale("log")
            ax.set_xlabel("Training step")
            ax.set_ylabel(ylabel)
            arrow = "" if "H/E" in title else (" ↑" if higher else " ↓")
            ax.set_title(f"{ds_name} — {title}{arrow}")
            if "H/E" in title or not higher:
                ax.invert_yaxis()  # up = better
            style_axes(ax, log_y=log_y)
            if SHOW_PRETRAINED and isinstance(metric, str):
                pre_val = pretrained_overlay.load_gen().get(metric)
                if pre_val is not None:
                    if metric in RATE_KEYS:
                        pre_val = pre_val * 100.0
                    ax.axhline(
                        pre_val,
                        color=pretrained_overlay.PRETRAINED_COLOR,
                        linestyle="--",
                        linewidth=2.6,
                        alpha=0.9,
                        label=pretrained_overlay.PRETRAINED_LABEL
                        if (row_i == 0 and col_i == 0)
                        else None,
                        zorder=1,
                    )

    fig.suptitle(
        f"n={n} multi-seed convergence — designability / diversity / novelty / SS\n"
        "Line = mean across reps; band = min/max envelope. "
        "n=5 (seeds 42/1042/2042/3042/4042): n=128 PDB baseline + mpnn-L9. "
        "n=3 (seeds 42/1042/2042): all other multi-rep families. "
        "Single-rep legacy points show as bare markers.",
        fontsize=12,
    )
    handles, labels = [], []
    seen = set()
    for ax in axes.flat:
        if not ax.get_visible():
            continue
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab in seen:
                continue
            seen.add(lab)
            handles.append(h)
            labels.append(lab)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        bbox_to_anchor=(0.5, -0.01),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out_png = fig_out / "convergence_des.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main():
    for n, fig_subdir, datasets in CONFIGS:
        plot_one(n, fig_subdir, datasets)


if __name__ == "__main__":
    main()
