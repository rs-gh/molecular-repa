"""Shared rendering machinery for n=128 / n=256 h2h convergence plots.

Both ``plot_h2h_n128.py`` and ``plot_h2h_n256.py`` define a ``COMPARISONS`` list
of (dataset, prefix-tuples) and delegate the actual plotting to ``run_all``
below. Keep visual conventions / metric tables in one place.
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
from evaluation.proteina.lib import pretrained_overlay  # noqa: E402


SHOW_PRETRAINED = True


# Visual language (kept consistent across every panel):
#   colour = alignment layer  → baseline=blue, L4=red, L9=green
#   marker = encoder          → GearNet=circle, MPNN=triangle (baseline=square)
#   linestyle = solid everywhere; markers carry the encoder distinction.

DES_METRICS = [
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
    # H/E ratio on log y: at step 100k E≈0 (pure-helix warmup), so the ratio
    # spans ~0.3-250 across training; linear y would crush the post-warmup
    # regime into a sliver.
    (
        (
            "H/E",
            lambda r: (r.get("_res_ss_frac_H_designable") or 0) / e
            if (e := r.get("_res_ss_frac_E_designable") or 0) > 0
            else None,
        ),
        "H/E ratio (des)",
        "H/E (log y)",
        True,
    ),
    ("_res_ss_frac_H_designable", "Helix fraction (des)", "frac H", True),
    ("_res_ss_frac_E_designable", "Sheet fraction (des)", "frac E", True),
]
_LOG_Y_DES = {"H/E ratio (des)"}

# FID-N labels are filled in per-call via ``_fid_label`` so they reflect the
# actual sample budget (n=128 → 500 PDBs, n=256 → 1125 PDBs), not the
# image-generation FID-50K convention.
FID_DATASET_METRICS = [
    ("FID", "FID"),
    ("fJSD_A", "fJSD (Architecture)"),
    ("fJSD_C", "fJSD (Class)"),
    ("fJSD_T", "fJSD (Topology)"),
]


def _fid_label(n_label: str) -> str:
    """Map figure n_label (e.g. ``"n=128"``, ``"n=256"``) → FID-<samples>."""
    if "128" in n_label:
        return "FID-500"
    if "256" in n_label:
        return "FID-1.1K"
    return "FID"


FS_METRICS = [
    ("_res_fS_C", "fS (Class)"),
    ("_res_fS_A", "fS (Architecture)"),
    ("_res_fS_T", "fS (Topology)"),
]

QUALITY_METRICS = [
    ("_res_plddt_mean", "pLDDT (mean)", "pLDDT", True),
    ("_res_scRMSD_mean", "scRMSD (mean)", "Å", False),
    (
        "_res_novelty_foldseek_pdb_max_tm_mean",
        "Novelty vs PDB (max-TM)",
        "mean max TM",
        False,
    ),
    (
        "_res_novelty_foldseek_afdb_swissprot_max_tm_mean",
        "Novelty vs AFDB-SP (max-TM)",
        "mean max TM",
        False,
    ),
    ("_res_tm_score_self_mean", "TM-self (mean)", "mean pairwise TM", False),
]


def _figure_legend(fig, axes):
    handles, labels = [], []
    seen = set()
    for ax in axes:
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
        ncol=min(len(labels), 4),
        bbox_to_anchor=(0.5, -0.01),
        fontsize=9,
        frameon=False,
    )


def _overlay_pretrained(ax, metric_key, *, first_col, n_label):
    if not (SHOW_PRETRAINED and isinstance(metric_key, str)):
        return
    # n128 plots ignored the n-key (used the default); n256 callers pass "n256"
    # to pick up the n=256 pretrained baseline numbers.
    try:
        pre = (
            pretrained_overlay.load_gen(n_label)
            if n_label
            else pretrained_overlay.load_gen()
        )
    except TypeError:
        pre = pretrained_overlay.load_gen()
    pre_val = pre.get(metric_key)
    if pre_val is None:
        return
    if metric_key in RATE_KEYS:
        pre_val = pre_val * 100.0
    ax.axhline(
        pre_val,
        color=pretrained_overlay.PRETRAINED_COLOR,
        linestyle="--",
        linewidth=2.6,
        alpha=0.9,
        label=pretrained_overlay.PRETRAINED_LABEL if first_col else None,
        zorder=1,
    )


def _plot_des(rows, comp, fig_out, *, n_label, pretrained_key):
    ds = comp["dataset"]
    families = comp["families"]
    n_cols = len(DES_METRICS)
    fig, axes = plt.subplots(
        nrows=1, ncols=n_cols, figsize=(3.8 * n_cols, 3.8), sharex=False
    )
    for col_i, (metric, title, ylabel, higher) in enumerate(DES_METRICS):
        ax = axes[col_i]
        for prefix, label, color, ls, marker in families:
            xs, mu, lo, hi, _ = extract_bands(rows, prefix, metric)
            mu = _rate_to_pct(mu, metric)
            lo = _rate_to_pct(lo, metric)
            hi = _rate_to_pct(hi, metric)
            plot_band(ax, xs, mu, lo, hi, color, ls, marker, label)
        ax.set_xscale("log")
        log_y = title in _LOG_Y_DES
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        arrow = "" if "H/E" in title else (" ↑" if higher else " ↓")
        ax.set_title(f"{ds} — {title}{arrow}")
        is_pct = isinstance(metric, str) and metric in RATE_KEYS
        style_axes(ax, log_y=log_y, percent=is_pct)
        _overlay_pretrained(ax, metric, first_col=(col_i == 0), n_label=pretrained_key)
        # Up-is-better axis flip: invert y for lower-is-better metrics; H/E
        # ratio also inverted so "closer to 1" sits nearer the top.
        if not higher or "H/E" in title:
            ax.invert_yaxis()
    fig.suptitle(
        f"{n_label} {ds} head-to-head — {comp['title']} (designability / diversity / novelty / SS)\n"
        f"Line = mean across reps; band = min/max envelope. {comp['reps_note']}.",
        fontsize=12,
    )
    _figure_legend(fig, axes)
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    out_png = fig_out / "h2h_des.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _plot_fid(rows, comp, fig_out, *, n_label, pretrained_key):
    ds = comp["dataset"]
    fid_suffix = comp["fid_suffix"]
    # Cross-dataset reference: every run computes FID/fJSD against BOTH the
    # PDB and AFDB GearNet feature banks ([inference_fid_60m_baseline.yaml]).
    # ``fid_suffix`` selects the in-distribution side; ``cross_suffix`` is the
    # other one. Labelling distinguishes "vs PDB ref" / "vs AFDB ref" so it's
    # clear which reference each panel is computed against, independent of the
    # training dataset.
    cross_suffix = "AFDB" if fid_suffix == "PDB" else "PDB"
    families = comp["families"]
    n_cols = max(len(FID_DATASET_METRICS), len(FS_METRICS))
    fig, axes = plt.subplots(
        nrows=3, ncols=n_cols, figsize=(3.8 * n_cols, 3.6 * 3), sharex=False
    )

    fid_label = _fid_label(n_label)

    def _render_fid_row(row_axes, suffix):
        for col_i, (metric_suffix, title) in enumerate(FID_DATASET_METRICS):
            ax = row_axes[col_i]
            col = f"_res_{suffix}_{metric_suffix}"
            for prefix, label, color, ls, marker in families:
                xs, mu, lo, hi, _ = extract_bands(rows, prefix, col)
                plot_band(ax, xs, mu, lo, hi, color, ls, marker, label)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Training step")
            ax.set_ylabel("value (lower = closer, log y)")
            display_title = fid_label if title == "FID" else title
            ax.set_title(f"vs {suffix} — {display_title} ↓")
            style_axes(ax, log_y=True)
            _overlay_pretrained(ax, col, first_col=(col_i == 0), n_label=pretrained_key)
            ax.invert_yaxis()
        for j in range(len(FID_DATASET_METRICS), len(row_axes)):
            row_axes[j].set_visible(False)

    _render_fid_row(axes[0], fid_suffix)
    _render_fid_row(axes[1], cross_suffix)

    # Row 3: fS metrics (encoder-agnostic, no dataset suffix).
    for j, (col, title) in enumerate(FS_METRICS):
        ax = axes[2][j]
        for prefix, label, color, ls, marker in families:
            xs, mu, lo, hi, _ = extract_bands(rows, prefix, col)
            plot_band(ax, xs, mu, lo, hi, color, ls, marker, label)
        ax.set_xscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel("entropy (higher = more coverage)")
        ax.set_title(f"{title} ↑")
        style_axes(ax)
        _overlay_pretrained(ax, col, first_col=(j == 0), n_label=pretrained_key)
    for j in range(len(FS_METRICS), n_cols):
        axes[2][j].set_visible(False)

    fig.suptitle(
        f"{n_label} {ds}-trained head-to-head — {comp['title']} (FID / fJSD / fS)\n"
        f"Row 1: vs {fid_suffix} reference (in-distribution). "
        f"Row 2: vs {cross_suffix} reference (cross-dataset generalization). "
        f"Row 3: fS coverage. "
        f"Line = mean across reps; band = min/max envelope. {comp['reps_note']}. "
        "FID/fJSD: lower = closer (log y, axis inverted ⇒ up = better). fS: linear, higher = more coverage.",
        fontsize=12,
    )
    _figure_legend(fig, axes.flatten())
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out_png = fig_out / "h2h_fid.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _plot_quality(rows, comp, fig_out, *, n_label, pretrained_key):
    ds = comp["dataset"]
    families = comp["families"]
    n_cols = len(QUALITY_METRICS)
    fig, axes = plt.subplots(
        nrows=1, ncols=n_cols, figsize=(3.8 * n_cols, 3.8), sharex=False
    )
    for col_i, (metric, title, ylabel, higher) in enumerate(QUALITY_METRICS):
        ax = axes[col_i]
        for prefix, label, color, ls, marker in families:
            xs, mu, lo, hi, _ = extract_bands(rows, prefix, metric)
            plot_band(ax, xs, mu, lo, hi, color, ls, marker, label)
        ax.set_xscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        arrow = " ↑" if higher else " ↓"
        ax.set_title(f"{ds} — {title}{arrow}")
        style_axes(ax)
        _overlay_pretrained(ax, metric, first_col=(col_i == 0), n_label=pretrained_key)
        if not higher:
            ax.invert_yaxis()  # up = better
    fig.suptitle(
        f"{n_label} {ds} head-to-head — {comp['title']} (sample quality / continuous novelty)\n"
        f"Line = mean across reps; band = min/max envelope. {comp['reps_note']}.",
        fontsize=12,
    )
    _figure_legend(fig, axes)
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    out_png = fig_out / "h2h_quality.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def run_all(
    *,
    comparisons,
    results_root: Path,
    fig_base: Path,
    n_label: str,
    pretrained_key: str | None,
) -> None:
    """Render every comparison's three panels under ``fig_base``."""
    for comp in comparisons:
        jsonl = results_root / comp["results_subdir"] / "sweep_results.clean.jsonl"
        rows = load_jsonl_with_reps(jsonl)
        if not rows:
            print(f"[skip] no rows in {jsonl}")
            continue
        fig_out = fig_base / comp["dataset"].lower() / comp["name"]
        fig_out.mkdir(parents=True, exist_ok=True)
        _plot_des(rows, comp, fig_out, n_label=n_label, pretrained_key=pretrained_key)
        _plot_fid(rows, comp, fig_out, n_label=n_label, pretrained_key=pretrained_key)
        _plot_quality(
            rows, comp, fig_out, n_label=n_label, pretrained_key=pretrained_key
        )
