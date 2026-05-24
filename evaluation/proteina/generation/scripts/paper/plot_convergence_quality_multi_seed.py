"""Multi-seed quality / continuous-novelty convergence plot — PDB + AFDB.

Companion to `plot_convergence_des_multi_seed.py` and
`plot_convergence_fid_multi_seed.py`. Those cover designability rate /
distributional metrics; this one fills the remaining sample-level
quality and continuous-novelty signals that exist in the jsonl but
aren't plotted elsewhere:

  * pLDDT (mean)           — sample quality, continuous version of designability
  * scRMSD (mean)          — raw signal behind designability rate
  * max-TM novelty PDB     — continuous companion to novelty_rate (TM<0.5 threshold)
  * max-TM novelty AFDB-SP — same, vs AFDB-SwissProt
  * TM-self (mean)         — intra-set self-consistency / mode-collapse signal

Same band semantics as the sibling plots: mean line + min/max envelope
across reps, with single-rep legacy rows showing as bare markers.
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

# (jsonl_key, title, ylabel, higher_is_better)
METRICS = [
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
                plot_band(ax, xs, mu, lo, hi, color, ls, marker, label)
            ax.set_xscale("log")
            ax.set_xlabel("Training step")
            ax.set_ylabel(ylabel)
            arrow = " ↑" if higher else " ↓"
            ax.set_title(f"{ds_name} — {title}{arrow}")
            if not higher:
                ax.invert_yaxis()  # up = better
            style_axes(ax)
            if SHOW_PRETRAINED:
                pre_val = pretrained_overlay.load_gen().get(metric)
                if pre_val is not None:
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
        f"n={n} multi-seed convergence — sample quality / continuous novelty\n"
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
    out_png = fig_out / "convergence_quality.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main():
    for n, fig_subdir, datasets in CONFIGS:
        plot_one(n, fig_subdir, datasets)


if __name__ == "__main__":
    main()
