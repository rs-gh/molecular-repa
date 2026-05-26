"""Multi-seed FID-family convergence plot — PDB + AFDB, n=128 and n=256.

Multi-seed companion to `plot_convergence_fid.py` / `plot_convergence_fid_n128.py`.
Reads `sweep_results.clean.jsonl` and shows mean across reps with a min/max
envelope band. Single-seed legacy rows appear as bare markers with no band.

PDB has full 3-rep coverage on both n=128 and n=256. AFDB has 3-rep coverage
only on the n=128 ext-sweep ckpts (baseline 800/900/1100/1200k and mpnn-L4
800/900/1100k); other AFDB points are single-seed bare markers.
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
from evaluation.proteina.lib import pretrained_overlay  # noqa: E402

SHOW_PRETRAINED = True

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/paper"

# Per-N RUN_FAMILIES, matched to the single-seed scripts. Visual encoding:
#   colour    = REPA layer depth (baseline=blue, L4=red, L9=orange)
#   linestyle = same as colour group (B/W readability)
#   marker    = encoder family (baseline=square, GearNet=circle, MPNN=triangle)
RUN_FAMILIES_N128 = {
    "PDB": [
        ("baseline_128_bs80", "Baseline (PDB)", "tab:blue", "-", "s"),
        ("repa_l4_128_bs80", "REPA L4 GearNet (PDB)", "tab:red", "-", "o"),
        ("repa_l9_128_bs80", "REPA L9 GearNet (PDB)", "tab:green", "-", "o"),
        ("repa_mpnn_l4_128_bs80", "REPA L4 MPNN (PDB)", "tab:red", "--", "^"),
        ("repa_mpnn_l9_128_bs80_2gpu", "REPA L9 MPNN (PDB)", "tab:green", "--", "^"),
        (
            "repa_l4_128_random",
            "REPA L4 GearNet-rand (PDB, ctrl)",
            "tab:gray",
            ":",
            "D",
        ),
    ],
    "AFDB": [
        ("baseline_afdb_128_bs80", "Baseline (AFDB)", "tab:blue", "-", "s"),
        ("repa_l4_afdb_128_bs80", "REPA L4 GearNet (AFDB)", "tab:red", "-", "o"),
        ("repa_mpnn_l4_afdb_128_bs80", "REPA L4 MPNN (AFDB)", "tab:red", "--", "^"),
        (
            "repa_mpnn_l9_afdb_128_bs80_2gpu",
            "REPA L9 MPNN (AFDB)",
            "tab:green",
            "--",
            "^",
        ),
    ],
}
RUN_FAMILIES_N256 = {
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
            "tab:green",
            "-",
            "o",
        ),
        ("repa_mpnn_l4_256_per_residue", "REPA L4 MPNN (PDB)", "tab:red", "--", "^"),
        ("repa_mpnn_l9_256_per_residue", "REPA L9 MPNN (PDB)", "tab:green", "--", "^"),
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
        ("repa_l9_afdb_256", "REPA L9 GearNet (AFDB, partial)", "tab:green", ":", "o"),
        ("repa_mpnn_l4_afdb_256", "REPA L4 MPNN (AFDB)", "tab:red", "--", "^"),
        ("repa_mpnn_l9_afdb_256", "REPA L9 MPNN (AFDB)", "tab:green", "--", "^"),
    ],
}

# (n, fig_subdir, [(ds_name, results_subdir, families), ...])
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

DATASET_METRICS = [
    # FID title is filled in per-call with the actual sample budget:
    # n=128 → 500 PDBs, n=256 → 1125 PDBs. The image-generation "FID-50K"
    # convention doesn't apply here.
    ("FID", "FID"),
    ("fJSD_A", "fJSD (Architecture)"),
    ("fJSD_C", "fJSD (Class)"),
    ("fJSD_T", "fJSD (Topology)"),
]


def _fid_label(n: int) -> str:
    return {128: "FID-500", 256: "FID-1.1K"}.get(n, "FID")


FS_METRICS = [
    ("_res_fS_C", "fS (Class)"),
    ("_res_fS_A", "fS (Architecture)"),
    ("_res_fS_T", "fS (Topology)"),
]


def plot_one(n: int, fig_subdir: str, datasets) -> None:
    fig_out = ROOT / "figures/paper" / fig_subdir
    fig_out.mkdir(parents=True, exist_ok=True)
    n_cols = len(DATASET_METRICS) + len(FS_METRICS)
    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=n_cols,
        figsize=(3.8 * n_cols, 3.6 * len(datasets)),
        sharex=False,
        squeeze=False,
    )

    for row_i, (ds_name, results_subdir, families) in enumerate(datasets):
        jsonl = RESULTS / results_subdir / "sweep_results.clean.jsonl"
        rows = load_jsonl_with_reps(jsonl)
        if not rows:
            for col_i in range(n_cols):
                axes[row_i, col_i].set_visible(False)
            continue

        for col_i, (suffix, title) in enumerate(DATASET_METRICS):
            ax = axes[row_i, col_i]
            col = f"_res_{ds_name}_{suffix}"
            for prefix, label, color, ls, marker in families:
                xs, mu, lo, hi, _ = extract_bands(rows, prefix, col)
                plot_band(ax, xs, mu, lo, hi, color, ls, marker, label)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Training step")
            ax.set_ylabel("value (lower = closer, log y)")
            display_title = _fid_label(n) if title == "FID" else title
            ax.set_title(f"{ds_name} — {display_title} ↓")
            ax.invert_yaxis()  # up = better
            style_axes(ax, log_y=True)
            if SHOW_PRETRAINED:
                pre_val = pretrained_overlay.load_gen().get(f"_res_{ds_name}_{suffix}")
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

        for j, (col, title) in enumerate(FS_METRICS):
            ax = axes[row_i, len(DATASET_METRICS) + j]
            for prefix, label, color, ls, marker in families:
                xs, mu, lo, hi, _ = extract_bands(rows, prefix, col)
                plot_band(ax, xs, mu, lo, hi, color, ls, marker, label)
            ax.set_xscale("log")
            ax.set_xlabel("Training step")
            ax.set_ylabel("entropy (higher = more coverage)")
            ax.set_title(f"{ds_name} — {title} ↑")
            style_axes(ax)
            if SHOW_PRETRAINED:
                pre_val = pretrained_overlay.load_gen().get(col)
                if pre_val is not None:
                    ax.axhline(
                        pre_val,
                        color=pretrained_overlay.PRETRAINED_COLOR,
                        linestyle="--",
                        linewidth=2.6,
                        alpha=0.9,
                        zorder=1,
                    )

    fig.suptitle(
        f"n={n} multi-seed convergence — FID-family distributional metrics\n"
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
    out_png = fig_out / "convergence_fid.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    for n, fig_subdir, datasets in CONFIGS:
        plot_one(n, fig_subdir, datasets)


if __name__ == "__main__":
    main()
