"""Generation-vs-representation envelope — REPA paper Fig 3c analog.

Scatter:
    x = best CATH-T probe accuracy across layers at (run, step)
    y = designability rate at the same (run, step)
Each point is one checkpoint; lines connect same-run points in step order to
visualise per-run trajectories. REPA variants should sit above-right of the
baseline curve (better generation AND better representations).

Output: ``evaluation/proteina/joint/figures/paper/n256_convergence/gen_vs_rep_envelope.{png,pdf}``
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[5]
GEN_RESULTS = REPO_ROOT / "evaluation/proteina/generation/results/paper"
REP_RESULTS = REPO_ROOT / "evaluation/proteina/representation/results/paper"
FIG_BASE = (
    REPO_ROOT / "evaluation/proteina/joint/figures/paper/n256_convergence/envelope"
)

GEN_PDB = GEN_RESULTS / "n256_convergence_pdb" / "sweep_results.jsonl"
GEN_AFDB = GEN_RESULTS / "n256_convergence_afdb" / "sweep_results.jsonl"


def _rep(dirname: str) -> Path:
    return REP_RESULTS / dirname / "pretrained_sweep_results.csv"


# One envelope figure per variant; each variant pulls CATH-T accuracy
# (best layer) from a different rep source. Datasets missing from a variant
# are skipped.
VARIANTS: Dict[str, Dict[str, dict]] = {
    "cath_if_dih": {
        "PDB": {
            "gen_jsonl": GEN_PDB,
            "rep_csv": _rep("n256_convergence_cath_if_dih_pdb"),
        },
        "AFDB": {
            "gen_jsonl": GEN_AFDB,
            "rep_csv": _rep("n256_convergence_cath_if_dih_afdb"),
        },
    },
    "cleantrain": {
        "PDB": {
            "gen_jsonl": GEN_PDB,
            "rep_csv": _rep("n256_convergence_cleantrain_pdb"),
        },
    },
    "xclean": {
        "PDB": {"gen_jsonl": GEN_PDB, "rep_csv": _rep("n256_xclean_afdb_pdb")},
        "AFDB": {"gen_jsonl": GEN_AFDB, "rep_csv": _rep("n256_xclean_pdb_afdb")},
    },
}

RUN_FAMILIES = {
    "PDB": [
        ("baseline_256_bs24_2gpu", "Baseline (PDB)", "tab:blue", "o"),
        ("repa_l4_256_per_residue_bs24_2gpu", "REPA L4 GearNet (PDB)", "tab:red", "o"),
        (
            "repa_l9_256_per_residue_bs24_2gpu",
            "REPA L9 GearNet (PDB)",
            "tab:green",
            "o",
        ),
        ("repa_mpnn_l4_256_per_residue", "REPA L4 MPNN (PDB)", "tab:red", "^"),
        ("repa_mpnn_l9_256_per_residue", "REPA L9 MPNN (PDB)", "tab:green", "^"),
    ],
    "AFDB": [
        ("baseline_afdb_256", "Baseline (AFDB)", "tab:blue", "o"),
        ("repa_l4_afdb_256", "REPA L4 GearNet (AFDB)", "tab:red", "o"),
        ("repa_l9_afdb_256", "REPA L9 GearNet (AFDB)", "tab:green", "o"),
        ("repa_mpnn_l4_afdb_256", "REPA L4 MPNN (AFDB)", "tab:red", "^"),
        ("repa_mpnn_l9_afdb_256", "REPA L9 MPNN (AFDB)", "tab:green", "^"),
    ],
}


def fnum(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_gen(path: Path) -> Dict[Tuple[str, int], dict]:
    """Return {(run, step): {designability, fid_pdb, fid_afdb}} for ok rows."""
    out: Dict[Tuple[str, int], dict] = {}
    if not path.exists():
        return out
    for line in path.open():
        r = json.loads(line)
        if r.get("error", "NONE") != "NONE":
            continue
        run = r.get("run")
        step = r.get("step")
        if run is None or step is None:
            continue
        out[(run, int(step))] = {
            "designability": fnum(r.get("_res_designability_rate")),
            "fid_pdb": fnum(r.get("_res_PDB_FID")),
            "fid_afdb": fnum(r.get("_res_AFDB_FID")),
        }
    return out


# Pair each gen dataset with its matched FID column (PDB-gen → FID vs PDB,
# AFDB-gen → FID vs AFDB).
FID_COL_BY_DS = {"PDB": ("fid_pdb", "FID vs PDB"), "AFDB": ("fid_afdb", "FID vs AFDB")}

# X-axis rep proxies: (key, label, probe_kind, cath_level, metric_col).
X_METRICS = [
    ("cath_A", "CATH-A top1 acc (best layer)", "cath", "A", "cath_accuracy"),
    ("if", "IF top1 acc (best layer)", "inverse_folding", None, "if_top1_acc"),
]


def load_rep_best(
    path: Path,
    probe_kind: str = "cath",
    cath_level: Optional[str] = "A",
    metric_col: str = "cath_accuracy",
) -> Dict[Tuple[str, int], float]:
    """Return {(run, step): max-over-layers metric}."""
    out: Dict[Tuple[str, int], float] = defaultdict(lambda: float("-inf"))
    if not path.exists():
        return {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("probe_kind") != probe_kind:
                continue
            if cath_level is not None and r.get("cath_level") != cath_level:
                continue
            run = r.get("run")
            s = fnum(r.get("step"))
            v = fnum(r.get(metric_col))
            if run is None or s is None or v is None:
                continue
            k = (run, int(s))
            if v > out[k]:
                out[k] = v
    return dict(out)


Y_METRICS = [
    ("designability", "Designability rate", False),  # higher is better
    ("fid", "FID (matched to dataset)", True),  # lower is better → invert
]


def _plot_variant(variant: str, datasets: Dict[str, dict]) -> None:
    fig_out = FIG_BASE / variant
    fig_out.mkdir(parents=True, exist_ok=True)
    n_ds = len(datasets)
    n_x = len(X_METRICS)
    n_y = len(Y_METRICS)
    # Rows = dataset × x-metric (cath_A, IF). Cols = y-metric (designability, FID).
    n_rows = n_ds * n_x
    fig, axes = plt.subplots(
        n_rows, n_y, figsize=(5.5 * n_y, 4.6 * n_rows), squeeze=False
    )
    for ds_idx, (ds_name, paths) in enumerate(datasets.items()):
        gen_map = load_gen(paths["gen_jsonl"])
        fid_key, fid_label = FID_COL_BY_DS[ds_name]
        for x_idx, (x_key, x_label, probe_kind, cath_level, metric_col) in enumerate(
            X_METRICS
        ):
            rep_map = load_rep_best(
                paths["rep_csv"],
                probe_kind=probe_kind,
                cath_level=cath_level,
                metric_col=metric_col,
            )
            row = ds_idx * n_x + x_idx
            for col, (y_key, y_label, lower_better) in enumerate(Y_METRICS):
                ax = axes[row, col]
                gen_col = fid_key if y_key == "fid" else y_key
                y_axis_label = fid_label if y_key == "fid" else y_label
                for prefix, label, color, marker in RUN_FAMILIES[ds_name]:
                    pts = []
                    for (run, step), gen_vals in gen_map.items():
                        if not run.startswith(prefix):
                            continue
                        acc = rep_map.get((run, step))
                        yv = gen_vals.get(gen_col)
                        if acc is None or acc == float("-inf") or yv is None:
                            continue
                        pts.append((step, acc, yv))
                    if not pts:
                        continue
                    pts.sort()
                    xs = [p[1] for p in pts]
                    ys = [p[2] for p in pts]
                    sizes = [20 + 0.00006 * p[0] for p in pts]
                    ax.plot(xs, ys, linewidth=1.0, color=color, alpha=0.6)
                    ax.scatter(
                        xs,
                        ys,
                        s=sizes,
                        color=color,
                        marker=marker,
                        edgecolor="black",
                        linewidth=0.4,
                        label=label if col == 0 else None,
                        zorder=3,
                    )
                ax.set_xlabel(x_label)
                ax.set_ylabel(
                    y_axis_label
                    + (" (axis inverted; up = better)" if lower_better else "")
                )
                ax.set_title(f"{ds_name} / {x_key}: {y_axis_label}")
                ax.grid(True, alpha=0.3)
                if lower_better:
                    ax.invert_yaxis()
            axes[row, 0].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"n=256 (rep={variant}) — generation vs representation envelope (point size ∝ training step)\n"
        f"Up-right = better on both axes; lines connect same-run checkpoints in step order.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = fig_out / "gen_vs_rep_envelope.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    for variant, datasets in VARIANTS.items():
        _plot_variant(variant, datasets)


if __name__ == "__main__":
    main()
