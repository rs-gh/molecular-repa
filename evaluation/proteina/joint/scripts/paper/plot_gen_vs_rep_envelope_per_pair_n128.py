"""Per-pair generation-vs-representation envelope — REPA paper Fig 3c analog.

One panel per (dataset, REPA variant). Each panel shows:
  - the dataset's baseline trajectory (blue, connected in step order)
  - one REPA variant's trajectory (colored, connected in step order)
  - bubble size proportional to training step, with a tiny step label per bubble

Figures: y ∈ {FID-PDB, FID-AFDB, designability}, x ∈ {CATH-{C,A,T} top1, IF top1}.
FID y-axis is inverted so "up and to the right = better" in every panel.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
GEN_RESULTS = REPO_ROOT / "evaluation/proteina/generation/results/paper"
REP_RESULTS = REPO_ROOT / "evaluation/proteina/representation/results/paper"
FIG_OUT = REPO_ROOT / "evaluation/proteina/joint/figures/paper/n128_convergence"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# Rep CSV source depends on probe (cleaner-protocol switch, 2026-05-26):
#   - CATH probes: cleantrain (probe-side cleaned PDB val) for PDB-trained
#     models; cath_if_dih_afdb for AFDB-trained models (no cleantrain_afdb
#     dir exists, see project_repa_evidence_framing.md).
#   - IF probe: xclean for PDB-trained models (n128_xclean_afdb_pdb).
#     No n128_xclean_pdb_afdb exists, so AFDB-trained models fall back to
#     cath_if_dih_afdb for IF too.
DATASETS = {
    "PDB": {
        "gen_jsonl": GEN_RESULTS / "n128_convergence_pdb" / "sweep_results.jsonl",
        "rep_csv_cath": REP_RESULTS
        / "n128_convergence_cleantrain_pdb"
        / "pretrained_sweep_results.csv",
        "rep_csv_if": REP_RESULTS
        / "n128_xclean_afdb_pdb"
        / "pretrained_sweep_results.csv",
        "baseline": ("baseline_128_bs80", "Baseline (PDB)"),
        "repa_variants": [
            ("repa_l4_128_bs80", "REPA L4 GearNet", "tab:red", "o"),
            ("repa_l9_128_bs80", "REPA L9 GearNet", "tab:green", "o"),
            ("repa_mpnn_l4_128_bs80", "REPA L4 MPNN", "tab:red", "^"),
            ("repa_mpnn_l9_128_bs80_2gpu", "REPA L9 MPNN", "tab:green", "^"),
        ],
    },
    "AFDB": {
        "gen_jsonl": GEN_RESULTS / "n128_convergence_afdb" / "sweep_results.jsonl",
        "rep_csv_cath": REP_RESULTS
        / "n128_convergence_cath_if_dih_afdb"
        / "pretrained_sweep_results.csv",
        "rep_csv_if": REP_RESULTS
        / "n128_convergence_cath_if_dih_afdb"
        / "pretrained_sweep_results.csv",
        "baseline": ("baseline_afdb_128_bs80", "Baseline (AFDB)"),
        "repa_variants": [
            ("repa_l4_afdb_128_bs80", "REPA L4 GearNet", "tab:red", "o"),
            ("repa_mpnn_l4_afdb_128_bs80", "REPA L4 MPNN", "tab:red", "^"),
            ("repa_mpnn_l9_afdb_128_bs80_2gpu", "REPA L9 MPNN", "tab:green", "^"),
        ],
    },
}


def rep_csv_for(ds_cfg: dict, rep_key: str) -> Path:
    return ds_cfg["rep_csv_if"] if rep_key == "if_top1" else ds_cfg["rep_csv_cath"]


GEN_METRICS = {
    "fid_pdb": {"col": "_res_PDB_FID", "label": "FID vs PDB", "lower_better": True},
    "fid_afdb": {"col": "_res_AFDB_FID", "label": "FID vs AFDB", "lower_better": True},
    "designability": {
        "col": "_res_designability_rate",
        "label": "Designability rate",
        "lower_better": False,
    },
}


def fnum(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def load_gen(path: Path) -> Dict[Tuple[str, int], dict]:
    out: Dict[Tuple[str, int], dict] = {}
    if not path.exists():
        return out
    for line in path.open():
        r = json.loads(line)
        err = r.get("error")
        if err and err != "NONE":
            continue
        run = r.get("run")
        step = r.get("step")
        if run is None or step is None:
            continue
        out[(run, int(step))] = {
            "_res_PDB_FID": fnum(r.get("_res_PDB_FID")),
            "_res_AFDB_FID": fnum(r.get("_res_AFDB_FID")),
            "_res_designability_rate": fnum(r.get("_res_designability_rate")),
        }
    return out


# X-axis rep proxies, computed as best-layer reduction per checkpoint.
# (key, label, probe_kind, optional cath_level, metric_column, reduce-direction)
REP_X_OPTIONS = {
    "cath_C_top1": ("CATH-C top1 acc", "cath", "C", "cath_accuracy", "max"),
    "cath_A_top1": ("CATH-A top1 acc", "cath", "A", "cath_accuracy", "max"),
    "cath_T_top1": ("CATH-T top1 acc", "cath", "T", "cath_accuracy", "max"),
    "if_top1": ("IF top1 acc", "inverse_folding", None, "if_top1_acc", "max"),
}


def load_rep_best(path: Path, rep_key: str) -> Dict[Tuple[str, int], float]:
    """Best-layer rep proxy per (run, step) for the given REP_X_OPTIONS key."""
    _, kind, level, col, direction = REP_X_OPTIONS[rep_key]
    init = -np.inf if direction == "max" else np.inf
    out: Dict[Tuple[str, int], float] = defaultdict(lambda: init)
    if not path.exists():
        return {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("probe_kind") != kind:
                continue
            if level is not None and r.get("cath_level") != level:
                continue
            run = r.get("run")
            s = fnum(r.get("step"))
            v = fnum(r.get(col))
            if run is None or s is None or v is None:
                continue
            k = (run, int(s))
            cur = out[k]
            if (direction == "max" and v > cur) or (direction == "min" and v < cur):
                out[k] = v
    return {k: v for k, v in out.items() if np.isfinite(v)}


def runs_for_prefix(prefix: str, keys) -> List[Tuple[str, int]]:
    return sorted([k for k in keys if k[0].startswith(prefix)], key=lambda k: k[1])


def _draw_run(ax, pts, color, marker, label):
    """Plot a single run's trajectory: line + bubbles + tiny per-step labels."""
    if not pts:
        return
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    sizes = [25 + 0.00007 * p[0] for p in pts]
    ax.plot(xs, ys, "-", color=color, alpha=0.55, linewidth=1.2, zorder=2)
    ax.scatter(
        xs,
        ys,
        s=sizes,
        color=color,
        marker=marker,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
        label=label,
        zorder=3,
    )
    for step, x, y in pts:
        ax.annotate(
            f"{step // 1000}k",
            xy=(x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=5.5,
            color="dimgray",
            alpha=0.9,
            zorder=4,
        )


def plot_metric(metric_key: str, rep_key: str) -> None:
    meta = GEN_METRICS[metric_key]
    gen_col = meta["col"]
    y_label = meta["label"]
    lower_better = meta["lower_better"]
    x_label = REP_X_OPTIONS[rep_key][0]

    n_rows = len(DATASETS)
    n_cols = max(len(d["repa_variants"]) for d in DATASETS.values())
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.8 * n_rows),
        squeeze=False,
        sharey="row",
    )

    for row, (ds_name, ds_cfg) in enumerate(DATASETS.items()):
        gen_map = load_gen(ds_cfg["gen_jsonl"])
        rep_map = load_rep_best(rep_csv_for(ds_cfg, rep_key), rep_key)

        base_prefix, base_label = ds_cfg["baseline"]
        base_keys = runs_for_prefix(base_prefix, gen_map.keys())
        base_pts = []  # (step, cath, y)
        for k in base_keys:
            cath = rep_map.get(k)
            y = gen_map[k].get(gen_col)
            if cath is None or y is None:
                continue
            base_pts.append((k[1], cath, y))

        for col, (rep_prefix, rep_label, rep_color, rep_marker) in enumerate(
            ds_cfg["repa_variants"]
        ):
            ax = axes[row, col]
            rep_keys = runs_for_prefix(rep_prefix, gen_map.keys())
            rep_pts = []
            for k in rep_keys:
                cath = rep_map.get(k)
                y = gen_map[k].get(gen_col)
                if cath is None or y is None:
                    continue
                rep_pts.append((k[1], cath, y))

            _draw_run(ax, base_pts, "tab:blue", "o", base_label)
            _draw_run(ax, rep_pts, rep_color, rep_marker, rep_label)

            ax.set_xlabel(f"{x_label} (best layer)")
            ax.set_ylabel(y_label)
            ax.set_title(f"{ds_name}: baseline vs {rep_label}", fontsize=10)
            ax.grid(True, alpha=0.3)

        if lower_better:
            # Invert only once per shared-y row (acts on all axes in the row)
            axes[row, 0].invert_yaxis()

    # One shared figure-level legend on the right (deduplicated across axes).
    handles_by_label: Dict[str, object] = {}
    for ax in axes.ravel():
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            handles_by_label.setdefault(lbl, h)
    fig.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        fontsize=8,
        frameon=True,
    )

    rep_source = (
        "xclean (cross-DB clean; AFDB fallback: cath_if_dih_afdb)"
        if rep_key == "if_top1"
        else "cleantrain (PDB) / cath_if_dih (AFDB)"
    )
    suptitle = (
        f"n=128 — generation vs representation envelope per baseline–REPA pair\n"
        f"y = {y_label}{' (axis inverted; up = better)' if lower_better else ''}; "
        f"x = {x_label} (best layer at t=1.0; rep source: {rep_source}). "
        f"Bubble size ∝ step; tiny gray label = step in k."
    )
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.88, 0.92])
    out = FIG_OUT / f"gen_vs_rep_envelope_per_pair_{rep_key}_vs_{metric_key}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    for rep_key in REP_X_OPTIONS:
        for gen_key in GEN_METRICS:
            plot_metric(gen_key, rep_key)


if __name__ == "__main__":
    main()
