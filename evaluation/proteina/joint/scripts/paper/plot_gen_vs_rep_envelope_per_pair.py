"""Per-pair generation-vs-representation envelope — REPA paper Fig 3c analog.

One panel per (dataset, REPA variant). Each panel shows:
  - the dataset's baseline trajectory (blue, connected in step order)
  - one REPA variant's trajectory (colored, connected in step order)
  - bubble size proportional to training step
  - a dashed arrow from the baseline checkpoint to the REPA checkpoint at the
    shared training step closest to TARGET_STEP (a pre-committed mid-training
    target — currently 400k, mirroring the REPA paper Fig 3c convention),
    labelled with the improvement

Two figures are produced per dataset metric pair:
  - gen_vs_rep_envelope_per_pair_fid.png       (y = FID, axis inverted so up=better, matches the REPA paper)
  - gen_vs_rep_envelope_per_pair_designability.png  (y = designability rate)

X axis is best-layer CATH-T top1 accuracy at t=1.0.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
from evaluation.proteina.lib import pretrained_overlay  # noqa: E402

# Overlay the pretrained Proteina (NVIDIA NGC 60M, ~1.3M steps) as an external
# eyeball baseline. Not step-comparable to our runs — see
# project_proteina_60m_layer_mismatch.md. Flip to False to hide.
SHOW_PRETRAINED = True
GEN_RESULTS = REPO_ROOT / "evaluation/proteina/generation/results/paper"
REP_RESULTS = REPO_ROOT / "evaluation/proteina/representation/results/paper"
FIG_OUT = REPO_ROOT / "evaluation/proteina/joint/figures/paper/n256_convergence"
FIG_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "PDB": {
        "gen_jsonl": GEN_RESULTS / "n256_convergence_pdb" / "sweep_results.jsonl",
        "rep_csv": REP_RESULTS
        / "n256_convergence_cath_if_dih_pdb"
        / "pretrained_sweep_results.csv",
        "baseline": ("baseline_256_bs24_2gpu", "Baseline (PDB)"),
        "repa_variants": [
            ("repa_l4_256_per_residue_bs24_2gpu", "REPA L4 GearNet", "tab:red", "s"),
            ("repa_l9_256_per_residue_bs24_2gpu", "REPA L9 GearNet", "tab:orange", "s"),
            ("repa_mpnn_l4_256_per_residue", "REPA L4 MPNN", "tab:green", "^"),
            ("repa_mpnn_l9_256_per_residue", "REPA L9 MPNN", "tab:purple", "^"),
        ],
    },
    "AFDB": {
        "gen_jsonl": GEN_RESULTS / "n256_convergence_afdb" / "sweep_results.jsonl",
        "rep_csv": REP_RESULTS
        / "n256_convergence_cath_if_dih_afdb"
        / "pretrained_sweep_results.csv",
        "baseline": ("baseline_afdb_256", "Baseline (AFDB)"),
        "repa_variants": [
            ("repa_l4_afdb_256", "REPA L4 GearNet", "tab:red", "s"),
            ("repa_l9_afdb_256", "REPA L9 GearNet", "tab:orange", "s"),
            ("repa_mpnn_l4_afdb_256", "REPA L4 MPNN", "tab:green", "^"),
            ("repa_mpnn_l9_afdb_256", "REPA L9 MPNN", "tab:purple", "^"),
        ],
    },
}

# Pre-committed step at which to draw the baseline → REPA arrow. We pick the
# shared step closest to this target. 400k is early enough that both methods
# are out of the "designability ≈ 0" noise floor on PDB but the gap hasn't
# closed yet, mirroring the REPA paper's mid-training Fig 3c convention.
TARGET_STEP = 400_000

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
        rep_map = load_rep_best(ds_cfg["rep_csv"], rep_key)

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

            # baseline
            if base_pts:
                xs = [p[1] for p in base_pts]
                ys = [p[2] for p in base_pts]
                sizes = [25 + 0.00007 * p[0] for p in base_pts]
                ax.plot(
                    xs, ys, "-", color="tab:blue", alpha=0.55, linewidth=1.2, zorder=2
                )
                ax.scatter(
                    xs,
                    ys,
                    s=sizes,
                    color="tab:blue",
                    marker="o",
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.9,
                    label=base_label,
                    zorder=3,
                )

            # repa variant
            if rep_pts:
                xs = [p[1] for p in rep_pts]
                ys = [p[2] for p in rep_pts]
                sizes = [25 + 0.00007 * p[0] for p in rep_pts]
                ax.plot(
                    xs, ys, "-", color=rep_color, alpha=0.55, linewidth=1.2, zorder=2
                )
                ax.scatter(
                    xs,
                    ys,
                    s=sizes,
                    color=rep_color,
                    marker=rep_marker,
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.9,
                    label=rep_label,
                    zorder=3,
                )

            # step-matched arrow at the shared step closest to TARGET_STEP
            # (pre-committed target, not data-dependent; mirrors the REPA-paper
            # convention of highlighting a mid-training gap rather than the
            # converged endpoint or the biggest win)
            base_steps = {p[0]: (p[1], p[2]) for p in base_pts}
            rep_steps = {p[0]: (p[1], p[2]) for p in rep_pts}
            shared = sorted(set(base_steps) & set(rep_steps))
            if shared:
                s_match = min(shared, key=lambda s: abs(s - TARGET_STEP))
                bx, by = base_steps[s_match]
                rx, ry = rep_steps[s_match]
                ax.annotate(
                    "",
                    xy=(rx, ry),
                    xytext=(bx, by),
                    arrowprops=dict(
                        arrowstyle="->",
                        linestyle="--",
                        color="black",
                        linewidth=1.4,
                        alpha=0.85,
                    ),
                    zorder=4,
                )
                dx = rx - bx
                dy = ry - by
                direction_ok = (dx > 0) and ((dy < 0) if lower_better else (dy > 0))
                txt = f"step {s_match // 1000}k\nΔ{x_label}={dx:+.2f}\nΔ{y_label}={dy:+.2f}"
                # anchor in upper-right of axes (in axes-fraction coords) to
                # avoid colliding with data points
                ax.text(
                    0.97,
                    0.97,
                    txt,
                    transform=ax.transAxes,
                    fontsize=7.5,
                    ha="right",
                    va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor=("#e6ffe6" if direction_ok else "#fff0f0"),
                        edgecolor="black",
                        linewidth=0.4,
                        alpha=0.92,
                    ),
                    zorder=5,
                )

            # Optional: overlay pretrained Proteina (NGC v1.3 60M) as a
            # gold ★ at its (rep, gen) coordinate. Not step-comparable to our
            # runs — purely an external eyeball reference.
            if SHOW_PRETRAINED:
                # Map our short REP_X_OPTIONS keys to pretrained_overlay's column names.
                _pretrained_rep_key = {"if_top1": "if_top1_acc"}.get(rep_key, rep_key)
                pre_rep = pretrained_overlay.load_rep(ds_name).get(_pretrained_rep_key)
                pre_gen = pretrained_overlay.load_gen().get(gen_col)
                if pre_rep is not None and pre_gen is not None:
                    ax.scatter(
                        [pre_rep],
                        [pre_gen],
                        s=260,
                        marker=pretrained_overlay.PRETRAINED_MARKER,
                        color=pretrained_overlay.PRETRAINED_COLOR,
                        edgecolor="black",
                        linewidth=0.7,
                        zorder=6,
                        label=pretrained_overlay.PRETRAINED_LABEL,
                    )

            ax.set_xlabel(f"{x_label} (best layer)")
            ax.set_ylabel(y_label)
            ax.set_title(f"{ds_name}: baseline vs {rep_label}", fontsize=10)
            ax.grid(True, alpha=0.3)
            # Legend in upper-left for FID (good corner is now bottom-right, so
            # upper-left is the empty quadrant). For designability the good
            # corner is top-right so lower-left stays empty — pick by metric.
            ax.legend(loc="upper left" if lower_better else "lower right", fontsize=7)

    # Y-axis orientation: FID stays in its natural direction (lower at bottom,
    # matches REPA paper Fig 3c — "good corner" is bottom-right). Designability
    # is left as default (higher at top — "good corner" is top-right).

    suptitle = (
        f"n=256 — generation vs representation envelope per baseline–REPA pair\n"
        f"y = {y_label}{' (lower = better, matches REPA paper convention)' if lower_better else ''}; "
        f"x = {x_label} (best layer at t=1.0). "
        f"Bubble size ∝ training step. Dashed arrow: baseline → REPA at the shared step closest to {TARGET_STEP // 1000}k."
    )
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
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
