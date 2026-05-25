"""Per-pair envelope: SS 2D-JSD (designable) vs designability rate.

Mirrors plot_gen_vs_rep_envelope_per_pair.py but with two generation metrics:
    x = designability rate          (higher = better, going right)
    y = SS 2D-JSD (designable)      (lower  = better, going down; not inverted)

Good corner = bottom-right. Same baseline → REPA dashed arrow at the shared
step closest to TARGET_STEP=400k as the gen-vs-rep envelope plot. Bubble size
∝ training step. Pretrained Proteina overlaid as a gold ★.

Two figures are produced — one per SS-JSD reference set:
    ssjsd_pdb_vs_des_envelope_per_pair.png   (y = _res_ss_jsd_pdb_designable_2d)
    ssjsd_afdb_vs_des_envelope_per_pair.png  (y = _res_ss_jsd_afdb_designable_2d)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def _humanize(v, _pos=None):
    av = abs(v)
    if av >= 1e6:
        return f"{v / 1e6:g}M"
    if av >= 1e3:
        return f"{v / 1e3:g}K"
    if av >= 1:
        return f"{v:g}"
    return f"{v:.3g}"


REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
from evaluation.proteina.lib import pretrained_overlay  # noqa: E402

SHOW_PRETRAINED = True

GEN_RESULTS = REPO_ROOT / "evaluation/proteina/generation/results/paper"
FIG_OUT = REPO_ROOT / "evaluation/proteina/generation/figures/paper/n256_convergence"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# Pre-committed step at which to draw the baseline → REPA arrow. Picks the
# shared step closest to this target. Matches plot_gen_vs_rep_envelope_per_pair.
TARGET_STEP = 400_000

DATASETS = {
    "PDB": {
        "gen_jsonl": GEN_RESULTS / "n256_convergence_pdb" / "sweep_results.jsonl",
        "baseline": ("baseline_256_bs24_2gpu", "Baseline (PDB)"),
        "repa_variants": [
            ("repa_l4_256_per_residue_bs24_2gpu", "REPA L4 GearNet", "tab:red", "s"),
            ("repa_l9_256_per_residue_bs24_2gpu", "REPA L9 GearNet", "tab:green", "s"),
            ("repa_mpnn_l4_256_per_residue", "REPA L4 MPNN", "tab:green", "^"),
            ("repa_mpnn_l9_256_per_residue", "REPA L9 MPNN", "tab:purple", "^"),
        ],
    },
    "AFDB": {
        "gen_jsonl": GEN_RESULTS / "n256_convergence_afdb" / "sweep_results.jsonl",
        "baseline": ("baseline_afdb_256", "Baseline (AFDB)"),
        "repa_variants": [
            ("repa_l4_afdb_256", "REPA L4 GearNet", "tab:red", "s"),
            ("repa_l9_afdb_256", "REPA L9 GearNet", "tab:green", "s"),
            ("repa_mpnn_l4_afdb_256", "REPA L4 MPNN", "tab:green", "^"),
            ("repa_mpnn_l9_afdb_256", "REPA L9 MPNN", "tab:purple", "^"),
        ],
    },
}

Y_METRICS = {
    "pdb": ("_res_ss_jsd_pdb_designable_2d", "SS 2D-JSD vs PDB (designable)"),
    "afdb": ("_res_ss_jsd_afdb_designable_2d", "SS 2D-JSD vs AFDB (designable)"),
}
X_COL = "_res_designability_rate"
X_LABEL = "Designability (%)"
X_SCALE = 100.0  # _res_designability_rate is a 0–1 fraction; show as 0–100 %.


def fnum(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def load_gen(path: Path, y_col: str) -> Dict[Tuple[str, int], Tuple[float, float]]:
    """Return {(run, step): (x, y)} dropping rows missing either metric."""
    out: Dict[Tuple[str, int], Tuple[float, float]] = {}
    if not path.exists():
        return out
    # Deduplicate: latest line per (run, step) wins.
    latest: Dict[Tuple[str, int], dict] = {}
    for line in path.open():
        r = json.loads(line)
        if r.get("error") and r["error"] != "NONE":
            continue
        run = r.get("run")
        step = r.get("step")
        if run is None or step is None:
            continue
        latest[(run, int(step))] = r
    for (run, step), r in latest.items():
        x = fnum(r.get(X_COL))
        y = fnum(r.get(y_col))
        if x is None or y is None:
            continue
        out[(run, step)] = (x * X_SCALE, y)
    return out


def runs_for_prefix(prefix: str, keys) -> List[Tuple[str, int]]:
    return sorted([k for k in keys if k[0].startswith(prefix)], key=lambda k: k[1])


def plot_one(ref: str) -> None:
    y_col, y_label = Y_METRICS[ref]
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
        gen_map = load_gen(ds_cfg["gen_jsonl"], y_col)

        base_prefix, base_label = ds_cfg["baseline"]
        base_keys = runs_for_prefix(base_prefix, gen_map.keys())
        base_pts = [(k[1], *gen_map[k]) for k in base_keys]  # (step, x, y)

        for col, (rep_prefix, rep_label, rep_color, rep_marker) in enumerate(
            ds_cfg["repa_variants"]
        ):
            ax = axes[row, col]
            rep_keys = runs_for_prefix(rep_prefix, gen_map.keys())
            rep_pts = [(k[1], *gen_map[k]) for k in rep_keys]

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

            # Step-matched arrow at shared step closest to TARGET_STEP.
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
                # Good corner = bottom-right (high designability, low SS-JSD).
                direction_ok = (dx > 0) and (dy < 0)
                txt = f"step {s_match // 1000}k\nΔ{X_LABEL}={dx:+.2f}\nΔ{y_label}={dy:+.2f}"
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

            # Pretrained overlay (gold ★ at its (x, y) coordinate).
            if SHOW_PRETRAINED:
                pre = pretrained_overlay.load_gen("n256")
                px, py = pre.get(X_COL), pre.get(y_col)
                if px is not None:
                    px = px * X_SCALE
                if px is not None and py is not None:
                    ax.scatter(
                        [px],
                        [py],
                        s=260,
                        marker=pretrained_overlay.PRETRAINED_MARKER,
                        color=pretrained_overlay.PRETRAINED_COLOR,
                        edgecolor="black",
                        linewidth=0.7,
                        zorder=6,
                        label=pretrained_overlay.PRETRAINED_LABEL,
                    )

            ax.set_xlabel(X_LABEL)
            ax.set_ylabel(y_label)
            ax.set_title(f"{ds_name}: baseline vs {rep_label}", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(FuncFormatter(_humanize))
            ax.yaxis.set_major_formatter(FuncFormatter(_humanize))
            # Good corner = bottom-right; legend in upper-left empty quadrant.
            ax.legend(loc="upper left", fontsize=7)

    suptitle = (
        f"n=256 — SS 2D-JSD vs designability envelope per baseline–REPA pair\n"
        f"y = {y_label} (lower = better); x = {X_LABEL} (higher = better). "
        f"Bubble size ∝ training step. Dashed arrow: baseline → REPA at the shared step closest to {TARGET_STEP // 1000}k."
    )
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIG_OUT / f"ssjsd_{ref}_vs_des_envelope_per_pair.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    for ref in Y_METRICS:
        plot_one(ref)


if __name__ == "__main__":
    main()
