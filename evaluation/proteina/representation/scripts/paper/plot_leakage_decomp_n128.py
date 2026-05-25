"""Three-way leakage decomposition: dirty | cleantrain | cleanval at n=128.

baseline_128_bs80 vs repa_l4_128_bs80, best layer per (run, step).
pretrained_dfs_60m_n128_paper drawn as a horizontal reference per panel
(single fixed ckpt at ~1.3M steps, not step-comparable).

Columns:
  dirty       — both probe-fit & eval are leaky (baseline regime, n_eval=1237)
  cleantrain  — probe-fit is homology-filtered, eval is leaky (n_eval=1237)
  cleanval    — probe-fit leaky, eval is homology-filtered (n_eval=44; CATH n=9)

Rows: IF top-1, Dihedral total MAE, CATH-C, CATH-A, CATH-T.

Built 2026-05-23, pretrained added 2026-05-24.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parents[5]
RES = REPO / "evaluation/proteina/representation/results/paper"
FIG = REPO / "evaluation/proteina/representation/figures/paper/leakage_decomp/n128/pdb"
FIG.mkdir(parents=True, exist_ok=True)

CONDS = [
    ("dirty", RES / "n128_convergence_cath_if_dih_pdb", 1237),
    ("cleantrain", RES / "n128_convergence_cleantrain_pdb", 1237),
    ("cleanval", RES / "n128_convergence_cleanval_pdb", 44),
    ("xclean-AFDB", RES / "n128_xclean_afdb_pdb", 154),
]
# Convention: blue=baseline, red=L4 REPA, green=L9 REPA;
#             circle=GearNet target, triangle=MPNN target;
#             baseline = square, pretrained = blue dashed reference.
COLORS = {
    "baseline": "#1f77b4",
    "repa_l4_gearnet": "#d62728",
    "repa_l4_mpnn": "#d62728",
    "repa_l4_random": "#d62728",
    "repa_l9_gearnet": "#2ca02c",
    "repa_l9_mpnn": "#2ca02c",
    "baseline_afdb": "#1f77b4",
    "pretrained": "#1f77b4",
}
MARKERS = {
    "baseline": "s",
    "repa_l4_gearnet": "o",
    "repa_l4_mpnn": "^",
    "repa_l4_random": "x",
    "repa_l9_gearnet": "o",
    "repa_l9_mpnn": "^",
    "baseline_afdb": "D",
}
LINESTYLES = {
    "baseline": "-",
    "repa_l4_gearnet": "-",
    "repa_l4_mpnn": "-",
    "repa_l4_random": "--",
    "repa_l9_gearnet": "-",
    "repa_l9_mpnn": "-",
    "baseline_afdb": ":",
}
LABELS = {
    "baseline": "baseline (PDB)",
    "repa_l4_gearnet": "REPA L4 GearNet (PDB)",
    "repa_l4_mpnn": "REPA L4 MPNN (PDB)",
    "repa_l4_random": "REPA L4 random-encoder (PDB)",
    "repa_l9_gearnet": "REPA L9 GearNet (PDB)",
    "repa_l9_mpnn": "REPA L9 MPNN (PDB)",
    "baseline_afdb": "baseline (AFDB)",
}
RUN_RE = {
    "baseline": re.compile(r"^baseline_128_bs80(_step\d+k)?$"),
    "repa_l4_gearnet": re.compile(r"^repa_l4_128_bs80(_step\d+k)?$"),
    "repa_l4_mpnn": re.compile(r"^repa_mpnn_l4_128_bs80(_step\d+k)?$"),
    "repa_l4_random": re.compile(r"^repa_l4_128_random(_step\d+k)?$"),
    "repa_l9_gearnet": re.compile(r"^repa_l9_128_bs80(_step\d+k)?$"),
    "repa_l9_mpnn": re.compile(r"^repa_mpnn_l9_128_bs80_2gpu(_step\d+k)?$"),
    "baseline_afdb": re.compile(r"^baseline_afdb_128_bs80(_step\d+k)?$"),
}
XCLEAN_ONLY = {"baseline_afdb"}
PRETRAINED = "pretrained_dfs_60m_n128_paper"


def load(d: Path):
    rows = []
    for p in sorted(d.glob("pretrained_sweep_results*.jsonl")):
        for ln in open(p):
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def best_layer(rows, run_filter, probe, field, lower=False, lvl=None):
    """run_filter: callable str -> bool."""
    by_step = defaultdict(list)
    for r in rows:
        if not run_filter(r.get("run", "")):
            continue
        if r.get("probe_kind") != probe:
            continue
        if r.get("t") != 1.0:
            continue
        if r.get("layer", -1) < 0:
            continue
        if lvl and (r.get("cath_level") != lvl or r.get("cath_head_type") != "linear"):
            continue
        if field not in r:
            continue
        by_step[r["step"]].append(r[field])
    return [(s, min(vs) if lower else max(vs)) for s, vs in sorted(by_step.items())]


def best_layer_single(rows, run_name, probe, field, lower=False, lvl=None):
    """Best layer over the (single-checkpoint) pretrained run."""
    cands = []
    for r in rows:
        if r.get("run") != run_name:
            continue
        if r.get("probe_kind") != probe:
            continue
        if r.get("t") != 1.0:
            continue
        if r.get("layer", -1) < 0:
            continue
        if lvl and (r.get("cath_level") != lvl or r.get("cath_head_type") != "linear"):
            continue
        if field not in r:
            continue
        cands.append(r[field])
    if not cands:
        return None
    return min(cands) if lower else max(cands)


def _human(v, _p=None):
    av = abs(v)
    if av >= 1e6:
        return f"{v/1e6:g}M"
    if av >= 1e3:
        return f"{v/1e3:g}K"
    return f"{v:g}"


PROBES = [
    ("IF top-1 acc", "accuracy ↑", "inverse_folding", "if_top1_acc", False, None),
    (
        "Dihedral total MAE",
        "MAE [deg] ↑ (axis inverted)",
        "dihedral",
        "dih_mae_total_deg",
        True,
        None,
    ),
    ("CATH-C linear acc", "accuracy ↑", "cath", "cath_accuracy", False, "C"),
    ("CATH-A linear acc", "accuracy ↑", "cath", "cath_accuracy", False, "A"),
    ("CATH-T linear acc", "accuracy ↑", "cath", "cath_accuracy", False, "T"),
]


def main():
    data = {name: load(p) for name, p, _ in CONDS}
    print({k: len(v) for k, v in data.items()})

    fig, axes = plt.subplots(
        len(PROBES), len(CONDS), figsize=(13, 3.0 * len(PROBES)), sharey="row"
    )
    for col, (cond, _, n_eval) in enumerate(CONDS):
        rows = data[cond]
        for row, (title, ylab, probe, field, lower, lvl) in enumerate(PROBES):
            ax = axes[row, col]
            # Step-indexed lines
            for who, regex in RUN_RE.items():
                if who in XCLEAN_ONLY and cond != "xclean-AFDB":
                    continue
                pts = best_layer(
                    rows,
                    lambda n, _r=regex: bool(_r.match(n)),
                    probe,
                    field,
                    lower=lower,
                    lvl=lvl,
                )
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(
                    xs,
                    ys,
                    color=COLORS[who],
                    marker=MARKERS[who],
                    markersize=5,
                    linewidth=1.6,
                    linestyle=LINESTYLES.get(who, "-"),
                    label=LABELS[who],
                )
            # Pretrained horizontal reference
            p_val = best_layer_single(
                rows, PRETRAINED, probe, field, lower=lower, lvl=lvl
            )
            if p_val is not None:
                ax.axhline(
                    p_val,
                    color=COLORS["pretrained"],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.85,
                    label="pretrained (NVIDIA 60M)",
                )
            if row == 0:
                ax.set_title(f"{cond}\n(n_eval={n_eval})", fontsize=10)
            if col == 0:
                ax.set_ylabel(ylab + "\n" + title, fontsize=9)
            if row == len(PROBES) - 1:
                ax.set_xlabel("training step")
            ax.xaxis.set_major_formatter(FuncFormatter(_human))
            ax.grid(True, axis="y", alpha=0.3)
            # invert_yaxis deferred until after the loop — see comment in
            # plot_leakage_decomp_n256.py (sharey toggle bug).
            if cond in ("cleanval", "xclean-AFDB") and lvl is not None:
                cath_iv = max(
                    (
                        r.get("cath_n_eval_in_vocab", 0)
                        for r in rows
                        if r.get("probe_kind") == "cath" and r.get("cath_level") == lvl
                    ),
                    default=0,
                )
                ax.text(
                    0.02,
                    0.97,
                    f"in-vocab n={cath_iv}",
                    transform=ax.transAxes,
                    fontsize=7,
                    va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85
                    ),
                )

    # Invert each "lower is better" row exactly once (sharey="row" toggles
    # the shared axis state per call).
    for row, (_title, _ylab, _probe, _field, lower, _lvl) in enumerate(PROBES):
        if lower:
            axes[row, 0].invert_yaxis()

    # Legend (de-dupe across panels)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = set()
    h2 = []
    l2 = []
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen.add(lbl)
            h2.append(h)
            l2.append(lbl)
    axes[0, 0].legend(h2, l2, loc="lower right", fontsize=8)

    fig.suptitle(
        "n=128 leakage decomposition — convention: blue=baseline, red=L4-REPA, green=L9-REPA; "
        "circle=GearNet target, triangle=MPNN target. Pretrained NVIDIA 60M (blue dashed ref).\n"
        "dirty = both probe-fit & eval leaky · cleantrain = probe-fit filtered, eval leaky · "
        "cleanval = eval filtered, probe-fit leaky",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / "n128_leakage_decomp.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
