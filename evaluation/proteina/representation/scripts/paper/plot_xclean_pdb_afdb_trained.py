"""AFDB-trained models: dirty (AFDB val) vs xclean-PDB (doubly-clean PDB val).

Two-column mirror of the PDB-trained leakage decomp. For AFDB-trained models:
  - dirty (left):       eval on AFDB val (in-DB, leaky) — n_eval=4521
  - xclean-PDB (right): eval on PDB val with NO ≥30% hit in EITHER PDB or
                        AFDB train. Cross-DB, both leakage paths cleaned.
                        n_eval=62.

Models (AFDB-trained, n=256):
  baseline_afdb_256                  blue solid square
  repa_l4_afdb_256_per_residue       red solid circle      (L4 GearNet)
  repa_mpnn_l4_afdb_256_per_residue  red solid triangle    (L4 MPNN)
  repa_l9_afdb_256_per_residue       green solid circle    (L9 GearNet)
  repa_mpnn_l9_afdb_256_per_residue  green solid triangle  (L9 MPNN)
PDB-trained baseline (xclean column only): blue dotted diamond
Pretrained NVIDIA 60M: blue dashed horizontal reference.

Built 2026-05-24. dirty data from n256_convergence_cath_if_dih_afdb;
xclean-PDB data from job 29647383 → n256_xclean_pdb_afdb.
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
FIG_DIR = (
    REPO / "evaluation/proteina/representation/figures/paper/leakage_decomp/n256/afdb"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONDS = [
    ("dirty", RES / "n256_convergence_cath_if_dih_afdb", 4521),
    ("xclean-PDB", RES / "n256_xclean_pdb_afdb", 62),
]

COLORS = {
    "baseline": "#1f77b4",
    "repa_l4_gearnet": "#d62728",
    "repa_l4_mpnn": "#d62728",
    "repa_l9_gearnet": "#2ca02c",
    "repa_l9_mpnn": "#2ca02c",
    "baseline_pdb": "#1f77b4",
}
MARKERS = {
    "baseline": "s",
    "repa_l4_gearnet": "o",
    "repa_l4_mpnn": "^",
    "repa_l9_gearnet": "o",
    "repa_l9_mpnn": "^",
    "baseline_pdb": "D",
}
LINESTYLES = {
    "baseline": "-",
    "repa_l4_gearnet": "-",
    "repa_l4_mpnn": "-",
    "repa_l9_gearnet": "-",
    "repa_l9_mpnn": "-",
    "baseline_pdb": ":",
}
LABELS = {
    "baseline": "baseline (AFDB)",
    "repa_l4_gearnet": "REPA L4 GearNet (AFDB)",
    "repa_l4_mpnn": "REPA L4 MPNN (AFDB)",
    "repa_l9_gearnet": "REPA L9 GearNet (AFDB)",
    "repa_l9_mpnn": "REPA L9 MPNN (AFDB)",
    "baseline_pdb": "baseline (PDB, cross-DB ref)",
}
RUN_RE = {
    "baseline": re.compile(r"^baseline_afdb_256(_step\d+k)?$"),
    "repa_l4_gearnet": re.compile(r"^repa_l4_afdb_256(_step\d+k)?$"),
    "repa_l4_mpnn": re.compile(r"^repa_mpnn_l4_afdb_256(_step\d+k)?$"),
    "repa_l9_gearnet": re.compile(r"^repa_l9_afdb_256(_step\d+k)?$"),
    "repa_l9_mpnn": re.compile(r"^repa_mpnn_l9_afdb_256(_step\d+k)?$"),
    "baseline_pdb": re.compile(r"^baseline_256_bs24_2gpu(_step\d+k)?$"),
}
# PDB-trained baseline (cross-DB reference) only exists in the xclean-PDB
# results dir. Hide in dirty.
XCLEAN_ONLY = {"baseline_pdb"}
PRETRAINED = "pretrained_dfs_60m_n256_paper"

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


def load(d):
    rows = []
    for p in sorted(d.glob("pretrained_sweep_results*.jsonl")):
        for ln in open(p):
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def best_layer(rows, run_filter, probe, field, lower=False, lvl=None):
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


def main():
    data = {name: load(p) for name, p, _ in CONDS}
    print({k: len(v) for k, v in data.items()})
    fig, axes = plt.subplots(
        len(PROBES),
        len(CONDS),
        figsize=(4.0 * len(CONDS), 3.0 * len(PROBES)),
        sharey="row",
    )
    for col, (cond, _, n_eval) in enumerate(CONDS):
        rows = data[cond]
        for row, (title, ylab, probe, field, lower, lvl) in enumerate(PROBES):
            ax = axes[row, col]
            for who, regex in RUN_RE.items():
                if who in XCLEAN_ONLY and cond != "xclean-PDB":
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
            p_val = best_layer_single(
                rows, PRETRAINED, probe, field, lower=lower, lvl=lvl
            )
            if p_val is not None:
                ax.axhline(
                    p_val,
                    color="#1f77b4",
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
            # CATH in-vocab annotation for the small xclean-PDB eval (n=62)
            # — annotation matters because the probe is restricted to
            # in-vocab classes from probe-fit; in-vocab eval n is the
            # statistically meaningful sample count.
            if cond == "xclean-PDB" and lvl is not None:
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

    # Invert MAE rows ONCE per row (sharey toggle bug).
    for row, (_t, _y, _p, _f, lower, _lvl) in enumerate(PROBES):
        if lower:
            axes[row, 0].invert_yaxis()

    # De-duped legend from the xclean column (has the cross-DB ref + pretrained).
    handles, labels = axes[0, -1].get_legend_handles_labels()
    seen = set()
    h2 = []
    l2 = []
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen.add(lbl)
            h2.append(h)
            l2.append(lbl)
    axes[0, 0].legend(h2, l2, loc="lower right", fontsize=7)

    fig.suptitle(
        "n=256 AFDB-trained models: dirty (AFDB val, in-DB leaky) vs xclean-PDB (cross-DB, doubly clean).\n"
        "convention: blue=baseline, red=L4-REPA, green=L9-REPA; circle=GearNet, triangle=MPNN. "
        "PDB-trained baseline (blue dotted diamond, xclean only) = cross-DB reference.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "n256_afdb_trained_dirty_vs_xclean_pdb.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
