"""n256 leakage decomposition with 4 model lines + pretrained reference,
across 4 leakage regimes (dirty | cleantrain | cleanval | xclean AFDB).

Models compared:
  - baseline_256_bs24_2gpu          (gray)
  - repa_l4_256_per_residue_bs24_2gpu   = REPA L4 GearNet  (blue)
  - repa_mpnn_l4_256_per_residue                (orange)
  - repa_mpnn_l9_256_per_residue                (green)
  - pretrained_dfs_60m_n256_paper   horizontal red dashed reference
  - baseline_afdb_256 (xclean column only)  purple — AFDB-trained, no
    distribution shift on AFDB val: serves as the "input distribution-shift
    reference" so we can see how much PDB-trained models lose by being
    evaluated on AF2-predicted structures.

Columns:
  dirty       — both probe-fit & eval leaky (n_eval=3190)
  cleantrain  — probe-fit homology-filtered, eval leaky (n_eval=3190)
  cleanval    — probe-fit leaky, eval homology-filtered (n_eval=72; CATH n~20)
  xclean-AFDB — probe-fit on AFDB train, eval = AFDB val with NO ≥30% hit in
                EITHER PDB train OR AFDB train (n_eval=325; both leakage paths
                cleaned). Cross-DB structural shift caveat for PDB-trained models.

Built 2026-05-24.
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
FIG = REPO / "evaluation/proteina/representation/figures/paper/leakage_decomp/n256/pdb"
FIG.mkdir(parents=True, exist_ok=True)

CONDS = [
    ("dirty", RES / "n256_convergence_cath_if_dih_pdb", 3190),
    ("cleantrain", RES / "n256_convergence_cleantrain_pdb", 3190),
    ("cleanval", RES / "n256_convergence_cleanval_pdb", 72),
    ("xclean-AFDB", RES / "n256_xclean_afdb_pdb", 325),
]
# Convention: blue=baseline, red=L4 REPA, green=L9 REPA,
#             circle=GearNet target, triangle=MPNN target.
# baseline + pretrained have no encoder so use square markers.
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
    "baseline": re.compile(r"^baseline_256_bs24_2gpu(_step\d+k)?$"),
    "repa_l4_gearnet": re.compile(r"^repa_l4_256_per_residue_bs24_2gpu(_step\d+k)?$"),
    "repa_l4_mpnn": re.compile(r"^repa_mpnn_l4_256_per_residue(_step\d+k)?$"),
    "repa_l4_random": re.compile(
        r"^repa_l4_256_per_residue_random_bs24_2gpu(_step\d+k)?$"
    ),
    "repa_l9_gearnet": re.compile(r"^repa_l9_256_per_residue_bs24_2gpu(_step\d+k)?$"),
    "repa_l9_mpnn": re.compile(r"^repa_mpnn_l9_256_per_residue(_step\d+k)?$"),
    "baseline_afdb": re.compile(r"^baseline_afdb_256(_step\d+k)?$"),
}
# baseline_afdb is only meaningful in the xclean-AFDB column.
XCLEAN_ONLY = {"baseline_afdb"}
PRETRAINED = "pretrained_dfs_60m_n256_paper"


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
            p_val = best_layer_single(
                rows, PRETRAINED, probe, field, lower=lower, lvl=lvl
            )
            if p_val is not None:
                # NVIDIA pretrained 60M: square marker (baseline-like, no
                # encoder), drawn as horizontal dashed reference.
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
            # invert_yaxis on a shared-y row TOGGLES the shared state on every
            # call, so calling it inside the column loop double-flips when the
            # column count is even. Defer to a single call per row below.
            if lvl is not None and cond in ("cleanval", "xclean-AFDB"):
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
    # Invert each "lower is better" row exactly once (sharey="row" means each
    # call toggles the shared axis state, so a per-panel call would silently
    # cancel out for an even column count).
    for row, (_title, _ylab, _probe, _field, lower, _lvl) in enumerate(PROBES):
        if lower:
            axes[row, 0].invert_yaxis()

    # de-duped legend (drawn from the xclean column which has all line types)
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
        "n=256 leakage decomposition across 4 regimes — convention: "
        "blue=baseline, red=L4-REPA, green=L9-REPA; circle=GearNet target, triangle=MPNN target. "
        "Pretrained NVIDIA 60M (blue dashed ref). AFDB-trained baseline (blue dotted diamond, xclean only).",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / "n256_leakage_decomp.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
