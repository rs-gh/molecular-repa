"""Generate tables/table_rep_quality_afdb.tex --- the AFDB counterpart to the
PDB representation-quality table (Ch6 S6.2.1, Table~\\ref{tab:proteina-rep}).

Same recipe as that hand-built table: best-layer linear probe, n_train=1000,
seed 42, a single 1.0M checkpoint, Delta-from-baseline, with green/darker-green
marking improvement/best-per-probe. AFDB-TRAINED models; probes read on the
cross-database blinded set (n256_xclean_pdb_afdb), which is the only place AFDB
probes exist at n_train=1000.

Each variant is read at a single 1.0M checkpoint to mirror the PDB tables. A
variant with no 1.0M checkpoint falls back to its latest earlier checkpoint;
only the random control does so (it trained only to 500K), and its row is tagged.

Run from repo root:  python docs/masters-report/tables/scripts/make_rep_quality_afdb.py
"""

import csv
import re
import os
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_REP = f"{ROOT}/evaluation/proteina/representation/results/paper"
# n1000 rows live in the main dir (early ckpts) + the _n1000_compare dir (tail
# ckpts re-evaluated 2026-06-02: L9-GearNet 700-1000K, L4-GearNet 1300K, baseline
# 1700-1800K, L4-random 100-500K). Union both; (run,step) sets are disjoint.
REP_CSVS = [
    f"{_REP}/n256_xclean_pdb_afdb/pretrained_sweep_results.csv",
    f"{_REP}/_n1000_compare/n256_xclean_pdb_afdb/pretrained_sweep_results.csv",
]


def _rep_rows():
    for path in REP_CSVS:
        if os.path.exists(path):
            yield from csv.DictReader(open(path))


OUT = f"{ROOT}/docs/masters-report/tables/table_rep_quality_afdb_data.tex"

N_TRAIN = "1000"
TARGET_K = 1000  # single checkpoint, mirroring the PDB tables (Tab 6.2 / A.3)
_STEP = re.compile(r"_step(\d+)k$")

# canonical row order + display labels (baseline first, then random control, as
# in the PDB tables); auto-skipped if no data
VARIANTS = [
    ("baseline_afdb_256", "Baseline"),
    ("repa_l4_afdb_256_random", "REPA L4-random"),
    ("repa_l4_afdb_256", "REPA L4-GearNet"),
    ("repa_l9_afdb_256", "REPA L9-GearNet"),
    ("repa_mpnn_l4_afdb_256", "REPA L4-MPNN"),
    ("repa_mpnn_l9_afdb_256", "REPA L9-MPNN"),
]
# (column label, probe_kind, col, higher_better, cath_level)
PROBES = [
    ("IF", "inverse_folding", "if_top1_acc", True, None),
    ("dih", "dihedral", "dih_mae_total_deg", False, None),
    ("C", "cath", "cath_accuracy", True, "C"),
    ("A", "cath", "cath_accuracy", True, "A"),
    ("T", "cath", "cath_accuracy", True, "T"),
]


def fam(run):
    return _STEP.sub("", run)


def step_k(run):
    m = _STEP.search(run)
    return int(m.group(1)) if m else None


def best_layer_per_step(probe_kind, col, higher_better, cath_level):
    """best-layer value per (family, step)."""
    per_step = defaultdict(list)  # (family, step) -> [layer values]
    for r in _rep_rows():
        run = r.get("run", "")
        if "afdb" not in run or str(r.get("n_train")) != N_TRAIN:
            continue
        if (r.get("probe_seed") or "42") != "42" or r.get("probe_kind") != probe_kind:
            continue
        if cath_level and r.get("cath_level") != cath_level:
            continue
        s, v = step_k(run), r.get(col)
        if s is None or not v:
            continue
        try:
            per_step[(fam(run), s)].append(float(v))
        except ValueError:
            continue
    return {k: (max(vs) if higher_better else min(vs)) for k, vs in per_step.items()}


# probe label -> {(family, step): best-layer value}
_per_step = {pl: best_layer_per_step(pk, col, hb, cl) for pl, pk, col, hb, cl in PROBES}

# Each variant is read at one checkpoint: 1.0M where it exists (every trained
# variant), else the family's latest earlier checkpoint. Only the random control
# falls back --- it stopped at 500K --- and its row is tagged in the table.
_steps_avail = defaultdict(set)
for pl in _per_step:
    for family, s in _per_step[pl]:
        _steps_avail[family].add(s)
chosen_step = {
    family: (TARGET_K if TARGET_K in steps else max(steps))
    for family, steps in _steps_avail.items()
}

# probe label -> {family: value at the family's chosen step}
data = {
    pl: {
        family: _per_step[pl][(family, chosen_step[family])]
        for family in _steps_avail
        if (family, chosen_step[family]) in _per_step[pl]
    }
    for pl in _per_step
}
present = [(key, disp) for key, disp in VARIANTS if any(key in data[pl] for pl in data)]
base = {pl: data[pl].get("baseline_afdb_256") for pl in data}


def fmt_abs(pl, val):
    return (
        "--"
        if val is None
        else (f"{val:.3f}" if pl in ("IF", "C", "A", "T") else f"{val:.1f}")
    )


# below-epsilon changes are noise; leave them uncoloured (as the PDB table does)
EPS = {"IF": 0.005, "dih": 0.5, "C": 0.01, "A": 0.01, "T": 0.01}

# The per-residue probes (IF, dihedral) are noise-dominated on the single-seed,
# single-checkpoint AFDB blinded set (see caption / App.~rep-afdb), so we draw no
# per-residue conclusion and never colour these columns --- colouring would imply a
# trustworthy "best" we explicitly disclaim.
NO_COLOR = {"IF", "dih"}


def fmt_delta(pl, val, hb, col_best_family, family):
    if val is None or base[pl] is None:
        return "--"
    d = val - base[pl]
    improved = ((d > 0) if hb else (d < 0)) and abs(d) >= EPS[pl]
    mag = abs(d)
    body = (
        f"{'+' if d >= 0 else '$-$'}{mag:.3f}"
        if pl in ("IF", "C", "A", "T")
        else f"{'+' if d >= 0 else '$-$'}{mag:.1f}"
    )
    if not improved or pl in NO_COLOR:
        return body
    macro = "gb" if family == col_best_family else "gd"
    return f"\\{macro}{{{body}}}"


# best improver per probe column (for \gb). Only families read at the common
# 1.0M checkpoint compete --- the off-step random control (500K) would otherwise
# claim the "best" marker on a checkpoint nobody else is measured at.
col_best = {}
for pl, pk, col, hb, cl in PROBES:
    best_fam, best_imp = None, 0.0
    for key, _ in present:
        if key == "baseline_afdb_256" or key not in data[pl] or base[pl] is None:
            continue
        if chosen_step.get(key) != TARGET_K:
            continue
        d = data[pl][key] - base[pl]
        imp = d if hb else -d
        if imp > best_imp:
            best_imp, best_fam = imp, key
    col_best[pl] = best_fam

lines = [
    "% Numbers generated by tables/scripts/make_rep_quality_afdb.py --- safe to overwrite.",
    "% The caption and table environment are hand-maintained in table_rep_quality_afdb.tex.",
]
lines.append("\\begin{tabular}{lccccc}")
lines.append("\\toprule")
lines.append(" & & & \\multicolumn{3}{c}{\\textbf{CATH top-1 acc.\\,$\\uparrow$}} \\\\")
lines.append("\\cmidrule(lr){4-6}")
lines.append(
    "\\textbf{Variant} & \\textbf{IF top-1\\,$\\uparrow$} & \\textbf{dihedral MAE\\,(\\,$^\\circ$\\,)\\,$\\downarrow$} & C & A & T \\\\"
)
lines.append("\\midrule")
for key, disp in present:
    # tag any row read off a checkpoint other than 1.0M (only the random control)
    label = disp + ("$^{\\ddagger}$" if chosen_step.get(key) != TARGET_K else "")
    if key == "baseline_afdb_256":
        cells = [fmt_abs(pl, base[pl]) for pl in [p[0] for p in PROBES]]
    else:
        cells = []
        for pl, pk, col, hb, cl in PROBES:
            cells.append(fmt_delta(pl, data[pl].get(key), hb, col_best[pl], key))
    lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Saved {OUT}")
print("variants in window:", [d for _, d in present])
for pl in [p[0] for p in PROBES]:
    print(
        f"  {pl:4} baseline={base[pl]}  "
        + "  ".join(
            f"{k.split('_')[1]}:{data[pl][k]:.3f}"
            for k in data[pl]
            if k != "baseline_afdb_256"
        )
    )
