"""Generate tables/table_rep_quality_afdb.tex --- the AFDB counterpart to the
PDB representation-quality table (Ch6 S6.2.1, Table~\\ref{tab:proteina-rep}).

Same recipe as that hand-built table: best-layer linear probe, n_train=1000,
seed 42, mean over a converged training-step window, Delta-from-baseline, with
green/darker-green marking improvement/best-per-probe. AFDB-TRAINED models;
probes read on the cross-database blinded set (n256_xclean_pdb_afdb), which is
the only place AFDB probes exist at n_train=1000.

ALL variants with n1000 data in the window are included automatically, so the
table grows as the n1000 probe eval catches up on later AFDB checkpoints. With
WINDOW=(700,1200) the GearNet-L9 and random-control rows are absent: GN-L9's
n1000 probes currently stop at 600K (700-1000K exist only at n5000) and the
random control trained only to 500K.

Run from repo root:  python docs/masters-report/tables/scripts/make_rep_quality_afdb.py
"""

import csv
import re
import os
from collections import defaultdict
from statistics import mean

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_REP = f"{ROOT}/evaluation/proteina/representation/results/paper"
# n1000 rows live in the main dir (early ckpts) + the _n1000_compare dir (tail
# ckpts re-evaluated 2026-06-02: GN-L9 700-1000K, GN-L4 1300K, baseline
# 1700-1800K, L4-random 100-500K). Union both; (run,step) sets are disjoint.
REP_CSVS = [
    f"{_REP}/n256_xclean_pdb_afdb/pretrained_sweep_results.csv",
    f"{_REP}/_n1000_compare/n256_xclean_pdb_afdb/pretrained_sweep_results.csv",
]


def _rep_rows():
    for path in REP_CSVS:
        if os.path.exists(path):
            yield from csv.DictReader(open(path))


OUT = f"{ROOT}/docs/masters-report/tables/table_rep_quality_afdb.tex"

N_TRAIN = "1000"
WINDOW = (700, 1200)  # k steps; matches the PDB table's converged window
_STEP = re.compile(r"_step(\d+)k$")

# canonical row order + display labels (baseline first); auto-skipped if no data
VARIANTS = [
    ("baseline_afdb_256", "Baseline"),
    ("repa_l4_afdb_256", "REPA L4-GN"),
    ("repa_l9_afdb_256", "REPA L9-GN"),
    ("repa_l4_afdb_256_random", "REPA L4-random"),
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


def window_mean(probe_kind, col, higher_better, cath_level):
    """best-layer value per (family,step), then mean over the window, per family."""
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
        if s is None or not v or not (WINDOW[0] <= s <= WINDOW[1]):
            continue
        try:
            per_step[(fam(run), s)].append(float(v))
        except ValueError:
            continue
    best = {k: (max(vs) if higher_better else min(vs)) for k, vs in per_step.items()}
    out = defaultdict(list)
    for (family, _), v in best.items():
        out[family].append(v)
    return {family: mean(vs) for family, vs in out.items()}


# probe label -> {family: windowed value}
data = {pl: window_mean(pk, col, hb, cl) for pl, pk, col, hb, cl in PROBES}
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
    if not improved:
        return body
    macro = "gb" if family == col_best_family else "gd"
    return f"\\{macro}{{{body}}}"


# best improver per probe column (for \gb)
col_best = {}
for pl, pk, col, hb, cl in PROBES:
    best_fam, best_imp = None, 0.0
    for key, _ in present:
        if key == "baseline_afdb_256" or key not in data[pl] or base[pl] is None:
            continue
        d = data[pl][key] - base[pl]
        imp = d if hb else -d
        if imp > best_imp:
            best_imp, best_fam = imp, key
    col_best[pl] = best_fam

lines = [
    "% Auto-generated by tables/scripts/make_rep_quality_afdb.py --- do not hand-edit."
]
lines.append("\\begin{table}[tbp]")
lines.append("\\centering")
lines.append("\\small")
lines.append("\\setlength{\\tabcolsep}{6pt}")
lines.append("\\begin{tabular}{l|ccccc}")
lines.append("\\toprule")
lines.append(" & & & \\multicolumn{3}{c}{\\textbf{CATH top-1 acc.\\,$\\uparrow$}} \\\\")
lines.append("\\cmidrule(lr){4-6}")
lines.append(
    "\\textbf{Variant} & \\textbf{IF top-1\\,$\\uparrow$} & \\textbf{dihedral MAE\\,(\\,$^\\circ$\\,)\\,$\\downarrow$} & C & A & T \\\\"
)
lines.append("\\midrule")
for key, disp in present:
    if key == "baseline_afdb_256":
        cells = [fmt_abs(pl, base[pl]) for pl, *_ in [(p[0],) for p in PROBES]]
        cells = [fmt_abs(pl, base[pl]) for pl in [p[0] for p in PROBES]]
    else:
        cells = []
        for pl, pk, col, hb, cl in PROBES:
            cells.append(fmt_delta(pl, data[pl].get(key), hb, col_best[pl], key))
    lines.append(f"{disp} & " + " & ".join(cells) + " \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append(
    "\\caption{\\textbf{On AFDB, REPA's representation gain concentrates on fold structure.} "
    "The AFDB counterpart to Table~\\ref{tab:proteina-rep}. REPA-GearNet lifts the CATH fold "
    "probes here as on PDB, so the fold routing replicates. The per-residue picture does not: "
    "inverse-folding accuracy is flat, and the dihedral probe shows no consistent REPA effect "
    "(it tracks a baseline that itself swings several degrees between checkpoints), unlike the "
    "clear MPNN-led per-residue gain on PDB. At this checkpoint density the per-residue reads are "
    "noise-dominated, so we lean on the fold result. ($n{\\le}256$ AFDB-trained; best-layer "
    "linear probe at $n_\\text{train}{=}1000$ on the cross-database blinded set; mean over the "
    f"{WINDOW[0]}K--{WINDOW[1]/1000:.1f}M window; $\\Delta$ from baseline; "
    "\\colorbox{green!20}{green}/\\colorbox{green!42}{darker} = improvement/best per probe. "
    "The random control is omitted: it trained only to 500K, below this window.)}"
)
lines.append("\\label{tab:proteina-rep-afdb}")
lines.append("\\end{table}")

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
