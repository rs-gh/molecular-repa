"""Generate tables/table_rep_quality_full.tex --- the FULL six-variant PDB
representation-quality ablation (Appendix). Extends the four-row main-text
Table~\\ref{tab:proteina-rep} with the remaining encoder x depth combinations
(L4-GearNet, L4-MPNN), so a reader following the Ch6 "remaining combinations" and
"trained-L4 beats random-L4" pointers lands on a complete table.

Same recipe as the AFDB sibling (make_rep_quality_afdb.py) and the main PDB
table: best-layer linear probe, n_train=1000, seed 42, mean over the 700K-1.2M
converged window, Delta-from-baseline, green/darker-green = improvement/best.
PDB-TRAINED models; IF/dihedral read on the cross-database blinded set
(n256_xclean_afdb_pdb), CATH on the leakage-controlled cleantrain set
(n256_convergence_cleantrain_pdb) -- exactly the two sources the main-text
table cites. The baseline absolutes reproduce that table to printed digits.

Run from repo root:  python docs/masters-report/tables/scripts/make_rep_quality_full.py
"""

import csv
import os
import re
from collections import defaultdict
from statistics import mean

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_REP = f"{ROOT}/evaluation/proteina/representation/results/paper"

# Per-probe sources: IF/dihedral from the cross-DB blinded (xclean) set; CATH
# from the cleantrain set. Union the main dir with the _n1000_compare tail dir.
CSVS_XCLEAN = [
    f"{_REP}/n256_xclean_afdb_pdb/pretrained_sweep_results.csv",
    f"{_REP}/_n1000_compare/n256_xclean_afdb/pretrained_sweep_results.csv",
]
CSVS_CLEANTRAIN = [
    f"{_REP}/n256_convergence_cleantrain_pdb/pretrained_sweep_results.csv",
    f"{_REP}/_n1000_compare/n256_cleantrain_pdb/pretrained_sweep_results.csv",
]

OUT = f"{ROOT}/docs/masters-report/tables/table_rep_quality_full.tex"

N_TRAIN = "1000"
WINDOW = (700, 1200)  # k steps
_STEP = re.compile(r"_step(\d+)k$")

VARIANTS = [
    ("baseline_256_bs24_2gpu", "Baseline"),
    ("repa_l4_256_per_residue_random_bs24_2gpu", "REPA L4-random"),
    ("repa_l4_256_per_residue_bs24_2gpu", "REPA L4-GearNet"),
    ("repa_mpnn_l4_256_per_residue", "REPA L4-MPNN"),
    ("repa_l9_256_per_residue_bs24_2gpu", "REPA L9-GearNet"),
    ("repa_mpnn_l9_256_per_residue", "REPA L9-MPNN"),
]
# (label, probe_kind, col, higher_better, cath_level, csv_list)
PROBES = [
    ("IF", "inverse_folding", "if_top1_acc", True, None, CSVS_XCLEAN),
    ("dih", "dihedral", "dih_mae_total_deg", False, None, CSVS_XCLEAN),
    ("C", "cath", "cath_accuracy", True, "C", CSVS_CLEANTRAIN),
    ("A", "cath", "cath_accuracy", True, "A", CSVS_CLEANTRAIN),
    ("T", "cath", "cath_accuracy", True, "T", CSVS_CLEANTRAIN),
]


def _rows(csvs):
    for path in csvs:
        if os.path.exists(path):
            yield from csv.DictReader(open(path))


def fam(run):
    return _STEP.sub("", run)


def step_k(run):
    m = _STEP.search(run)
    return int(m.group(1)) if m else None


def window_mean(probe_kind, col, higher_better, cath_level, csvs):
    per_step = defaultdict(list)
    for r in _rows(csvs):
        run = r.get("run", "")
        if "afdb" in run or str(r.get("n_train")) != N_TRAIN:
            continue  # PDB-trained only
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


data = {pl: window_mean(pk, col, hb, cl, cs) for pl, pk, col, hb, cl, cs in PROBES}
present = [(k, d) for k, d in VARIANTS if any(k in data[pl] for pl in data)]
BASE = "baseline_256_bs24_2gpu"
base = {pl: data[pl].get(BASE) for pl in data}

EPS = {"IF": 0.005, "dih": 0.5, "C": 0.01, "A": 0.01, "T": 0.01}


def fmt_abs(pl, val):
    return (
        "--"
        if val is None
        else (f"{val:.3f}" if pl in ("IF", "C", "A", "T") else f"{val:.1f}")
    )


def fmt_delta(pl, val, hb, best_fam, family):
    if val is None or base[pl] is None:
        return "--"
    d = val - base[pl]
    improved = ((d > 0) if hb else (d < 0)) and abs(d) >= EPS[pl]
    body = (
        f"{'+' if d >= 0 else '$-$'}{abs(d):.3f}"
        if pl in ("IF", "C", "A", "T")
        else f"{'+' if d >= 0 else '$-$'}{abs(d):.1f}"
    )
    if not improved:
        return body
    return f"\\{'gb' if family == best_fam else 'gd'}{{{body}}}"


col_best = {}
for pl, pk, col, hb, cl, cs in PROBES:
    bf, bi = None, 0.0
    for k, _ in present:
        if k == BASE or k not in data[pl] or base[pl] is None:
            continue
        imp = (data[pl][k] - base[pl]) * (1 if hb else -1)
        if imp > bi:
            bi, bf = imp, k
    col_best[pl] = bf

L = [
    "% Auto-generated by tables/scripts/make_rep_quality_full.py --- do not hand-edit."
]
L += [
    "\\begin{table}[tbp]",
    "\\centering",
    "\\small",
    "\\setlength{\\tabcolsep}{6pt}",
    "\\begin{tabular}{lccccc}",
    "\\toprule",
    " & & & \\multicolumn{3}{c}{\\textbf{CATH top-1 acc.\\,$\\uparrow$}} \\\\",
    "\\cmidrule(lr){4-6}",
    "\\textbf{Variant} & \\textbf{IF top-1\\,$\\uparrow$} & \\textbf{dihedral MAE\\,(\\,$^\\circ$\\,)\\,$\\downarrow$} & C & A & T \\\\",
    "\\midrule",
]
for k, disp in present:
    if k == BASE:
        cells = [fmt_abs(pl, base[pl]) for pl in [p[0] for p in PROBES]]
    else:
        cells = [
            fmt_delta(pl, data[pl].get(k), hb, col_best[pl], k)
            for pl, pk, col, hb, cl, cs in PROBES
        ]
    L.append(f"{disp} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
L.append(
    "\\caption{\\textbf{Full PDB representation-quality ablation across all six variants.} "
    "The remaining encoder$\\times$depth combinations behind the four-row main-text "
    "Table~\\ref{tab:proteina-rep}. The encoder-routed pattern holds throughout: ProteinMPNN "
    "wins the per-residue probes (IF, dihedral), GearNet wins the fold probes (CATH C/A/T). "
    "At matched layer~4, the trained GearNet (L4-GearNet) beats the random control (L4-random) on "
    "every probe, so the learned-versus-random gap is not an artefact of injection depth. "
    "(Baseline absolute; REPA rows are $\\Delta$-from-baseline. Best-layer linear probe, "
    "$n_\\text{train}{=}1000$, seed 42, mean over the "
    f"{WINDOW[0]}K--{WINDOW[1]/1000:.1f}M window; IF/dihedral on the cross-database blinded set, "
    "CATH on the cleantrain set. \\colorbox{green!20}{green}/\\colorbox{green!42}{darker} = "
    "improvement/best per probe. In-window step coverage is uneven across rows.)}"
)
L += ["\\label{tab:proteina-rep-full}", "\\end{table}"]

with open(OUT, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"Saved {OUT}")
print("variants:", [d for _, d in present])
for pl in [p[0] for p in PROBES]:
    print(
        f"  {pl:4} base={base[pl]}  "
        + "  ".join(
            f"{k.split('_')[1] if '_' in k else k}:{data[pl][k]:.3f}"
            for k in data[pl]
            if k != BASE
        )
    )
