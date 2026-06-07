"""Generate tables/table_genrep_corr.tex --- representation-quality x
generation-quality correlations across training checkpoints (PDB, n=256).

Each point is a (run, step) checkpoint, pooled over baseline, the REPA variants,
and the random-target control --- the same population that defines the rep-gen
band in Fig 6.4. For each (rep metric, gen metric) pair we report the partial
Pearson correlation controlling for training step (raw Pearson in parentheses),
so the headline number is not just the shared "everything improves with training"
trend.

Orientation: correlations are computed on quality-oriented values (error metrics
negated), so a POSITIVE number always means "better representation accompanies
better generation". The column-max is bold to show the encoder-matched routing.

Run from repo root:  python docs/masters-report/tables/scripts/make_genrep_corr.py
"""

import csv
import json
import re
import os
from collections import defaultdict
from statistics import mean

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_REP = f"{ROOT}/evaluation/proteina/representation/results/paper"
# Report standardised on n_train=1000 (2026-06-02). The regime CSV holds early
# checkpoints at n=1000; the frontier "tail" checkpoints were re-run at n=1000
# into the separate _n1000_compare/ dir (additive — 5000 rows untouched). Read
# BOTH and filter to n_train==1000, seed 42, so the correlation pools the same
# single-protocol values the report figures plot.
N_TRAIN = "1000"
REP_X = [
    f"{_REP}/n256_xclean_afdb_pdb/pretrained_sweep_results.csv",
    f"{_REP}/_n1000_compare/n256_xclean_afdb/pretrained_sweep_results.csv",
]
REP_CT = [
    f"{_REP}/n256_convergence_cleantrain_pdb/pretrained_sweep_results.csv",
    f"{_REP}/_n1000_compare/n256_cleantrain_pdb/pretrained_sweep_results.csv",
]
GEN = f"{ROOT}/evaluation/proteina/generation/results/paper/n256_convergence_pdb/sweep_results.clean.jsonl"
OUT = f"{ROOT}/docs/masters-report/tables/table_genrep_corr.tex"


def fam(run):
    return re.sub(r"_step\d+k$", "", run)


def best_layer(csvpaths, probe_kind, col, higher_better, cath_level=None):
    """best-layer value per (family, step), n_train==1000 & seed 42 only.

    Accepts one path or a list (regime + compare); rows are pooled.
    """
    if isinstance(csvpaths, str):
        csvpaths = [csvpaths]
    agg = defaultdict(list)
    for csvpath in csvpaths:
        if not os.path.exists(csvpath):
            continue
        with open(csvpath) as f:
            for r in csv.DictReader(f):
                if str(r.get("n_train")) != N_TRAIN:
                    continue
                if (r.get("probe_seed") or "42") != "42":
                    continue
                if r.get("probe_kind") != probe_kind:
                    continue
                if cath_level and r.get("cath_level") != cath_level:
                    continue
                v = r.get(col)
                if not v:
                    continue
                try:
                    v = float(v)
                    step = int(r["step"])
                except Exception:
                    continue
                agg[(fam(r["run"]), step)].append(v)
    return {k: (max(vs) if higher_better else min(vs)) for k, vs in agg.items()}


def gen_metric(key):
    agg = defaultdict(list)
    with open(GEN) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            v = r.get(key)
            if v is None:
                continue
            agg[(fam(r["run"]), r["step"])].append(v)
    return {k: mean(vs) for k, vs in agg.items()}


def pearson(xs, ys):
    mx, my = mean(xs), mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs) ** 0.5
    vy = sum((b - my) ** 2 for b in ys) ** 0.5
    return cov / (vx * vy) if vx * vy else float("nan")


def partial(xs, ys, zs):
    """partial corr of x,y controlling for z (first order)."""
    rxy, rxz, ryz = pearson(xs, ys), pearson(xs, zs), pearson(ys, zs)
    denom = ((1 - rxz**2) * (1 - ryz**2)) ** 0.5
    return (rxy - rxz * ryz) / denom if denom else float("nan")


# (label, dict, higher_better)
REP = [
    (
        "Inverse folding (top-1)",
        best_layer(REP_X, "inverse_folding", "if_top1_acc", True),
        True,
    ),
    ("Dihedral MAE", best_layer(REP_X, "dihedral", "dih_mae_total_deg", False), False),
    ("CATH-A", best_layer(REP_CT, "cath", "cath_accuracy", True, cath_level="A"), True),
]
GEN_M = [
    ("FPSD", gen_metric("_res_PDB_FID"), False),  # display label is FPSD; data key unchanged
    ("Designability", gen_metric("_res_designability_rate"), True),
]


def oriented(d, higher_better):
    return d if higher_better else {k: -v for k, v in d.items()}


# Compute cells: list of rows; each row list of (partial, raw, n).
cells = []
ns = []
for _, rdict, rhb in REP:
    ro = oriented(rdict, rhb)
    row = []
    for _, gdict, ghb in GEN_M:
        go = oriented(gdict, ghb)
        keys = sorted(set(ro) & set(go))
        xs = [ro[k] for k in keys]
        ys = [go[k] for k in keys]
        zs = [k[1] for k in keys]  # step
        row.append((partial(xs, ys, zs), pearson(xs, ys), len(keys)))
    cells.append(row)
    ns.append(min(r[2] for r in row))

n_all = sorted(set(n for n in ns))
n_str = str(n_all[0]) if len(n_all) == 1 else f"{min(n_all)}--{max(n_all)}"

lines = []
lines.append(
    "% Auto-generated by tables/scripts/make_genrep_corr.py --- do not hand-edit."
)
lines.append("\\begin{table}[tbp]")
lines.append("\\centering")
lines.append("\\small")
lines.append("\\begin{tabular}{l" + " cc" * len(GEN_M) + "}")
lines.append("\\toprule")
# Grouped header: each generation metric spans a (partial, raw) pair.
group_head = ["\\textbf{Representation quality}"]
group_head += [f"\\multicolumn{{2}}{{c}}{{\\textbf{{{g[0]}}}}}" for g in GEN_M]
lines.append(" & ".join(group_head) + " \\\\")
lines.append(
    " ".join(f"\\cmidrule(lr){{{2 + 2 * j}-{3 + 2 * j}}}" for j in range(len(GEN_M)))
)
sub_head = [""] + ["partial & raw"] * len(GEN_M)
lines.append(" & ".join(sub_head) + " \\\\")
lines.append("\\midrule")
for i, (rlabel, _, _) in enumerate(REP):
    parts = []
    for j in range(len(GEN_M)):
        p, raw, _ = cells[i][j]
        parts.append(f"{p:+.2f} & {raw:+.2f}")
    lines.append(f"{rlabel} & " + " & ".join(parts) + " \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append(
    "\\caption{\\textbf{Better trunk representations track better generation --- "
    "strongly for designability, moderately for FPSD.} Partial Pearson correlation "
    "controlling for training step, with the raw value alongside; oriented so a positive "
    "value means better representation accompanies better generation. ($n{=}"
    + n_str
    + "$ checkpoints pooled over the baseline, the REPA variants, and the random "
    "control.)}"
)
lines.append("\\label{tab:proteina-genrep-corr}")
lines.append("\\end{table}")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")

# stdout summary for our reference.
print(f"Saved {OUT}  (n={n_str})")
print(f"{'rep / gen':24}", *[f"{g[0]:>22}" for g in GEN_M])
for i, (rlabel, _, _) in enumerate(REP):
    print(
        f"{rlabel:24}",
        *[
            f"  p{cells[i][j][0]:+.2f} raw{cells[i][j][1]:+.2f}"
            for j in range(len(GEN_M))
        ],
    )
