"""Compare n_train=1000 vs n_train=5000 for the 29 "tail" checkpoints.

Non-destructive read-only analysis. Reads:
  - n=5000 (existing): the regime CSVs (seed 42, n_train=5000 rows = the tails)
  - n=1000 (new):      the _n1000_compare/ CSVs produced by the compare profiles

For each of the 3 plotted report metrics it applies the SAME best-layer
reduction the report figure uses (fig2_representation.best_layer), then joins
on (run, step) and reports the per-checkpoint delta + summary stats.

Run AFTER consolidating the compare dirs:
  python .../pretrain_probe_sweep.py --config n256_cath_if_dih_cleantrain_pdb_n1000_compare --consolidate_only
  python .../pretrain_probe_sweep.py --config n256_cath_if_dih_xclean_afdb_pdb_n1000_compare  --consolidate_only
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(
    "/home/sr2173/git/molecular-repa/evaluation/proteina/representation/results/paper"
)

# (label, n5000_csv, n1000_csv, probe_kind, value_col, cath_level, higher_better)
SELECTIONS = [
    (
        "IF top-1  (xclean)",
        ROOT / "n256_xclean_afdb_pdb/pretrained_sweep_results.csv",
        ROOT / "_n1000_compare/n256_xclean_afdb/pretrained_sweep_results.csv",
        "inverse_folding",
        "if_top1_acc",
        None,
        True,
    ),
    (
        "Dihedral MAE (xclean) [deg]",
        ROOT / "n256_xclean_afdb_pdb/pretrained_sweep_results.csv",
        ROOT / "_n1000_compare/n256_xclean_afdb/pretrained_sweep_results.csv",
        "dihedral",
        "dih_mae_total_deg",
        None,
        False,
    ),
    (
        "CATH-A (cleantrain)",
        ROOT / "n256_convergence_cleantrain_pdb/pretrained_sweep_results.csv",
        ROOT / "_n1000_compare/n256_cleantrain_pdb/pretrained_sweep_results.csv",
        "cath",
        "cath_accuracy",
        "A",
        True,
    ),
]


def best_layer(csv_path, probe_kind, value_col, level, higher, n_train):
    """{(run, step): best-layer value} at seed 42 and the given n_train."""
    pick = max if higher else min
    by = defaultdict(list)
    if not Path(csv_path).exists():
        return {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("probe_kind") != probe_kind:
                continue
            if level is not None and r.get("cath_level") != level:
                continue
            if (r.get("probe_seed") or "42") != "42":
                continue
            if str(r.get("n_train")) != str(n_train):
                continue
            v = r.get(value_col)
            if not v:
                continue
            try:
                by[(r["run"], int(r["step"]))].append(float(v))
            except ValueError:
                continue
    return {k: pick(vs) for k, vs in by.items()}


def fam(run):
    return re.sub(r"_step\d+k$", "", run)


for label, c5, c1, pk, vc, lvl, higher in SELECTIONS:
    v5 = best_layer(c5, pk, vc, lvl, higher, 5000)
    v1 = best_layer(c1, pk, vc, lvl, higher, 1000)
    common = sorted(set(v5) & set(v1), key=lambda k: (fam(k[0]), k[1]))
    print(f"\n===== {label} =====")
    print(f"  5000 points: {len(v5)} | 1000 points: {len(v1)} | overlap: {len(common)}")
    if not common:
        print("  (no overlap yet — jobs still running / not consolidated)")
        continue
    deltas = []
    print(
        f"  {'checkpoint':52s} {'n=5000':>9} {'n=1000':>9} {'Δ(1k-5k)':>10} {'Δ%':>7}"
    )
    for k in common:
        a, b = v5[k], v1[k]
        d = b - a
        dp = 100 * d / a if a else float("nan")
        deltas.append(d)
        run = f"{fam(k[0])}@{k[1]//1000}k"
        print(f"  {run:52s} {a:9.4f} {b:9.4f} {d:+10.4f} {dp:+7.1f}")
    import statistics as st

    ad = [abs(x) for x in deltas]
    print(
        f"  --- summary: mean|Δ|={st.mean(ad):.4f}  median|Δ|={st.median(ad):.4f}  "
        f"max|Δ|={max(ad):.4f}  signed-mean Δ={st.mean(deltas):+.4f}"
    )
