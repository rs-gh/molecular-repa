"""Summarize train-vs-train cross-DB overlap.

For each (Q,T) pair and length bucket, reports % of Q-queries that have a hit
in T at identity thresholds {1.00, 0.99, 0.90, 0.70, 0.50, 0.30}.

mmseqs easy-search .m8 columns set via --format-output 'query,target,pident,evalue'.
pident in mmseqs2 is a fraction in [0,1].
"""

from collections import defaultdict
from pathlib import Path

THRESHOLDS = [100.0, 99.0, 90.0, 70.0, 50.0, 30.0]

PAIRS = [
    ("pdb_vs_afdb_le128", "pdb_train_le128.fasta", "PDB→AFDB", 128),
    ("afdb_vs_pdb_le128", "afdb_train_le128.fasta", "AFDB→PDB", 128),
    ("pdb_vs_afdb_le256", "pdb_train_le256.fasta", "PDB→AFDB", 256),
    ("afdb_vs_pdb_le256", "afdb_train_le256.fasta", "AFDB→PDB", 256),
]


def count_q(fasta):
    return sum(1 for line in open(fasta) if line.startswith(">"))


def best_pident_per_query(m8):
    best = defaultdict(float)
    with open(m8) as f:
        for line in f:
            q, t, pid, ev = line.rstrip().split("\t")
            pid = float(pid)
            if pid > best[q]:
                best[q] = pid
    return best


ROOT = Path(__file__).parent

rows = []
for stem, qfasta, label, L in PAIRS:
    m8 = ROOT / f"{stem}.m8"
    if not m8.exists():
        continue
    n_q = count_q(ROOT / qfasta)
    best = best_pident_per_query(m8)
    cts = {th: sum(1 for v in best.values() if v >= th) for th in THRESHOLDS}
    rows.append((stem, label, L, n_q, cts))

print(
    f"\n{'Direction':<14} {'L≤':<6} {'n_Q':>8}  "
    + "  ".join(f"≥{int(t):>3}%" for t in THRESHOLDS)
)
print("-" * 80)
for stem, label, L, n_q, cts in rows:
    pcts = "  ".join(f"{cts[t]/n_q*100:5.1f}" for t in THRESHOLDS)
    print(f"{label:<14} {L:<6} {n_q:>8}  {pcts}")

print("\nAbsolute counts:")
print(
    f"{'Direction':<14} {'L≤':<6} {'n_Q':>8}  "
    + "  ".join(f"≥{int(t):>3}%" for t in THRESHOLDS)
)
print("-" * 80)
for stem, label, L, n_q, cts in rows:
    counts = "  ".join(f"{cts[t]:>5d}" for t in THRESHOLDS)
    print(f"{label:<14} {L:<6} {n_q:>8}  {counts}")
