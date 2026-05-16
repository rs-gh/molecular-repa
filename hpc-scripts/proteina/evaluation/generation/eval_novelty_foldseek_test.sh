#!/bin/bash
#!
#! One-off timing test for the foldseek novelty path (compute_novelty_foldseek).
#! Targets the existing n256 paper-sweep eval dir which already has 1125 PDBs
#! and 230 designable backbones but no _res_novelty_foldseek_* columns. Merges
#! both pdb and afdb_swissprot columns into the existing results CSV.
#!
#! Why this script: the foldseek path has never been timed end-to-end on this
#! HPC. Smoke test on login (10 queries, 4 threads) measured
#!   pdb:           7.6s  -> ~3 min extrapolated for 230 queries
#!   afdb_swissprot 19.3s -> ~7 min extrapolated for 230 queries
#! With 16 threads on intr/icelake we expect ~1-3 min total.
#!
#! Also patches two latent bugs found during the smoke (now defaulted in
#! _metric_args.py, pinned explicitly here for traceability):
#!   - alignment_type=2 silently returns empty m8 on CA-only PDBs (proteina
#!     samples are CA-only); use alignment_type=1.
#!   - foldseek_threads=0 crashes the binary; use 16.
#!
#! This script does NOT invoke evaluate.py — the hydra config for the target
#! eval dir is no longer in src/proteina/configs/, so evaluate.py would fail
#! at compose. Instead we call compute_novelty_foldseek directly on the
#! designable subset from designability_index.csv and merge the resulting
#! columns into the existing results CSV using the same partial-run merge
#! logic evaluate.py would have applied.

#SBATCH -J nov-foldseek-test
#SBATCH -A computerlab-sl2-cpu
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/nov-foldseek-test-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/nov-foldseek-test-%j.err
#SBATCH -p icelake

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH=/rds/user/sr2173/hpc-work/tools/foldseek/bin:$PATH
mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

export EVAL_DIR="$REPO_DIR/eval_output/inference_paper_inference_fid_60m_paper_sweep_pretrained_dfs_60m_n256_paper_step_1300000"

echo "=== Foldseek novelty timing test ==="
echo "eval dir: $EVAL_DIR"
echo "node:     $(hostname), threads=$SLURM_CPUS_PER_TASK, time=$(date)"
foldseek version
echo ""

cd "$REPO_DIR"

START=$(date +%s)

python -u <<'PYEOF'
import os, sys, time, csv, glob
sys.path.insert(0, "evaluation/proteina/generation/scripts")

import pandas as pd
from utils.novelty_foldseek import compute_novelty_foldseek
from utils._metric_args import METRIC_PREFIX

EVAL_DIR = os.path.expandvars("$EVAL_DIR")
THREADS = 16
DBS = [
    ("pdb",            "/rds/user/sr2173/hpc-work/proteina/foldseek_dbs/pdb/db"),
    ("afdb_swissprot", "/rds/user/sr2173/hpc-work/proteina/foldseek_dbs/afdb_swissprot/db"),
]

# Reproduce evaluate.py's filter_designable=True path: queries = designable subset.
with open(os.path.join(EVAL_DIR, "designability_index.csv")) as f:
    rows = list(csv.DictReader(f))
designable_rel = [r["pdb_path"] for r in rows if r["designable"] == "True"]
queries = [os.path.join(EVAL_DIR, p) for p in designable_rel]
print(f"Designable queries: {len(queries)}", flush=True)
assert queries, "no designable PDBs found"

per_db_results = {}
per_db_wall = {}
for label, path in DBS:
    print(f"\n--- foldseek easy-search vs {label} ---", flush=True)
    t0 = time.time()
    res = compute_novelty_foldseek(
        queries, target_db=path, db_label=label,
        alignment_type=1, max_seqs=1000, sensitivity=9.5, threads=THREADS,
    )
    per_db_wall[label] = time.time() - t0
    print(f"wall: {per_db_wall[label]:.1f}s ({per_db_wall[label]/len(queries):.2f}s/query)", flush=True)
    for k, v in res.items():
        print(f"  {k}: {v}")
        per_db_results[k] = v

# Merge into the existing results CSV (same logic as evaluate.py's partial-run path).
csv_glob = glob.glob(os.path.join(EVAL_DIR, "results_*_fid.csv"))
assert len(csv_glob) == 1, f"expected 1 results CSV, got {csv_glob}"
results_csv = csv_glob[0]
df = pd.read_csv(results_csv)
assert len(df) == 1, f"expected single-row results CSV, got {len(df)} rows"
update_cols = [k for k in per_db_results if k.startswith(METRIC_PREFIX)]
print(f"\nMerging {len(update_cols)} columns into {results_csv}", flush=True)
for k in update_cols:
    df[k] = per_db_results[k]
df.to_csv(results_csv, index=False)

print("\n=== per-DB walls ===")
for label, w in per_db_wall.items():
    print(f"  {label}: {w:.1f}s")
print(f"  total foldseek wall: {sum(per_db_wall.values()):.1f}s")
PYEOF

END=$(date +%s)
echo ""
echo "=== TOTAL job wall: $((END - START))s, $(date) ==="
