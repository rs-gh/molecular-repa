#!/bin/bash
#! Backfill SS metric columns into all paper sweep_results.jsonl files.
#! CPU-only, ~5-10 min total (~5s per checkpoint × ~50 checkpoints across 16 sweeps).
#!
#! Usage:
#!   sbatch --dependency=afterok:<PDB_JOB>:<AFDB_JOB> \
#!       hpc-scripts/proteina/data_prep/backfill_ss_metrics.sh

#SBATCH -A computerlab-sl2-cpu
#SBATCH --job-name=ss_backfill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/ss_backfill-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/ss_backfill-%j.err
#SBATCH -p icelake

set -e

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export PYTHONPATH="$REPO_DIR/src/proteina:$PYTHONPATH"
export OMP_NUM_THREADS=1

cd "$REPO_DIR"

REF_PDB="/rds/user/sr2173/hpc-work/proteina/data/ss_reference_pdb.pt"
REF_AFDB="/rds/user/sr2173/hpc-work/proteina/data/ss_reference_afdb.pt"

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="
echo "=== Refs: PDB=$REF_PDB, AFDB=$REF_AFDB ==="

for sweep in evaluation/proteina/generation/results/paper/*/; do
    sweep="${sweep%/}"
    if [[ ! -f "$sweep/sweep_results.jsonl" ]]; then
        echo ">>> SKIP $sweep (no sweep_results.jsonl)"
        continue
    fi
    echo ""
    echo "=== Backfilling $sweep ==="
    python -u evaluation/proteina/generation/scripts/backfill_ss.py \
        --sweep_dir "$sweep" \
        --ss_reference_pdb_path "$REF_PDB" \
        --ss_reference_afdb_path "$REF_AFDB"
done

echo ""
echo "=== DONE, $(date) ==="
