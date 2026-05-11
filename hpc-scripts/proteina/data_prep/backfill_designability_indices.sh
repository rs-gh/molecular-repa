#!/bin/bash
#! Recompute per-PDB designability indices (designability_index.csv) for all
#! eval_output/ roots that still have surviving tmp_designability/ artefacts.
#! Then re-run backfill_ss.py --force on every results/paper/ sweep so the
#! _designable SS columns get populated everywhere the index now exists.
#!
#! CPU-only (RMSD + TM-score on stored ESMFold refolds; no GPU model load).
#! Estimate: ~0.7s per evaluated PDB. ~11k evaluated PDBs across 68 roots ≈ 2h.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/data_prep/backfill_designability_indices.sh

#SBATCH -A computerlab-sl2-cpu
#SBATCH --job-name=desig_index_backfill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/desig_index_backfill-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/desig_index_backfill-%j.err
#SBATCH -p icelake

set -e

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export PYTHONPATH="$REPO_DIR/src/proteina:$PYTHONPATH"
export OMP_NUM_THREADS=2

cd "$REPO_DIR"

REF_PDB="/rds/user/sr2173/hpc-work/proteina/data/ss_reference_pdb.pt"
REF_AFDB="/rds/user/sr2173/hpc-work/proteina/data/ss_reference_afdb.pt"

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="

# Stage 1: retrofit per-PDB designability_index.csv onto every eligible eval root.
# Two roots searched (mirroring backfill_ss.py's EVAL_OUTPUT_ROOTS list):
#   /rds/...  — jobs that ran with cwd on /rds (most paper runs)
#   /home/... — jobs that ran with cwd on /home (older/n128 paper runs)
# The script itself skips roots without samples_fid/, with empty
# tmp_designability/, or with an existing index.
for base in /rds/user/sr2173/hpc-work/proteina/eval_output /home/sr2173/git/molecular-repa/eval_output; do
    echo ""
    echo "############################################"
    echo "### Stage 1: retrofit indices under $base"
    echo "############################################"
    python -u evaluation/proteina/generation/scripts/backfill_designability_index.py \
        --glob "$base/*"
done

# Stage 2: now every paper sweep_results.jsonl row whose eval root has an
# index can also get the _designable SS columns. --force rewrites the row
# (overall SS values are deterministic, so unaffected; the new columns are
# additive). Backfill_ss.py automatically picks up designability_index.csv
# when present and falls back to skip the _designable columns when not.
echo ""
echo "############################################"
echo "### Stage 2: re-run backfill_ss with --force"
echo "############################################"
for sweep in evaluation/proteina/generation/results/paper/*/; do
    sweep="${sweep%/}"
    [[ -f "$sweep/sweep_results.jsonl" ]] || continue
    echo ""
    echo "=== $sweep ==="
    python -u evaluation/proteina/generation/scripts/backfill_ss.py \
        --sweep_dir "$sweep" \
        --ss_reference_pdb_path "$REF_PDB" \
        --ss_reference_afdb_path "$REF_AFDB" \
        --force
done

echo ""
echo "=== DONE, $(date) ==="
