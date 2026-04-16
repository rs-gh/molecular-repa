#!/bin/bash
#!
#! Build LMDB length index on a compute node.
#! Scans all LMDB entries to extract protein lengths, saves as .npy + .pkl
#! files alongside the LMDB. One-time cost (~90 min for 425k entries).
#!
#! Usage: sbatch hpc-scripts/proteina/smoke_tests/build_length_index.sh
#!

#SBATCH -J lmdb-idx
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/lmdb-idx-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/lmdb-idx-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="

LMDB_DIR="/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb"

python -u "$REPO_DIR/hpc-scripts/proteina/build_lmdb_length_index.py" \
    --lmdb_dir "$LMDB_DIR" \
    --splits train val

echo ""
echo "=== COMPLETE (exit code: $?) ==="
echo "=== Time: $(date) ==="
