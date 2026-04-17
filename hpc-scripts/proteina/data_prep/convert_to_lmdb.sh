#!/bin/bash
#! Convert raw CIF/PDB files directly to LMDB format.
#! Run AFTER download_raw_pdb.sh or download_raw_pdb_fast.sh completes.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/data_prep/convert_to_lmdb.sh          # default: PDB dataset
#!   sbatch hpc-scripts/proteina/data_prep/convert_to_lmdb.sh d_FS     # D_FS dataset

#SBATCH -J prot-lmdb-conv
# Using GPU account/partition for a CPU-only job because the CPU account
# (COMPUTERLAB-SL2-CPU) has hit AssocGrpCPUMinutesLimit on all partitions.
#SBATCH -A lio-charm-sl2-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/lmdb-conv-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/lmdb-conv-%j.err
#SBATCH -p ampere

module load rhel8/default

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"

DATASET="${1:-pdb}"

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="
echo "=== Dataset: $DATASET ==="
echo ""

if [ "$DATASET" = "pdb" ]; then
    python "$REPO_DIR/hpc-scripts/proteina/data_prep/convert_to_lmdb.py" \
        --config_name pdb_train_compile --config_subdir pdb/original
elif [ "$DATASET" = "d_FS" ]; then
    python "$REPO_DIR/hpc-scripts/proteina/data_prep/convert_to_lmdb.py" \
        --config_name d_FS --config_subdir afdb
else
    echo "Unknown dataset: $DATASET (use 'pdb' or 'd_FS')"
    exit 1
fi

echo ""
echo "=== LMDB CONVERSION COMPLETE ==="
echo "=== Time: $(date) ==="
