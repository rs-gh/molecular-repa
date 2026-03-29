#!/bin/bash
#! Convert processed .pt protein files to LMDB format.
#! Run AFTER prepare_data_pdb.sh or prepare_data_dfs.sh completes.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/convert_to_lmdb.sh              # default: PDB dataset
#!   sbatch hpc-scripts/proteina/convert_to_lmdb.sh d_FS         # D_FS dataset
#!   sbatch hpc-scripts/proteina/convert_to_lmdb.sh pdb --clean  # PDB + delete raw files after

#SBATCH -J prot-lmdb-conv
#SBATCH -A COMPUTERLAB-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/lmdb-conv-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/lmdb-conv-%j.err
#SBATCH -p icelake

module load rhel8/default

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"

DATASET="${1:-pdb}"
CLEAN_RAW=false
if [ "$2" = "--clean" ]; then
    CLEAN_RAW=true
fi

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="
echo "=== Dataset: $DATASET ==="
echo "=== Clean raw files: $CLEAN_RAW ==="
echo ""

if [ "$DATASET" = "pdb" ]; then
    RAW_DIR="$DATA_PATH/pdb_train/raw"
    PROCESSED_DIR="$DATA_PATH/pdb_train/processed"
    python "$REPO_DIR/hpc-scripts/proteina/convert_to_lmdb.py" \
        --config_name pdb_train_compile --config_subdir pdb
elif [ "$DATASET" = "d_FS" ]; then
    RAW_DIR="$DATA_PATH/d_FS/raw"
    PROCESSED_DIR="$DATA_PATH/d_FS/processed"
    python "$REPO_DIR/hpc-scripts/proteina/convert_to_lmdb.py" \
        --config_name d_FS --config_subdir afdb
else
    echo "Unknown dataset: $DATASET (use 'pdb' or 'd_FS')"
    exit 1
fi

CONVERT_EXIT=$?

if [ $CONVERT_EXIT -ne 0 ]; then
    echo "=== LMDB conversion failed (exit $CONVERT_EXIT), skipping cleanup ==="
    exit $CONVERT_EXIT
fi

if [ "$CLEAN_RAW" = true ]; then
    echo ""
    echo "=== Cleaning up raw files ==="
    N_RAW=$(ls "$RAW_DIR" 2>/dev/null | wc -l)
    echo "Deleting $N_RAW files in $RAW_DIR..."
    rm -rf "$RAW_DIR"
    mkdir -p "$RAW_DIR"
    echo "Raw files deleted."

    if [ -d "$PROCESSED_DIR" ]; then
        N_PROC=$(ls "$PROCESSED_DIR" 2>/dev/null | wc -l)
        if [ "$N_PROC" -gt 0 ]; then
            echo "Deleting $N_PROC legacy .pt files in $PROCESSED_DIR..."
            rm -rf "$PROCESSED_DIR"
            mkdir -p "$PROCESSED_DIR"
            echo "Processed files deleted."
        fi
    fi
fi

echo ""
echo "=== LMDB CONVERSION COMPLETE ==="
echo "=== Time: $(date) ==="
