#!/bin/bash
#! Bake CATH labels into PDB and AFDB LMDBs in-place.
#! Run after TED TSV has been downloaded for AFDB.
#! Safe to re-run: entries already enriched are skipped.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/data_prep/bake_cath_labels.sh pdb
#!   sbatch hpc-scripts/proteina/data_prep/bake_cath_labels.sh afdb
#!   sbatch --qos=intr hpc-scripts/proteina/data_prep/bake_cath_labels.sh pdb   # smoke-test priority

#SBATCH -J prot-bake-cath
#SBATCH -A computerlab-sl3-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/bake-cath-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/bake-cath-%j.err
#SBATCH -p icelake

DATASET="${1:-pdb}"   # pdb | afdb

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export PYTHONPATH="$REPO_DIR/src/proteina:$REPO_DIR/src/proteina/proteinfoundation:${PYTHONPATH:-}"

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="
echo "=== Dataset: $DATASET ==="

if [ "$DATASET" = "pdb" ]; then
    python "$REPO_DIR/hpc-scripts/proteina/data_prep/bake_cath_labels.py" \
        --dataset pdb \
        --lmdb_dir  "$DATA_PATH/pdb_train/lmdb" \
        --cath_dir  "$DATA_PATH/pdb_train" \
        --splits train val test

elif [ "$DATASET" = "afdb" ]; then
    INTERPRO_TSV="$DATA_PATH/afdb_swissprot/interpro_gene3d_swissprot.tsv"
    if [ ! -f "$INTERPRO_TSV" ]; then
        echo "ERROR: InterPro Gene3D TSV not found at $INTERPRO_TSV"
        echo "Run: sbatch hpc-scripts/proteina/data_prep/download_interpro_gene3d.sh first."
        exit 1
    fi
    python "$REPO_DIR/hpc-scripts/proteina/data_prep/bake_cath_labels.py" \
        --dataset afdb \
        --lmdb_dir      "$DATA_PATH/afdb_swissprot/lmdb" \
        --interpro_tsv  "$INTERPRO_TSV" \
        --splits train val test
else
    echo "ERROR: unknown dataset '$DATASET'. Use 'pdb' or 'afdb'."
    exit 1
fi

echo ""
echo "=== BAKE CATH COMPLETE: $DATASET ==="
echo "=== Time: $(date) ==="
