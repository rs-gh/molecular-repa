#!/bin/bash
#! Build LMDB from AFDB Swiss-Prot PDB files (max 256 residues).
#! Run AFTER download_afdb_swissprot.sh completes. Supports incremental
#! reruns: new files in raw/ are appended, existing LMDB entries are skipped.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/data_prep/build_afdb_lmdb.sh            # full build
#!   sbatch --qos=intr hpc-scripts/proteina/data_prep/build_afdb_lmdb.sh # intr queue (short)
#!
#! After this completes, build the length index:
#!   sbatch hpc-scripts/proteina/data_prep/build_length_index.sh afdb_swissprot

#SBATCH -J prot-afdb-lmdb
#SBATCH -A computerlab-sl3-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/afdb-lmdb-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/afdb-lmdb-%j.err
#SBATCH -p icelake

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPO_DIR/src/proteina:$REPO_DIR/src/proteina/proteinfoundation:${PYTHONPATH:-}"

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="
echo "=== Tar: $DATA_PATH/afdb_swissprot/swissprot_pdb_v6.tar ==="
echo ""

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

python "$REPO_DIR/hpc-scripts/proteina/data_prep/build_afdb_lmdb.py" \
    --data-dir "$DATA_PATH/afdb_swissprot" \
    --max-residues 256 \
    --num-workers 24 \
    --map-size-gb 50 \
    --batch-size 500

echo ""
echo "=== AFDB LMDB BUILD COMPLETE ==="
echo "=== Time: $(date) ==="
echo ""
echo "Next: build length index:"
echo "  sbatch hpc-scripts/proteina/data_prep/build_length_index.sh afdb_swissprot"
echo "  (passes afdb_swissprot as dataset name -> LMDB_DIR=\$DATA_PATH/afdb_swissprot/lmdb)"
