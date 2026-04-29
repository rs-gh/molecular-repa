#!/bin/bash
#! CATH distribution characterisation - PDB only.
#! Usage: sbatch hpc-scripts/proteina/data_prep/analyse_cath_pdb.sh

#SBATCH -J prot-cath-pdb
#SBATCH -A computerlab-sl2-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/cath-pdb-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/cath-pdb-%j.err
#SBATCH -p icelake

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

python "$REPO_DIR/hpc-scripts/proteina/data_prep/analyse_cath_distribution.py" \
    --pdb_lmdb_dir  "$DATA_PATH/pdb_train/lmdb" \
    --cath_dir      "$DATA_PATH/pdb_train" \
    --out_dir       "$REPO_DIR/evaluation/proteina/cath_distribution" \
    --splits train val

echo "=== CATH PDB COMPLETE === $(date) ==="
