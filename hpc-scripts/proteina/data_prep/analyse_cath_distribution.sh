#!/bin/bash
#! Run CATH distribution characterisation for PDB and AFDB datasets.
#! Outputs JSON summary + figures to evaluation/proteina/cath_distribution/.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/data_prep/analyse_cath_distribution.sh
#!   sbatch --qos=intr hpc-scripts/proteina/data_prep/analyse_cath_distribution.sh

#SBATCH -J prot-cath-analysis
#SBATCH -A computerlab-sl2-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/cath-analysis-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/cath-analysis-%j.err
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
    --afdb_lmdb_dir "$DATA_PATH/afdb_swissprot/lmdb" \
    --cath_dir      "$DATA_PATH/pdb_train" \
    --interpro_tsv  "$DATA_PATH/afdb_swissprot/interpro_gene3d_swissprot.tsv" \
    --out_dir       "$REPO_DIR/evaluation/proteina/cath_distribution" \
    --splits train val

echo ""
echo "=== CATH ANALYSIS COMPLETE ==="
echo "=== Time: $(date) ==="
