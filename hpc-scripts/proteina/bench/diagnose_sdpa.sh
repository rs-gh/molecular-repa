#!/bin/bash
#!
#! Quick SDPA backend diagnostic — prints raw errors per backend for
#! proteina's attention pattern. ~5 min GPU time.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/bench/diagnose_sdpa.sh
#!

#SBATCH -J prot-sdpa-diag
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:04:47
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/sdpa-diag-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/sdpa-diag-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export PYTHONUNBUFFERED=1

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo ""

python -u "$REPO_DIR/hpc-scripts/proteina/bench/diagnose_sdpa.py"

echo ""
echo "=== DONE: $(date) ==="
