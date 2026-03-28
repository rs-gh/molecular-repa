#!/bin/bash
#! Run SDPA attention tests on GPU compute node
#SBATCH -J prot-sdpa-test
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/sdpa-test-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/sdpa-test-%j.err
#SBATCH -p ampere

module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

cd "$REPO_DIR"

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo ""

python -m pytest tests/proteina/test_sdpa_attention.py -v -s 2>&1

echo ""
echo "=== SDPA TESTS COMPLETE ==="
echo "=== Time: $(date) ==="
