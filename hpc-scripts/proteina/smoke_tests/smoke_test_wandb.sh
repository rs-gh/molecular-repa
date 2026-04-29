#!/bin/bash
#! Proteina smoke test with WandB logging - verify logging pipeline
#SBATCH -J prot-wandb
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-wandb-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-wandb-%j.err
#SBATCH -p ampere

module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

cd "$REPO_DIR/src/proteina/proteinfoundation"

export WANDB_INIT_TIMEOUT=120

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== Time: $(date) ==="

# Run WITHOUT --nolog so WandB is active
python train_repa.py --config_name smoke_test --single --show_prog_bar

echo "=== SMOKE TEST WANDB COMPLETE ==="
echo "=== Time: $(date) ==="
