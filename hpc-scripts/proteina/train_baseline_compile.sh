#!/bin/bash
#!
#! SLURM job script for Proteina baseline (60M, no REPA, torch.compile)
#! Wilkes3 (AMD EPYC 7763, ConnectX-6, A100 80GB)
#!

#SBATCH -J prot-base-compile
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=36:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk

#! Output logs on RDS to avoid filling /home:
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/baseline-compile-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/baseline-compile-%j.err

#! Do not change:
#SBATCH -p ampere

#! Environment setup
. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
PROTEINA_DIR="$REPO_DIR/src/proteina/proteinfoundation"

#! Activate venv
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

#! Set data path
export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"

#! Suppress nested threading (good practice for num_workers > 0 later)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_INIT_TIMEOUT=120

#! Ensure log directory exists
mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

#! Run training
cd "$PROTEINA_DIR"

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Config: training_ca_baseline_compile ==="
echo "=== Time: $(date) ==="

CMD="python train_repa.py --config_name training_ca_baseline_compile --show_prog_bar"

echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD

echo "=== TRAINING COMPLETE ==="
echo "=== Time: $(date) ==="
