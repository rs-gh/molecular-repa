#!/bin/bash
#!
#! Smoke test: verify ProteinGenerationMetricsCallback works during training.
#! Runs ~150 steps, triggers callback at step 100, then exits.
#!
#! Usage: sbatch hpc-scripts/proteina/smoke_tests/smoke_genmetrics.sh
#!

#SBATCH -J smoke-gm
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-genmetrics-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-genmetrics-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
PROTEINA_DIR="$REPO_DIR/src/proteina/proteinfoundation"

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export LMDB_DIR="$DATA_PATH/pdb_train/lmdb"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_INIT_TIMEOUT=120

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== Smoke test: GenMetrics callback ==="
echo "=== Time: $(date) ==="
echo ""

# Clean stale caches
rm -rf /tmp/torchinductor_${USER} 2>/dev/null

cd "$PROTEINA_DIR"

# Run training for ~150 steps - callback should fire at step 100
# Then kill after the callback completes (or after 200 steps max)
python -u -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import sys, runpy
sys.argv = ['train_repa.py', '--config_name', 'training_baseline_smoke_genmetrics', '--show_prog_bar']
runpy.run_path('train_repa.py', run_name='__main__')
"
EXIT_CODE=$?

echo ""
echo "=== SMOKE TEST COMPLETE (exit code: $EXIT_CODE) ==="
echo "=== Time: $(date) ==="
echo "Check WandB for val/gen_* metrics in project proteina-repa, run smoke_genmetrics_test"

exit $EXIT_CODE
