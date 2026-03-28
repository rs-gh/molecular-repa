#!/bin/bash
#!
#! SLURM job script for Proteina REPA (60M, GearNet alignment, torch.compile)
#! Wilkes3 (AMD EPYC 7763, ConnectX-6, A100 80GB)
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/train_repa.sh          # fresh or auto-resume
#!
#! Checkpoint resume is automatic: if a last.ckpt exists under the run's
#! store directory, train_repa.py picks it up and continues training.
#! WandB likewise resumes the same run (keyed by run_name_ in the config).
#!

#SBATCH -J prot-repa
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=36:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk

#! Output logs on RDS to avoid filling /home (50GB quota):
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/repa-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/repa-%j.err

#! Do not change:
#SBATCH -p ampere

###############################################################
### Environment setup                                       ###
###############################################################

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
PROTEINA_DIR="$REPO_DIR/src/proteina/proteinfoundation"

#! Activate venv (use project venv, not conda)
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

#! Data and environment (DATA_PATH must contain metric_factory/model_weights/gearnet_ca.pth)
export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_INIT_TIMEOUT=120

#! Ensure log directory exists
mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

###############################################################
### Diagnostics                                             ###
###############################################################

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== Config: training_repa ==="
echo "=== Time: $(date) ==="
echo ""

###############################################################
### GPU utilization monitor (background, 10s intervals)     ###
###############################################################

GPU_LOG="/rds/user/sr2173/hpc-work/proteina/logs/gpu-util-repa-${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 10 > "$GPU_LOG" &
GPU_MON_PID=$!

###############################################################
### Training                                                ###
###############################################################

cd "$PROTEINA_DIR"

#! Use spawn start method for num_workers > 0 (avoids fork+CUDA segfaults).
#! train_repa.py handles checkpoint auto-resume and WandB continuation
#! internally via fetch_last_ckpt() and WandbLogger(id=run_name, resume=...).
python -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import sys, runpy
sys.argv = ['train_repa.py', '--config_name', 'training_repa', '--show_prog_bar']
runpy.run_path('train_repa.py', run_name='__main__')
"
TRAIN_EXIT=$?

###############################################################
### Cleanup and summary                                     ###
###############################################################

#! Stop GPU monitor
kill $GPU_MON_PID 2>/dev/null
wait $GPU_MON_PID 2>/dev/null

echo ""
echo "=== GPU utilization summary ==="
python -c "
import csv
with open('$GPU_LOG') as f:
    reader = csv.reader(f)
    header = next(reader)
    utils = []
    for row in reader:
        try:
            val = int(row[1].strip().replace(' %', ''))
            utils.append(val)
        except (ValueError, IndexError):
            pass
if utils:
    print(f'GPU utilization: min={min(utils)}%, max={max(utils)}%, mean={sum(utils)/len(utils):.1f}%, samples={len(utils)}')
    idle = sum(1 for u in utils if u == 0)
    print(f'Time at 0% (idle/loading): {idle}/{len(utils)} samples ({100*idle/len(utils):.1f}%)')
    high = sum(1 for u in utils if u >= 80)
    print(f'Time at 80%+: {high}/{len(utils)} samples ({100*high/len(utils):.1f}%)')
else:
    print('No GPU utilization data collected')
"

echo ""
echo "=== TRAINING COMPLETE (exit code: $TRAIN_EXIT) ==="
echo "=== Time: $(date) ==="

exit $TRAIN_EXIT
