#!/bin/bash
#!
#! Parameterised 2-GPU DDP smoke for proteina training configs.
#! Verifies: srun-launched DDP, rank-per-GPU binding, config loads on both
#! ranks, gradient all-reduce works for ~200 steps. Runs in eager mode
#! (--no_compile) for fast startup; prod uses compile=true from the config.
#!
#! Usage:
#!   sbatch smoke_2gpu_param.sh <config_name> <config_subdir>
#!
#! Example:
#!   sbatch smoke_2gpu_param.sh training_baseline_256_bs24_2gpu training/256

#SBATCH -J smoke-2gpu
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=0:45:00
#SBATCH --qos=intr
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-2gpu-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-2gpu-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
PROTEINA_DIR="$REPO_DIR/src/proteina/proteinfoundation"

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export ESM_MODEL_PATH="/rds/user/sr2173/hpc-work/proteina/hf_cache/esm2_t33_650M_UR50D"
export PROTEINMPNN_WEIGHTS_DIR="/rds/user/sr2173/hpc-work/proteina/ProteinMPNN"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_INIT_TIMEOUT=120

CONFIG_NAME="${1:?config_name required}"
CONFIG_SUBDIR="${2:?config_subdir required}"

JOB_SHORT=$(echo "$CONFIG_NAME" | sed 's/^training_//' | sed 's/_/-/g')
scontrol update JobId="$SLURM_JOB_ID" JobName="smoke-2gpu-${JOB_SHORT}" 2>/dev/null

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs
rm -rf /tmp/torchinductor_${USER} 2>/dev/null
find "$REPO_DIR/src/proteina" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

echo "=== SMOKE 2-GPU: $CONFIG_SUBDIR/$CONFIG_NAME ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID, NTASKS: $SLURM_NTASKS ==="
echo "=== Time: $(date) ==="
echo ""

# Copy PDB LMDB to local NVMe
LMDB_SRC="$DATA_PATH/pdb_train/lmdb"
LMDB_LOCAL="/tmp/proteina_pdb_lmdb"

if [ -d "$LMDB_SRC" ]; then
    LMDB_SIZE_KB=$(du -sk "$LMDB_SRC" 2>/dev/null | cut -f1)
    TMP_FREE_KB=$(df /tmp --output=avail 2>/dev/null | tail -1 | tr -d ' ')
    echo "LMDB total: $((LMDB_SIZE_KB / 1024))MB, /tmp free: $((TMP_FREE_KB / 1024))MB"
    if [ "${TMP_FREE_KB:-0}" -gt "$((LMDB_SIZE_KB + 1048576))" ]; then
        echo "=== LMDB_SOURCE: nvme ==="
        rm -rf "$LMDB_LOCAL"
        mkdir -p "$LMDB_LOCAL"
        for split in val test train; do
            src="$LMDB_SRC/${split}.lmdb"
            if [ -f "$src" ]; then
                echo "Copying ${split}.lmdb to local NVMe..."
                cp "$src" "$LMDB_LOCAL/${split}.lmdb"
            fi
        done
        export LMDB_DIR="$LMDB_LOCAL"
    else
        echo "=== LMDB_SOURCE: lustre ==="
        export LMDB_DIR="$LMDB_SRC"
    fi
else
    echo "WARNING: LMDB not found at $LMDB_SRC"
    export LMDB_DIR="$LMDB_SRC"
fi

echo ""
echo "=== Starting 2-GPU DDP smoke (200 steps, eager, --nolog) ==="
echo ""

cd "$PROTEINA_DIR"

unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK 2>/dev/null
srun --cpus-per-task=8 --kill-on-bad-exit=1 python -u -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import sys, runpy
sys.argv = [
    'train_repa.py',
    '--config_name', '${CONFIG_NAME}',
    '--config_subdir', '${CONFIG_SUBDIR}',
    '--max_steps', '200',
    '--show_prog_bar',
    '--no_compile',
    '--nolog',
]
runpy.run_path('train_repa.py', run_name='__main__')
"
EXIT=$?

echo ""
echo "=== SMOKE COMPLETE (exit code: $EXIT) ==="
echo "=== Time: $(date) ==="
exit $EXIT
