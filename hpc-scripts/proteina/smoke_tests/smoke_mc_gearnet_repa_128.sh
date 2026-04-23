#!/bin/bash
#!
#! Smoke test: MC-GearNet-Edge REPA n=128 (Step 3 of MC-GearNet integration plan).
#!
#! Runs 200 training steps with the n=128 MC-GearNet REPA config to verify:
#!   - Checkpoint loads cleanly on a compute node
#!   - REPA loss is finite and non-zero
#!   - No OOM at bs=80
#!   - Wandb logging works
#!
#! Usage: sbatch hpc-scripts/proteina/smoke_tests/smoke_mc_gearnet_repa_128.sh

#SBATCH -J smoke-mc-gearnet
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:45:00
#SBATCH --qos=gpu1
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-mc-gearnet-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-mc-gearnet-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
PROTEINA_DIR="$REPO_DIR/src/proteina/proteinfoundation"

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_INIT_TIMEOUT=120

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

rm -rf /tmp/torchinductor_${USER} 2>/dev/null
find "$REPO_DIR/src/proteina" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

echo "=== SMOKE: MC-GearNet-Edge REPA n=128 ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== Time: $(date) ==="
echo ""

# Copy LMDB to local NVMe
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
                echo "Done ($(du -h "$LMDB_LOCAL/${split}.lmdb" | cut -f1))"
            fi
        done
        export LMDB_DIR="$LMDB_LOCAL"
    else
        echo "=== LMDB_SOURCE: lustre (not enough /tmp space) ==="
        export LMDB_DIR="$LMDB_SRC"
    fi
else
    echo "WARNING: LMDB not found at $LMDB_SRC"
    export LMDB_DIR="$LMDB_SRC"
fi

echo ""
echo "=== Starting training (200 steps, wandb enabled, no checkpoint) ==="
echo ""

cd "$PROTEINA_DIR"

python -u -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import sys, runpy
sys.argv = [
    'train_repa.py',
    '--config_name', 'training_repa_l0_128_per_residue_mc_edge',
    '--config_subdir', 'training/128/gearnet_rep/per_residue',
    '--max_steps', '200',
    '--show_prog_bar',
    '--no_compile',
    '--batch_size', '40',
]
runpy.run_path('train_repa.py', run_name='__main__')
"
EXIT=$?

echo ""
echo "=== SMOKE COMPLETE (exit code: $EXIT) ==="
echo "=== Time: $(date) ==="
exit $EXIT
