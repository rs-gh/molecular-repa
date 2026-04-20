#!/bin/bash
#!
#! Smoke: length-bucketed training end-to-end.
#!
#! Verifies on real data + real GPU:
#!   - bucketed sampler yields 4 distinct (B, N) shapes
#!   - collator pads to bucket upper edge (bucket_length attached)
#!   - PerBucketGradAccumCallback mutates trainer.accumulate_grad_batches per
#!     batch according to which bucket the batch came from
#!   - torch.compile hits cached graph per bucket after first visit
#!   - loss is finite, backward flows, optimizer steps happen
#!
#! 80 steps total = enough to visit every bucket multiple times post-compile.
#! --nolog skips wandb + checkpoint writes. --qos=intr for fast queue.
#!
#! Usage: sbatch hpc-scripts/proteina/smoke_tests/smoke_bucketed_training.sh

#SBATCH -J smoke-bucket
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:45:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-bucket-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-bucket-%j.err
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

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

rm -rf /tmp/torchinductor_${USER} 2>/dev/null
find "$REPO_DIR/src/proteina" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

echo "=== SMOKE: length-bucketed training ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== Time: $(date) ==="
echo ""

# Copy LMDB to local NVMe (same pattern as production scripts)
LMDB_SRC="$DATA_PATH/pdb_train/lmdb"
LMDB_LOCAL="/tmp/proteina_pdb_lmdb"

if [ -d "$LMDB_SRC" ]; then
    LMDB_SIZE_KB=$(du -sk "$LMDB_SRC" 2>/dev/null | cut -f1)
    TMP_FREE_KB=$(df /tmp --output=avail 2>/dev/null | tail -1 | tr -d ' ')
    if [ "${TMP_FREE_KB:-0}" -gt "$((LMDB_SIZE_KB + 1048576))" ]; then
        echo "=== LMDB_SOURCE: nvme ==="
        rm -rf "$LMDB_LOCAL"
        mkdir -p "$LMDB_LOCAL"
        for split in val test train; do
            src="$LMDB_SRC/${split}.lmdb"
            if [ -f "$src" ]; then
                echo "Copying ${split}.lmdb..."
                cp "$src" "$LMDB_LOCAL/${split}.lmdb"
            fi
        done
        export LMDB_DIR="$LMDB_LOCAL"
    else
        echo "=== LMDB_SOURCE: lustre (not enough /tmp) ==="
        export LMDB_DIR="$LMDB_SRC"
    fi
else
    echo "WARNING: LMDB not found at $LMDB_SRC"
    export LMDB_DIR="$LMDB_SRC"
fi

echo ""

cd "$PROTEINA_DIR"

python -u -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import sys, runpy
sys.argv = [
    'train_repa.py',
    '--config_name', 'training_baseline_bucketed',
    '--config_subdir', 'training/512',
    '--nolog',
    '--max_steps', '80',
    '--show_prog_bar',
]
runpy.run_path('train_repa.py', run_name='__main__')
"
TRAIN_EXIT=$?

echo ""
echo "=== SMOKE COMPLETE (exit code: $TRAIN_EXIT) ==="
echo "=== Time: $(date) ==="
exit $TRAIN_EXIT
