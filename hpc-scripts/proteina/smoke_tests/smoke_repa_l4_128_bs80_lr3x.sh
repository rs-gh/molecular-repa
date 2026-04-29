#!/bin/bash
#!
#! Smoke test: REPA L4 n=128 at bs=80 with linear-scaled LR (3.33e-4).
#!
#! Existing bs=80 REPA config uses LR=1e-4 (the bs=24 LR). This run tests whether
#! linear LR scaling (1e-4 x 80/24 ~= 3.33e-4) is stable WITHOUT warmup - Adam
#! adaptive moments are most fragile in the first few hundred steps, and the base
#! training_ca.yaml has no scheduler. If grads/loss stay finite for 500 steps,
#! the full run is safe to launch.
#!
#! Watch for:
#!   - NaN grads (should be skipped, not crash, due to skip_nan_grad: True)
#!   - trans_loss spiking instead of dropping
#!   - REPA loss diverging
#!
#! Usage: sbatch hpc-scripts/proteina/smoke_tests/smoke_repa_l4_128_bs80_lr3x.sh

#SBATCH -J smoke-repa-lr3x
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:45:00
#SBATCH --qos=intr
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-repa-lr3x-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-repa-lr3x-%j.err
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

#! Wipe torchinductor cache so compile errors (if any) surface, not stale cache hits
rm -rf /tmp/torchinductor_${USER} 2>/dev/null
find "$REPO_DIR/src/proteina" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

echo "=== SMOKE: REPA L4 n=128 bs=80 LR=3.33e-4 (linear-scaled) ==="
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
echo "=== Starting smoke run (500 steps, eager mode for fast startup) ==="
echo ""

cd "$PROTEINA_DIR"

#! 500 steps covers the riskiest phase for Adam at higher LR without warmup.
#! Eager mode (--no_compile) avoids the ~10min compile step so we get to the
#! decision quickly. Production run uses compile=true.
python -u -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import sys, runpy
sys.argv = [
    'train_repa.py',
    '--config_name', 'training_repa_l4_128_per_residue_bs80_lr3x',
    '--config_subdir', 'training/128/gearnet/per_residue',
    '--max_steps', '500',
    '--show_prog_bar',
    '--no_compile',
    '--nolog',  # prevent smoke test from writing checkpoints into the production store dir
]
runpy.run_path('train_repa.py', run_name='__main__')
"
EXIT=$?

echo ""
echo "=== SMOKE COMPLETE (exit code: $EXIT) ==="
echo "=== Time: $(date) ==="
echo ""
echo "Decision criteria:"
echo "  - exit 0 + finite trans_loss + finite repa_loss -> safe to launch full run"
echo "  - NaN spam in logs -> fall back to sqrt-scaled LR (1.83e-4) instead"
echo "  - loss climbing instead of dropping -> LR too high; try sqrt scaling"
echo ""
exit $EXIT
