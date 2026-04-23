#!/bin/bash
#!
#! Smoke test: MC-GearNet-Edge REPA n=128, torch.compile ON.
#! Companion to smoke_mc_gearnet_repa_128.sh (compile=off).
#! Purpose: verify compile works and measure steady-state step time vs no-compile.
#! Needs 2h wall — GearNet graph construction takes ~30 min to trace through compile.
#!
#! Usage: sbatch hpc-scripts/proteina/smoke_tests/smoke_mc_gearnet_repa_128_compile.sh

#SBATCH -J smoke-mc-compile
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --qos=gpu1
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-mc-gearnet-compile-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-mc-gearnet-compile-%j.err
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

echo "=== SMOKE (compile=on): MC-GearNet-Edge REPA n=128 ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== Time: $(date) ==="
echo ""

LMDB_SRC="$DATA_PATH/pdb_train/lmdb"
LMDB_LOCAL="/tmp/proteina_pdb_lmdb"

if [ -d "$LMDB_SRC" ]; then
    LMDB_SIZE_KB=$(du -sk "$LMDB_SRC" 2>/dev/null | cut -f1)
    TMP_FREE_KB=$(df /tmp --output=avail 2>/dev/null | tail -1 | tr -d ' ')
    echo "LMDB total: $((LMDB_SIZE_KB / 1024))MB, /tmp free: $((TMP_FREE_KB / 1024))MB"
    if [ "${TMP_FREE_KB:-0}" -gt "$((LMDB_SIZE_KB + 1048576))" ]; then
        echo "=== LMDB_SOURCE: nvme ==="
        rm -rf "$LMDB_LOCAL" && mkdir -p "$LMDB_LOCAL"
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
        echo "=== LMDB_SOURCE: lustre ==="
        export LMDB_DIR="$LMDB_SRC"
    fi
else
    export LMDB_DIR="$LMDB_SRC"
fi

echo ""
echo "=== Starting training (300 steps, compile=on, nolog) ==="
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
    '--max_steps', '300',
    '--show_prog_bar',
    '--nolog',
]
runpy.run_path('train_repa.py', run_name='__main__')
"
EXIT=$?

echo ""
echo "=== SMOKE COMPLETE (exit code: \$EXIT) ==="
echo "=== Time: \$(date) ==="
exit $EXIT
