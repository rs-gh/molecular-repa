#!/bin/bash
#!
#! Parameterised smoke test for n=256 AFDB REPA configs.
#! Verifies: config loads, encoder weights load, REPA forward+backward,
#! finite trans/repa loss for 200 steps. Runs in eager mode (--no_compile)
#! for fast startup; production launch uses compile=true from the config.
#!
#! Mirrors smoke_repa_128_param.sh but stages the AFDB LMDB instead of PDB.
#!
#! Usage:
#!   sbatch smoke_repa_256_afdb_param.sh <config_name> <config_subdir> [eager|compile]
#!
#! Third arg defaults to `eager` (fast startup; verifies code path only).
#! Use `compile` to replicate prod memory profile (slower startup; the only
#! way to validate fit at bs=24).
#!
#! Example:
#!   sbatch smoke_repa_256_afdb_param.sh \
#!     training_repa_l4_256_afdb_per_residue \
#!     training/256/afdb/mpnn compile

#SBATCH -J smoke-repa-256-afdb
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --qos=intr
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-repa-256-afdb-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-repa-256-afdb-%j.err
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
COMPILE_MODE="${3:-eager}"   # eager | compile

JOB_SHORT=$(echo "$CONFIG_NAME" | sed 's/^training_//' | sed 's/_/-/g')
scontrol update JobId="$SLURM_JOB_ID" JobName="smoke-${JOB_SHORT}" 2>/dev/null

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

rm -rf /tmp/torchinductor_${USER} 2>/dev/null
find "$REPO_DIR/src/proteina" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

echo "=== SMOKE: $CONFIG_SUBDIR/$CONFIG_NAME ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== Time: $(date) ==="
echo ""

# Copy AFDB LMDB to local NVMe (avoid Lustre mmap thrashing)
LMDB_SRC="$DATA_PATH/afdb_swissprot/lmdb"
LMDB_LOCAL="/tmp/proteina_afdb_lmdb"

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
        for f in "$LMDB_SRC"/*.npy "$LMDB_SRC"/*.pkl; do
            [ -f "$f" ] && cp "$f" "$LMDB_LOCAL/"
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
echo "=== Starting smoke (200 steps, mode=${COMPILE_MODE}, --nolog so prod ckpt dir is untouched) ==="
echo ""

cd "$PROTEINA_DIR"

python -u -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import sys, runpy
argv = [
    'train_repa.py',
    '--config_name', '${CONFIG_NAME}',
    '--config_subdir', '${CONFIG_SUBDIR}',
    '--max_steps', '200',
    '--show_prog_bar',
    '--nolog',
    '--single',
]
if '${COMPILE_MODE}' == 'eager':
    argv.append('--no_compile')
sys.argv = argv
runpy.run_path('train_repa.py', run_name='__main__')
"
EXIT=$?

echo ""
echo "=== SMOKE COMPLETE (exit code: $EXIT) ==="
echo "=== Time: $(date) ==="
exit $EXIT
