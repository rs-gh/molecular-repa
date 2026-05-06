#!/bin/bash
#!
#! Smoke test: REPA L4 n=128 at bs=80 with weight_decay=1e-2 (AdamW).
#!
#! Verifies:
#!   1. AdamW (not Adam) is selected when opt.weight_decay > 0.
#!   2. Decoupled decay path runs without NaN/Inf.
#!   3. trans_loss + repa_loss stay finite for 500 steps.
#!
#! All other knobs identical to training_repa_l4_128_per_residue_bs80.yaml
#! (lr=1e-4, lambda_repa=0.5, layer 4 alignment, per_residue averaging).
#!
#! Usage: sbatch hpc-scripts/proteina/smoke_tests/smoke_repa_l4_128_bs80_wd1e-2.sh

#SBATCH -J smoke-repa-wd1e-2
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:45:00
#SBATCH --qos=intr
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-repa-wd1e-2-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-repa-wd1e-2-%j.err
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

echo "=== SMOKE: REPA L4 n=128 bs=80 weight_decay=1e-2 (AdamW) ==="
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

#! 500 steps is enough to see optimizer init + a few hundred AdamW steps. Eager
#! (--no_compile) for fast startup. --nolog so smoke output never touches the
#! production checkpoint dir for this run_name.
python -u -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import sys, runpy
sys.argv = [
    'train_repa.py',
    '--config_name', 'training_repa_l4_128_per_residue_bs80_wd1e-2',
    '--config_subdir', 'training/128/gearnet/per_residue',
    '--max_steps', '500',
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
echo ""
echo "Decision criteria:"
echo "  - exit 0 + finite trans_loss + finite repa_loss + AdamW in optimizer log -> safe to launch full run"
echo "  - NaN/Inf in losses or grads -> revert to wd=1e-3 or wd=1e-4"
echo "  - exit !=0 -> read err log; common causes: cfg parse error, OOM"
echo ""
exit $EXIT
