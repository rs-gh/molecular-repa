#!/bin/bash
#!
#! Smoke test: torch.compile on GearNet-CA REPA encoder (n=128, bs=80).
#! Verifies the encoder compile path in train_repa.py works cleanly and
#! measures steady-state step time vs the previous eager-encoder baseline.
#!
#! Usage: sbatch hpc-scripts/proteina/smoke_tests/smoke_gearnet_ca_encoder_compile.sh

#SBATCH -J smoke-ca-enc-compile
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --qos=gpu1
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-ca-enc-compile-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-ca-enc-compile-%j.err
#SBATCH -p ampere
#SBATCH --exclude=gpu-q-43

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

REPO_DIR="/home/sr2173/git/molecular-repa"
PROTEINA_DIR="$REPO_DIR/src/proteina/proteinfoundation"

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="${DATA_PATH:-/rds/user/sr2173/hpc-work/proteina/data}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WANDB_MODE=disabled
export WANDB_INIT_TIMEOUT=120

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== Time: $(date) ==="
echo ""

# Stage LMDB to local NVMe
LMDB_SRC="$DATA_PATH/pdb_train/lmdb"
LMDB_LOCAL="/tmp/proteina_pdb_lmdb_$$"
mkdir -p "$LMDB_LOCAL"
trap 'rm -rf "$LMDB_LOCAL"' EXIT

LMDB_SIZE_KB=$(du -sk "$LMDB_SRC" 2>/dev/null | cut -f1)
TMP_FREE_KB=$(df /tmp --output=avail 2>/dev/null | tail -1 | tr -d ' ')
if [ "${TMP_FREE_KB:-0}" -gt "$((LMDB_SIZE_KB + 1048576))" ]; then
    echo "=== Staging LMDB to local NVMe ==="
    for split in val train; do
        src="$LMDB_SRC/${split}.lmdb"
        [ -f "$src" ] && cp "$src" "$LMDB_LOCAL/${split}.lmdb" && echo "  Staged ${split}.lmdb"
    done
    export LMDB_DIR="$LMDB_LOCAL"
else
    echo "=== Using Lustre LMDB ==="
    export LMDB_DIR="$LMDB_SRC"
fi

rm -rf /tmp/torchinductor_${USER} 2>/dev/null

echo ""
echo "=== Starting GearNet-CA encoder compile smoke (50 steps, bs=80) ==="
echo ""

cd "$PROTEINA_DIR"
# Monkeypatch hydra compose to override run_name_ so the smoke does not resume
# from the production run's checkpoint (and does not pollute its wandb run id).
python -u -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import sys, runpy, hydra
_orig_compose = hydra.compose
def _patched_compose(*a, **kw):
    cfg = _orig_compose(*a, **kw)
    try:
        cfg.run_name_ = 'smoke_enc_compile_ca_' + str(${SLURM_JOB_ID:-0})
    except Exception:
        pass
    return cfg
hydra.compose = _patched_compose
sys.argv = [
    'train_repa.py',
    '--config_name', 'training_repa_l0_128_per_residue_bs80',
    '--config_subdir', 'training/128/gearnet/per_residue',
    '--max_steps', '50',
    '--show_prog_bar',
    '--batch_size', '80',
    '--nolog',  # belt-and-braces: also disable wandb + ckpt writing
]
runpy.run_path('train_repa.py', run_name='__main__')
"
EXIT=$?

echo ""
echo "=== SMOKE COMPLETE (exit $EXIT) — $(date) ==="
exit $EXIT
