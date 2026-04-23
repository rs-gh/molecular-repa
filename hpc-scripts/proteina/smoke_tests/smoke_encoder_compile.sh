#!/bin/bash
#!
#! Smoke test: verify torch.compile on REPA encoder for both GearNet-CA and
#! MC-GearNet-Edge. Runs 50 steps each (enough to get past Dynamo tracing and
#! measure steady-state step time).
#!
#! Needs 2h wall — encoder compile warmup is ~20-40 min per run.
#!
#! Usage: sbatch hpc-scripts/proteina/smoke_tests/smoke_encoder_compile.sh

#SBATCH -J smoke-enc-compile
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --qos=gpu1
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-enc-compile-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-enc-compile-%j.err
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

# Stage LMDB to local NVMe
LMDB_SRC="$DATA_PATH/pdb_train/lmdb"
LMDB_LOCAL="/tmp/proteina_pdb_lmdb_$$"
mkdir -p "$LMDB_LOCAL"
trap 'rm -rf "$LMDB_LOCAL"' EXIT

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo ""

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
    echo "=== Using Lustre LMDB (insufficient /tmp space) ==="
    export LMDB_DIR="$LMDB_SRC"
fi

rm -rf /tmp/torchinductor_${USER}_enc_compile 2>/dev/null

run_smoke() {
    local label="$1"
    local config_name="$2"
    local config_subdir="$3"
    echo ""
    echo "======================================================"
    echo "=== SMOKE: $label ==="
    echo "=== Start: $(date) ==="
    echo "======================================================"

    cd "$PROTEINA_DIR"
    python -u -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import sys, runpy
sys.argv = [
    'train_repa.py',
    '--config_name', '$config_name',
    '--config_subdir', '$config_subdir',
    '--max_steps', '50',
    '--show_prog_bar',
    '--batch_size', '80',
]
runpy.run_path('train_repa.py', run_name='__main__')
"
    local exit_code=$?
    echo "=== $label exit code: $exit_code ==="
    echo "=== End: $(date) ==="
    return $exit_code
}

# Run GearNet-CA first (faster warmup, gives confidence before the longer MC run)
run_smoke \
    "GearNet-CA encoder compile (bs=80)" \
    "training_repa_l0_128_per_residue_bs80" \
    "training/128/gearnet/per_residue"
CA_EXIT=$?

# Run MC-GearNet-Edge
run_smoke \
    "MC-GearNet-Edge encoder compile (bs=80)" \
    "training_repa_l0_128_per_residue_mc_edge" \
    "training/128/gearnet_rep/per_residue"
MC_EXIT=$?

echo ""
echo "======================================================"
echo "=== SUMMARY ==="
echo "  GearNet-CA:       exit $CA_EXIT"
echo "  MC-GearNet-Edge:  exit $MC_EXIT"
echo "=== Done: $(date) ==="
echo "======================================================"

[ $CA_EXIT -eq 0 ] && [ $MC_EXIT -eq 0 ] && exit 0 || exit 1
