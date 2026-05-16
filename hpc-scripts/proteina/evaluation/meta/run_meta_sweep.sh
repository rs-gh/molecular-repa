#!/bin/bash
#!
#! SLURM array job: meta-evaluation sweep.
#!
#! Drives evaluation/proteina/meta/run_meta_sweep.py, which fans out
#! (target_run, seed) pairs as independent tasks. Used for:
#!   1. Sanity-checking that the seed actually changes the generated PDBs
#!      (sanity_seed_n128, sanity_seed_n256).
#!   2. Timing the FID-only path at the full paper-protocol N=5625
#!      (fid_scaling_n256, fid_scaling_n128).
#!   3. Estimating the between-rep variance of FID / designability / diversity
#!      (variance_n128_layer, variance_n256_layer).
#!
#! Each array task runs ONE (target, seed) pair. Use --dry_run on the python
#! driver first to print the (target, seed) table and confirm the --array range.
#!
#! Usage:
#!   # 1. Sanity-check a single fresh seed=43 on the n=128 baseline
#!   python evaluation/proteina/meta/run_meta_sweep.py --config sanity_seed_n128 --dry_run
#!   sbatch --array=0-0 hpc-scripts/proteina/evaluation/meta/run_meta_sweep.sh --config sanity_seed_n128
#!
#!   # 1b. Same for n=256 baseline
#!   sbatch --array=0-0 hpc-scripts/proteina/evaluation/meta/run_meta_sweep.sh --config sanity_seed_n256
#!
#!   # 2. Time N=5625 FID-only at n=256 (expect ~2h25 wall-clock)
#!   sbatch --array=0-0 hpc-scripts/proteina/evaluation/meta/run_meta_sweep.sh --config fid_scaling_n256
#!
#!   # 3. Variance: 5 seeds × 2 ckpts at n=128 (10 array tasks)
#!   sbatch --array=0-9 hpc-scripts/proteina/evaluation/meta/run_meta_sweep.sh --config variance_n128_layer
#!
#!   # 3b. Variance: 5 seeds × 1 ckpt at n=256 (5 array tasks)
#!   sbatch --array=0-4 hpc-scripts/proteina/evaluation/meta/run_meta_sweep.sh --config variance_n256_layer
#!
#! TIME BUDGET:
#!   - n=128 paper protocol with des: ~25 min/task (~10 gen + ~15 des). Fits
#!     in --qos=intr (1h cap) easily.
#!   - n=256 paper protocol with des: ~1h45/task (gen=27 min, des dominates).
#!     Use --time=03:00:00 to leave margin.
#!   - fid_scaling_n256 (N=5625, FID-only): ~2h30/task. Bumped --time to 4h.
#!
#! Time is set to 04:00 for safety across all profiles; reduce manually for
#! sanity / variance_n128 if the queue is busy.

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --job-name=meta-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/meta-sweep-%A_%a.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/meta-sweep-%A_%a.err
#SBATCH -p ampere

set -e

PY_ARGS=("$@")

###############################################################
### Environment setup (mirrors run_sweep.sh)                ###
###############################################################

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

REPO_DIR="/home/sr2173/git/molecular-repa"

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH="/rds/user/sr2173/hpc-work/tools/foldseek/bin:$PATH"

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

###############################################################
### ProteinMPNN CA weights (designability)                  ###
###############################################################
CA_WEIGHTS_ROOT="/rds/user/sr2173/hpc-work/proteina/ProteinMPNN"
mkdir -p "$CA_WEIGHTS_ROOT/ca_model_weights"
for v in v_48_002 v_48_010 v_48_020; do
    f="$CA_WEIGHTS_ROOT/ca_model_weights/${v}.pt"
    if [ ! -f "$f" ]; then
        echo "Downloading ProteinMPNN CA weights: ${v}.pt"
        wget -q -O "$f" "https://github.com/dauparas/ProteinMPNN/raw/main/ca_model_weights/${v}.pt"
    fi
done
export PROTEINMPNN_DIR="$REPO_DIR/src/proteina/ProteinMPNN"
export PROTEINMPNN_WEIGHTS_DIR="$CA_WEIGHTS_ROOT"

###############################################################
### HF cache + TMPDIR on RDS (ESMFold ~3GB)                 ###
###############################################################
export CACHE_DIR="/rds/user/sr2173/hpc-work/proteina/hf_cache"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_HUB_DISABLE_XET=1
export TMPDIR="/rds/user/sr2173/hpc-work/proteina/tmp"
mkdir -p "$CACHE_DIR/hub" "$TMPDIR"

###############################################################
### Per-task torchinductor cache                            ###
###############################################################
TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${SLURM_JOB_ID:-nojob}_${SLURM_ARRAY_TASK_ID:-0}"
export TORCHINDUCTOR_CACHE_DIR
rm -rf "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
find "$REPO_DIR/src/proteina" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

###############################################################
### Diagnostics                                             ###
###############################################################
echo "=== Meta sweep ==="
echo "=== Args: ${PY_ARGS[*]} ==="
echo "=== SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-none} ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== Time: $(date) ==="
echo ""

###############################################################
### GPU utilization monitor (background, 30s intervals)     ###
###############################################################
GPU_LOG="/rds/user/sr2173/hpc-work/proteina/logs/gpu-util-meta-sweep-${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 30 > "$GPU_LOG" &
GPU_MON_PID=$!

###############################################################
### Run meta sweep                                          ###
###############################################################
cd "$REPO_DIR"

python -u -c "
import os, sys
sys.path.insert(0, 'src/proteina/proteinfoundation')
sys.path.insert(0, 'src/proteina')
sys.path.insert(0, 'evaluation/proteina')

import proteinfoundation.repa.pyg_compat  # patches sys.modules
from lib.torch_load_patch import apply as _apply_torch_load_patch
_apply_torch_load_patch(strip_repa=True)

args = list('${PY_ARGS[*]}'.split()) if '${PY_ARGS[*]}' else []
sys.argv = ['run_meta_sweep.py'] + args

import runpy
runpy.run_path('evaluation/proteina/meta/run_meta_sweep.py', run_name='__main__')
"
SWEEP_EXIT=$?

###############################################################
### Cleanup                                                 ###
###############################################################
kill $GPU_MON_PID 2>/dev/null
wait $GPU_MON_PID 2>/dev/null
rm -rf "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null

echo ""
echo "=== META SWEEP TASK COMPLETE (exit code: $SWEEP_EXIT) ==="
echo "=== Time: $(date) ==="

exit $SWEEP_EXIT
