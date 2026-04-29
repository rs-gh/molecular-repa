#!/bin/bash
#!
#! SLURM array job: generation sweep across multiple checkpoints.
#! Wilkes3 (AMD EPYC 7763, ConnectX-6, A100 80GB)
#!
#! Each array task runs one (run_name, step) pair and appends its result row
#! to a shared sweep_results.jsonl.  Use --dry_run first to print the full
#! task index table and confirm the --array range before submitting.
#!
#! Usage:
#!   # n=512 convergence curve - 11 tasks for baseline, 12 for repa variants.
#!   # Use --dry_run to get the exact count first.
#!   python evaluation/proteina/generation/scripts/run_sweep.py --config n512_convergence --dry_run
#!   sbatch --array=0-10 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n512_convergence
#!
#!   # sample-matched (4 tasks each)
#!   sbatch --array=0-3 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n128
#!   sbatch --array=0-3 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n256
#!   sbatch --array=0-3 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n512_sm
#!
#!   # backfill a single run within a profile
#!   sbatch --array=0-11 hpc-scripts/proteina/evaluation/generation/run_sweep.sh \
#!     --config n512_convergence --runs repa_l4
#!
#!   # ad-hoc single checkpoint (no --array needed)
#!   sbatch hpc-scripts/proteina/evaluation/generation/run_sweep.sh \
#!     --ckpt_path /rds/...ckpt --ckpt_label myrun_100k \
#!     --config_name inference/inference_fid_60m_baseline_lite
#!
#!   # override seed
#!   EVAL_SEED=123 sbatch --array=0-3 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n128
#!

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --job-name=gen-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/gen-sweep-%A_%a.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/gen-sweep-%A_%a.err
#SBATCH -p ampere

set -e

#! Random seed for generation and PDB subset sampling.
#! Override via: EVAL_SEED=123 sbatch ...
EVAL_SEED="${EVAL_SEED:-42}"

#! Pass all arguments through to run_sweep.py.
#! The script reads SLURM_ARRAY_TASK_ID from the environment.
PY_ARGS=("$@")

###############################################################
### Environment setup                                       ###
###############################################################

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
#! torch_scatter wheels need CXXABI_1.3.15 (gcc 13+); system libstdc++ only has 1.3.11.
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

REPO_DIR="/home/sr2173/git/molecular-repa"

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

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
### Clean stale caches                                      ###
###############################################################

rm -rf /tmp/torchinductor_${USER} 2>/dev/null
find "$REPO_DIR/src/proteina" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

###############################################################
### Diagnostics                                             ###
###############################################################

echo "=== Generation sweep ==="
echo "=== Args: ${PY_ARGS[*]} ==="
echo "=== SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-none} ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== EVAL_SEED: $EVAL_SEED ==="
echo "=== Time: $(date) ==="
echo ""

###############################################################
### GPU utilization monitor (background, 30s intervals)     ###
###############################################################

GPU_LOG="/rds/user/sr2173/hpc-work/proteina/logs/gpu-util-gen-sweep-${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 30 > "$GPU_LOG" &
GPU_MON_PID=$!

###############################################################
### Run sweep                                               ###
###############################################################

cd "$REPO_DIR"

python -u -c "
import os, sys
sys.path.insert(0, 'src/proteina/proteinfoundation')
sys.path.insert(0, 'src/proteina')

import proteinfoundation.repa.pyg_compat  # patches sys.modules

import torch
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    result = _orig_torch_load(*args, **kwargs)
    if isinstance(result, dict) and 'state_dict' in result:
        sd = result['state_dict']
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        result['state_dict'] = {k: v for k, v in sd.items() if not k.startswith('repa_loss.')}
    return result
torch.load = _patched_load

# Build argv: pass all shell args + inject seed
args = list('${PY_ARGS[*]}'.split()) if '${PY_ARGS[*]}' else []
if '--seed' not in args:
    args += ['--seed', '$EVAL_SEED']
sys.argv = ['run_sweep.py'] + args

import runpy
runpy.run_path('evaluation/proteina/generation/scripts/run_sweep.py', run_name='__main__')
"
SWEEP_EXIT=$?

###############################################################
### Cleanup and summary                                     ###
###############################################################

kill $GPU_MON_PID 2>/dev/null
wait $GPU_MON_PID 2>/dev/null

echo ""
echo "=== SWEEP TASK COMPLETE (exit code: $SWEEP_EXIT) ==="
echo "=== Time: $(date) ==="

exit $SWEEP_EXIT
