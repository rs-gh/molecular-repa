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
#!   # n=256 paper protocol (2400 PDBs FID, 250 PDBs des/div, epoch matched)
#!   sbatch --array=0-7 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n256_paper_pdb
#!   sbatch --array=0-5 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n256_paper_random
#!   sbatch --array=0-3 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n256_paper_afdb
#!
#! --time bumped from 01:30 to 02:30 because the paper-protocol DES tasks
#! (250 PDBs × 8 ESMFold = 2000 calls at avg L=150) take ~1h45 — designability
#! is the bottleneck. Lite tasks (~70 min) still fit comfortably.
#!

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --job-name=gen-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:30:00
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

#! Foldseek for the novelty pipeline. evaluate.py / novelty_foldseek.py shells
#! out to `foldseek easy-search` if foldseek_target_dbs is set in
#! sweep_config.yaml; if foldseek isn't on PATH the wrapper logs and skips.
export PATH="/rds/user/sr2173/hpc-work/tools/foldseek/bin:$PATH"

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

###############################################################
### ProteinMPNN CA weights (designability)                  ###
###############################################################
#! Without these, every designability sample silently fails (ProteinMPNN can't
#! find weights at its default path -> .fa file never written -> rate=nan).
#! Same setup as eval_designability_only.sh.

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
### Per-job-scoped torchinductor cache                      ###
###############################################################
#! Concurrent array tasks used to race on /tmp/torchinductor_${USER}: when
#! task N+1 starts, its `rm -rf` would clobber task N's mid-compile state
#! and hang task N indefinitely. We now scope to the SLURM array task so
#! every task has a private cache dir. Setting TORCHINDUCTOR_CACHE_DIR
#! tells torch.compile where to write; the rm cleans up any leftover from
#! a prior run with the SAME job ID (rare but possible after a node
#! reschedule). __pycache__ wipe stays — bytecode staleness across edits
#! is a separate problem.
TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${SLURM_JOB_ID:-nojob}_${SLURM_ARRAY_TASK_ID:-0}"
export TORCHINDUCTOR_CACHE_DIR
rm -rf "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
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
sys.path.insert(0, 'evaluation/proteina')

import proteinfoundation.repa.pyg_compat  # patches sys.modules

# Centralised torch.load patcher (weights_only=False, _orig_mod. prefix strip,
# repa_loss.* filtering for the generation flavour). Old inline patch used
# to live here; see lib/torch_load_patch.py for the full rationale and the
# representation flavour (strip_repa=False).
from lib.torch_load_patch import apply as _apply_torch_load_patch
_apply_torch_load_patch(strip_repa=True)

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

#! Free the per-task torchinductor cache from /tmp. Each cache is scoped to
#! ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} so this only nukes our own task's
#! cache, never another running task's. Leaving these around long-term
#! would chew /tmp space (several GB per task across many shapes).
rm -rf "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null

echo ""
echo "=== SWEEP TASK COMPLETE (exit code: $SWEEP_EXIT) ==="
echo "=== Time: $(date) ==="

exit $SWEEP_EXIT
