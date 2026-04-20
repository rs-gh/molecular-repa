#!/bin/bash
#!
#! SLURM job script for Proteina FID evaluation
#! Wilkes3 (AMD EPYC 7763, ConnectX-6, A100 80GB)
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/evaluation/eval_fid.sh inference_fid_60m_baseline                   # flat (symlink)
#!   sbatch hpc-scripts/proteina/evaluation/eval_fid.sh inference_fid_60m_baseline inference          # subdir
#!   sbatch hpc-scripts/proteina/evaluation/eval_fid.sh inference_fid_60m_repa inference
#!

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk

#! Output logs on RDS to avoid filling /home (50GB quota):
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/eval-fid-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/eval-fid-%j.err

#! Do not change:
#SBATCH -p ampere

###############################################################
### Parse arguments                                         ###
###############################################################

CONFIG_NAME="${1:?Usage: sbatch eval_fid.sh <config_name> [config_subdir] [--designability_subset N] ...}"
CONFIG_SUBDIR="${2:-}"
# If second arg starts with --, it's an extra arg not a subdir
if [[ "$CONFIG_SUBDIR" == --* ]]; then
    CONFIG_SUBDIR=""
    shift
else
    shift 2 2>/dev/null || shift
fi
EXTRA_ARGS="$@"

#! Derive a descriptive job name from the config (e.g. inference_fid_60m_repa_layer0 -> eval-repa-l0)
SHORT_NAME=$(echo "$CONFIG_NAME" | sed 's/inference_fid_60m_/eval-/' | sed 's/_v[0-9]*//' | sed 's/repa_layer/repa-l/' | sed 's/_/-/g')

###############################################################
### Environment setup                                       ###
###############################################################

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
PROTEINA_DIR="$REPO_DIR/src/proteina/proteinfoundation"

#! Activate venv
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

#! Data and environment
export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

#! Ensure log directory exists
mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

###############################################################
### ProteinMPNN CA weights (for designability metric)       ###
###############################################################

#! ProteinMPNN itself is vendored at src/proteina/ProteinMPNN/ (2 files, no weights).
#! Stash the CA weights on RDS (persistent across jobs, outside /home 50GB quota).
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

#! load_esmfold() at designability.py:149-166 reads CACHE_DIR. Compute-node /tmp
#! is too small for ESMFold's staged download; xet_get writes to TMPDIR and
#! fails with ENOSPC even with HF_HOME set — disable xet and route TMPDIR to RDS.
export CACHE_DIR="/rds/user/sr2173/hpc-work/proteina/hf_cache"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_HUB_DISABLE_XET=1
export TMPDIR="/rds/user/sr2173/hpc-work/proteina/tmp"
mkdir -p "$CACHE_DIR/hub" "$TMPDIR"

###############################################################
### Clean stale caches from previous runs on this node      ###
###############################################################

echo "Clearing stale caches..."
rm -rf /tmp/torchinductor_${USER} 2>/dev/null
find "$REPO_DIR/src/proteina" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
echo "Done"

###############################################################
### Diagnostics                                             ###
###############################################################

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== Config: $CONFIG_NAME ==="
echo "=== Time: $(date) ==="
echo ""

###############################################################
### GPU utilization monitor (background, 30s intervals)     ###
###############################################################

GPU_LOG="/rds/user/sr2173/hpc-work/proteina/logs/gpu-util-eval-${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 30 > "$GPU_LOG" &
GPU_MON_PID=$!

###############################################################
### Run FID evaluation                                      ###
###############################################################

cd "$REPO_DIR"

#! Import pyg_compat shim BEFORE any proteina code to replace broken torch_scatter/
#! torch_cluster C extensions with pure-PyTorch implementations (same as training).
#! Also add parent dir to sys.path so graphein_utils is importable.
python -u -c "
import os, sys
sys.path.insert(0, 'src/proteina/proteinfoundation')
sys.path.insert(0, 'src/proteina')

import proteinfoundation.repa.pyg_compat  # patches sys.modules

# PyTorch 2.6+ defaults weights_only=True; our checkpoints contain omegaconf objects.
# Lightning explicitly passes weights_only=True, so force it to False.
# Also strip _orig_mod. prefix from torch.compile and remove REPA training-only keys.
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

argv = ['evaluate.py', '--config_name', '$CONFIG_NAME']
if '$CONFIG_SUBDIR':
    argv += ['--config_subdir', '$CONFIG_SUBDIR']
sys.argv = argv + '$EXTRA_ARGS'.split()
import runpy
runpy.run_path('evaluation/proteina/generation/scripts/evaluate.py', run_name='__main__')
"
EVAL_EXIT=$?

###############################################################
### Cleanup and summary                                     ###
###############################################################

#! Stop GPU monitor
kill $GPU_MON_PID 2>/dev/null
wait $GPU_MON_PID 2>/dev/null

echo ""
echo "=== GPU utilization summary ==="
python -c "
import csv
with open('$GPU_LOG') as f:
    reader = csv.reader(f)
    header = next(reader)
    utils = []
    for row in reader:
        try:
            val = int(row[1].strip().replace(' %', ''))
            utils.append(val)
        except (ValueError, IndexError):
            pass
if utils:
    print(f'GPU utilization: min={min(utils)}%, max={max(utils)}%, mean={sum(utils)/len(utils):.1f}%, samples={len(utils)}')
else:
    print('No GPU utilization data collected')
"

echo ""
echo "=== Results ==="
RESULTS_CSV="$REPO_DIR/eval_output/${CONFIG_NAME}/results_${CONFIG_NAME}_fid.csv"
if [ -f "$RESULTS_CSV" ]; then
    echo "Results saved to: $RESULTS_CSV"
    #! Append training step/epoch to CSV and print metric columns
    python -c "
import pandas as pd, torch
df = pd.read_csv('$RESULTS_CSV')

# Read training step from checkpoint
ckpt_path = df['ckpt_path'].iloc[0] + '/' + df['ckpt_name'].iloc[0]
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
df.insert(0, 'global_step', ckpt.get('global_step', -1))
df.insert(1, 'epoch', ckpt.get('epoch', -1))
df.to_csv('$RESULTS_CSV', index=False)

metric_cols = ['global_step', 'epoch'] + [c for c in df.columns if c.startswith('_res_')]
print(df[metric_cols].to_string(index=False))
"
else
    echo "WARNING: Results CSV not found at $RESULTS_CSV"
fi

echo ""
echo "=== EVALUATION COMPLETE (exit code: $EVAL_EXIT) ==="
echo "=== Time: $(date) ==="

exit $EVAL_EXIT
