#!/bin/bash
#!
#! SLURM array job: Lite FID evaluation across intermediate checkpoints.
#! Generates ~300 PDBs per checkpoint (~35 min) for convergence curves.
#!
#! Usage:
#!   sbatch --array=0-10 hpc-scripts/proteina/eval_fid_lite_sweep.sh baseline
#!   sbatch --array=0-11 hpc-scripts/proteina/eval_fid_lite_sweep.sh repa
#!   sbatch --array=0-11 hpc-scripts/proteina/eval_fid_lite_sweep.sh repa_layer0
#!   sbatch --array=0-11 hpc-scripts/proteina/eval_fid_lite_sweep.sh repa_layer9
#!
#! Submit all at once:
#!   sbatch --array=0-10 hpc-scripts/proteina/eval_fid_lite_sweep.sh baseline && \
#!   sbatch --array=0-11 hpc-scripts/proteina/eval_fid_lite_sweep.sh repa && \
#!   sbatch --array=0-11 hpc-scripts/proteina/eval_fid_lite_sweep.sh repa_layer0 && \
#!   sbatch --array=0-11 hpc-scripts/proteina/eval_fid_lite_sweep.sh repa_layer9
#!

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/eval-lite-%A_%a.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/eval-lite-%A_%a.err
#SBATCH -p ampere

###############################################################
### Parse arguments                                         ###
###############################################################

RUN_TYPE="${1:?Usage: sbatch --array=0-N eval_fid_lite_sweep.sh <baseline|repa|repa_layer0|repa_layer9>}"

# Checkpoint steps (log-spaced, verified to exist on disk)
BASELINE_STEPS=(10000 20000 40000 80000 150000 250000 350000 450000 550000 650000 740000)
REPA_STEPS=(10000 20000 40000 80000 150000 250000 350000 450000 550000 650000 750000 840000)
REPA_L0_STEPS=(10000 20000 40000 80000 150000 250000 350000 450000 550000 650000 750000 830000)

case "$RUN_TYPE" in
    baseline)
        CONFIG_NAME="inference_fid_60m_baseline_lite"
        CKPT_BASE="/rds/user/sr2173/hpc-work/proteina/store/proteina_60m_baseline_v2/checkpoints"
        STEPS=("${BASELINE_STEPS[@]}")
        ;;
    repa)
        CONFIG_NAME="inference_fid_60m_repa_lite"
        CKPT_BASE="/rds/user/sr2173/hpc-work/proteina/store/proteina_60m_repa_v2/checkpoints"
        STEPS=("${REPA_STEPS[@]}")
        ;;
    repa_layer0)
        CONFIG_NAME="inference_fid_60m_repa_layer0_lite"
        CKPT_BASE="/rds/user/sr2173/hpc-work/proteina/store/proteina_60m_repa_layer0_v2/checkpoints"
        STEPS=("${REPA_L0_STEPS[@]}")
        ;;
    repa_layer9)
        CONFIG_NAME="inference_fid_60m_repa_layer9_lite"
        CKPT_BASE="/rds/user/sr2173/hpc-work/proteina/store/proteina_60m_repa_layer9_v2/checkpoints"
        STEPS=("${REPA_STEPS[@]}")
        ;;
    *)
        echo "ERROR: Unknown run type '$RUN_TYPE'. Use: baseline, repa, repa_layer0, repa_layer9"
        exit 1
        ;;
esac

STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}
if [ -z "$STEP" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (max ${#STEPS[@]}-1)"
    exit 1
fi

# Find the EMA checkpoint matching this step (filename includes epoch which varies)
PADDED_STEP=$(printf "%012d" "$STEP")
CKPT_NAME=$(ls "$CKPT_BASE" | grep "step=${PADDED_STEP}-EMA.ckpt$" | head -1)
if [ -z "$CKPT_NAME" ]; then
    echo "ERROR: No EMA checkpoint found for step $STEP in $CKPT_BASE"
    echo "Looking for pattern: step=${PADDED_STEP}-EMA.ckpt"
    ls "$CKPT_BASE" | tail -5
    exit 1
fi

###############################################################
### Environment setup                                       ###
###############################################################

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

###############################################################
### Clean stale caches                                      ###
###############################################################

rm -rf /tmp/torchinductor_${USER} 2>/dev/null
find "$REPO_DIR/src/proteina" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

###############################################################
### Diagnostics                                             ###
###############################################################

echo "=== Lite FID Sweep ==="
echo "=== Run: $RUN_TYPE, Step: $STEP, Checkpoint: $CKPT_NAME ==="
echo "=== Config: $CONFIG_NAME ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID, ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID ==="
echo "=== Time: $(date) ==="
echo ""

###############################################################
### Run lite FID evaluation                                 ###
###############################################################

cd "$REPO_DIR"

python -u -c "
import os, sys
sys.path.insert(0, 'src/proteina/proteinfoundation')
sys.path.insert(0, 'src/proteina')

import proteinfoundation.repa.pyg_compat

# Patch torch.load for OmegaConf objects and torch.compile prefixes
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

sys.argv = ['evaluate.py',
    '--config_name', '$CONFIG_NAME',
    '--ckpt_name_override', '$CKPT_NAME',
    '--output_suffix', 'step_${STEP}',
    '--designability_subset', '0',
    '--diversity_subset_per_bin', '0',
]
import runpy
runpy.run_path('evaluation/proteina/scripts/evaluate.py', run_name='__main__')
"
EVAL_EXIT=$?

###############################################################
### Summary                                                 ###
###############################################################

echo ""
echo "=== LITE EVAL COMPLETE (exit code: $EVAL_EXIT) ==="
echo "=== Time: $(date) ==="

RESULTS_CSV="$REPO_DIR/eval_output/${CONFIG_NAME}_step_${STEP}/results_${CONFIG_NAME}_step_${STEP}_fid.csv"
if [ -f "$RESULTS_CSV" ]; then
    echo "Results: $RESULTS_CSV"
    python -c "
import pandas as pd
df = pd.read_csv('$RESULTS_CSV')
metric_cols = [c for c in df.columns if c.startswith('_res_')]
print(df[metric_cols].to_string(index=False))
"
else
    echo "WARNING: Results CSV not found at $RESULTS_CSV"
fi

exit $EVAL_EXIT
