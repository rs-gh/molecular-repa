#!/bin/bash
#!
#! SLURM job script for Proteina FID evaluation
#! Wilkes3 (AMD EPYC 7763, ConnectX-6, A100 80GB)
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/eval_fid.sh inference_fid_60m_baseline
#!   sbatch hpc-scripts/proteina/eval_fid.sh inference_fid_60m_repa
#!   sbatch hpc-scripts/proteina/eval_fid.sh inference_fid_60m_repa_layer0
#!   sbatch hpc-scripts/proteina/eval_fid.sh inference_fid_60m_repa_layer9
#!

#SBATCH -J prot-eval-fid
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

CONFIG_NAME="${1:?Usage: sbatch eval_fid.sh <config_name>}"

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

cd "$PROTEINA_DIR"

#! Import pyg_compat shim BEFORE inference.py to replace broken torch_scatter/
#! torch_cluster C extensions with pure-PyTorch implementations (same as training).
python -u -c "
import proteinfoundation.repa.pyg_compat  # patches sys.modules

import sys, runpy
sys.argv = ['inference.py', '--config_name', '$CONFIG_NAME']
runpy.run_path('inference.py', run_name='__main__')
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
RESULTS_CSV="$PROTEINA_DIR/inference/results_${CONFIG_NAME}_fid.csv"
if [ -f "$RESULTS_CSV" ]; then
    echo "Results saved to: $RESULTS_CSV"
    #! Print metric columns only
    python -c "
import pandas as pd
df = pd.read_csv('$RESULTS_CSV')
metric_cols = [c for c in df.columns if c.startswith('_res_')]
if metric_cols:
    print(df[metric_cols].to_string(index=False))
else:
    print('No metric columns found')
"
else
    echo "WARNING: Results CSV not found at $RESULTS_CSV"
fi

echo ""
echo "=== EVALUATION COMPLETE (exit code: $EVAL_EXIT) ==="
echo "=== Time: $(date) ==="

exit $EVAL_EXIT
