#!/bin/bash
#! Diagnostic smoke test — baseline timing + GPU util + /dev/shm check
#SBATCH -J prot-diag
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-diag-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-diag-%j.err
#SBATCH -p ampere

module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

cd "$REPO_DIR/src/proteina/proteinfoundation"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"

# Clear stale checkpoints from previous smoke tests so we don't try to resume
rm -rf ./store/proteina_smoke_test_single/checkpoints/last.ckpt 2>/dev/null

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== Time: $(date) ==="

# --- System diagnostics ---
echo ""
echo "=== /dev/shm (shared memory) ==="
df -h /dev/shm

echo ""
echo "=== ulimit (open files, max user processes) ==="
ulimit -n   # open files
ulimit -u   # max user processes

echo ""
echo "=== CPU info ==="
nproc
lscpu | grep "Model name"

echo ""
echo "=== Python multiprocessing start method ==="
python -c "import multiprocessing; print(f'Default start method: {multiprocessing.get_start_method()}')"

# --- Background GPU utilization logging ---
echo ""
echo "=== Starting GPU utilization monitor (every 2s) ==="
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 2 > /rds/user/sr2173/hpc-work/proteina/logs/gpu-util-${SLURM_JOB_ID}.csv &
GPU_MON_PID=$!

# --- Baseline training with num_workers=0 (current config) ---
echo ""
echo "=== BASELINE: num_workers=0, 3 epochs ==="
echo "=== Start: $(date) ==="

python train_repa.py --config_name smoke_test --single --nolog --show_prog_bar

echo "=== End: $(date) ==="

# --- Stop GPU monitor ---
kill $GPU_MON_PID 2>/dev/null
wait $GPU_MON_PID 2>/dev/null

echo ""
echo "=== GPU utilization summary ==="
# Print min/max/mean GPU util from the CSV
python -c "
import csv
with open('/rds/user/sr2173/hpc-work/proteina/logs/gpu-util-${SLURM_JOB_ID}.csv') as f:
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
    # Count time spent at 0%
    idle = sum(1 for u in utils if u == 0)
    print(f'Time at 0% (idle/loading): {idle}/{len(utils)} samples ({100*idle/len(utils):.1f}%)')
else:
    print('No GPU utilization data collected')
"

echo ""
echo "=== DIAGNOSTIC COMPLETE ==="
echo "=== Time: $(date) ==="
