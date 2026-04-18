#!/bin/bash
#!
#! SLURM job script for Proteina (baseline or REPA, GearNet/ESM alignment, torch.compile)
#! Wilkes3 (AMD EPYC 7763, ConnectX-6, A100 80GB)
#!
#! Config tree: training/<seq_len>/[<encoder>/[<averaging>/]]<config_name>.yaml
#!   - Baselines live at training/<seq_len>/ (no encoder subdir).
#!   - REPA configs live at training/<seq_len>/<encoder>/[<averaging>/],
#!     where <encoder> ∈ {gearnet, esm2, ...}.
#!
#! Usage (pass <config_name> and <config_subdir>):
#!   sbatch .../train_repa.sh training_repa training/512/gearnet                                  # 512 REPA gearnet (implicit layer 4)
#!   sbatch .../train_repa.sh training_baseline_256 training/256                                  # 256 baseline
#!   sbatch .../train_repa.sh training_repa_l4_256_per_residue training/256/gearnet/per_residue   # 256 REPA gearnet l4, per_residue
#!   sbatch .../train_repa.sh training_repa_l4_256_per_sample  training/256/gearnet/per_sample    # 256 REPA gearnet l4, per_sample
#!   sbatch .../train_repa.sh training_repa_l9_256_per_residue training/256/esm2/per_residue      # 256 REPA ESM-2, l9, per_residue
#!
#! Checkpoint resume is automatic: if a last.ckpt exists under the run's
#! store directory, train_repa.py picks it up and continues training.
#! WandB likewise resumes the same run (keyed by run_name_ in the config).
#!

#SBATCH -J prot-repa
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=36:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk

#! Output logs on RDS to avoid filling /home (50GB quota):
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/repa-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/repa-%j.err

#! Do not change:
#SBATCH -p ampere

###############################################################
### Environment setup                                       ###
###############################################################

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
PROTEINA_DIR="$REPO_DIR/src/proteina/proteinfoundation"

#! Activate venv (use project venv, not conda)
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

#! Data and environment (DATA_PATH must contain metric_factory/model_weights/gearnet_ca.pth)
export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
#! ESM-2-650M weights staged for REPA configs using `encoder.type: esm`.
#! Ignored by gearnet configs. Falls back to HF hub download if unset.
export ESM_MODEL_PATH="/rds/user/sr2173/hpc-work/proteina/hf_cache/esm2_t33_650M_UR50D"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_INIT_TIMEOUT=120

REPA_CONFIG="${1:-training_repa}"
REPA_SUBDIR="${2:-}"

#! Set descriptive SLURM job name from config (e.g. training_repa_l4_256_per_residue -> repa-l4-256-per-residue)
#! Normalize: training_repa (layer 4 implicit) -> repa-l4, training_repa_layer0 -> repa-l0
JOB_SHORT=$(echo "$REPA_CONFIG" | sed 's/^training_//' | sed 's/repa_layer/repa_l/' | sed 's/_/-/g')
# training_repa with no layer suffix is layer 4 — make that explicit
if [ "$JOB_SHORT" = "repa" ]; then JOB_SHORT="repa-l4"; fi
if [ "$JOB_SHORT" = "repa-v2" ]; then JOB_SHORT="repa-l4"; fi
scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_SHORT" 2>/dev/null

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
echo "=== Config: ${REPA_SUBDIR:+$REPA_SUBDIR/}$REPA_CONFIG ==="
echo "=== Time: $(date) ==="
echo ""

###############################################################
### GPU utilization monitor (background, 10s intervals)     ###
###############################################################

GPU_LOG="/rds/user/sr2173/hpc-work/proteina/logs/gpu-util-repa-${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 10 > "$GPU_LOG" &
GPU_MON_PID=$!

###############################################################
### Copy LMDB to local NVMe (avoid Lustre mmap thrashing)  ###
###############################################################

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
        echo "=== LMDB_SOURCE: lustre (not enough space on /tmp) ==="
        rm -rf "$LMDB_LOCAL"
        export LMDB_DIR="$LMDB_SRC"
    fi
else
    echo "WARNING: LMDB not found at $LMDB_SRC, using Lustre directly"
    echo "=== LMDB_SOURCE: lustre (not found) ==="
    export LMDB_DIR="$LMDB_SRC"
fi

echo ""

###############################################################
### Training                                                ###
###############################################################

cd "$PROTEINA_DIR"

#! Use spawn start method for num_workers > 0 (avoids fork+CUDA segfaults).
#! train_repa.py handles checkpoint auto-resume and WandB continuation
#! internally via fetch_last_ckpt() and WandbLogger(id=run_name, resume=...).
python -u -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import sys, runpy
argv = ['train_repa.py', '--config_name', '${REPA_CONFIG}', '--show_prog_bar']
if '${REPA_SUBDIR}':
    argv += ['--config_subdir', '${REPA_SUBDIR}']
sys.argv = argv
runpy.run_path('train_repa.py', run_name='__main__')
"
TRAIN_EXIT=$?

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
    idle = sum(1 for u in utils if u == 0)
    print(f'Time at 0% (idle/loading): {idle}/{len(utils)} samples ({100*idle/len(utils):.1f}%)')
    high = sum(1 for u in utils if u >= 80)
    print(f'Time at 80%+: {high}/{len(utils)} samples ({100*high/len(utils):.1f}%)')
else:
    print('No GPU utilization data collected')
"

echo ""
echo "=== TRAINING COMPLETE (exit code: $TRAIN_EXIT) ==="
echo "=== Time: $(date) ==="

exit $TRAIN_EXIT
