#!/bin/bash
#!
#! SLURM job script for Proteina (baseline or REPA) on PDB with 2-GPU DDP.
#! Mirrors hpc-scripts/proteina/training/pdb/train_repa.sh but requests 2 GPUs
#! and launches train_repa.py via `srun` so Lightning's SLURM auto-detection
#! assigns one rank per GPU (DDP). The chosen config must set
#! `hardware.ngpus_per_node_: 2` so Trainer.devices matches the SLURM layout.
#!
#! Effective batch size = num_gpus * datamodule.batch_size * accumulate_grad_batches.
#! For bs=24 effective with 2 GPUs, configs should set batch_size=12 per GPU
#! (e.g. via `dataset: pdb_lmdb_256_bs12`).
#!
#! Usage:
#!   sbatch .../train_2gpu.sh training_baseline_256_bs24_2gpu training/256
#!   sbatch .../train_2gpu.sh training_repa_l4_256_per_residue_bs24_2gpu \
#!                            training/256/gearnet/per_residue
#!

#SBATCH -J prot-2gpu
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=36:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/repa-2gpu-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/repa-2gpu-%j.err
#SBATCH -p ampere
#SBATCH --nice=1  # Tiebreaker so AFDB jobs queue ahead of PDB; -1 vs other users is negligible

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
PROTEINA_DIR="$REPO_DIR/src/proteina/proteinfoundation"

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export ESM_MODEL_PATH="/rds/user/sr2173/hpc-work/proteina/hf_cache/esm2_t33_650M_UR50D"
export PROTEINMPNN_WEIGHTS_DIR="/rds/user/sr2173/hpc-work/proteina/ProteinMPNN"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_INIT_TIMEOUT=120
export WANDB__SERVICE_WAIT=300
export WANDB_DIR="/rds/user/sr2173/hpc-work/proteina/wandb_runs"

REPA_CONFIG="${1:?config name required}"
REPA_SUBDIR="${2:?config subdir required}"

JOB_SHORT=$(echo "$REPA_CONFIG" | sed 's/^training_//' | sed 's/repa_layer/repa_l/' | sed 's/_/-/g')
scontrol update JobId="$SLURM_JOB_ID" JobName="2gpu-$JOB_SHORT" 2>/dev/null

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID, NTASKS: $SLURM_NTASKS ==="
echo "=== Config: ${REPA_SUBDIR:+$REPA_SUBDIR/}$REPA_CONFIG ==="
echo "=== Time: $(date) ==="
echo ""

# GPU utilization monitor (one log; queries all visible GPUs)
GPU_LOG="/rds/user/sr2173/hpc-work/proteina/logs/gpu-util-2gpu-${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 10 > "$GPU_LOG" &
GPU_MON_PID=$!

# Copy PDB LMDB to local NVMe (shared by both ranks)
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
            fi
        done
        export LMDB_DIR="$LMDB_LOCAL"
    else
        echo "=== LMDB_SOURCE: lustre (not enough /tmp space) ==="
        export LMDB_DIR="$LMDB_SRC"
    fi
else
    echo "WARNING: LMDB not found at $LMDB_SRC"
    export LMDB_DIR="$LMDB_SRC"
fi

echo ""
echo "=== Launching DDP via srun (2 tasks, 1 GPU per task) ==="
echo ""

cd "$PROTEINA_DIR"

# srun launches one Python process per task; Lightning auto-detects SLURM env
# (SLURM_NTASKS=2, SLURM_LOCALID, SLURM_PROCID) and binds DDP rank-per-GPU.
# Each rank's PyTorch sees one visible GPU via Slurm's GPU binding.
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK 2>/dev/null
srun --cpus-per-task=12 --kill-on-bad-exit=1 python -u -c "
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

kill $GPU_MON_PID 2>/dev/null
wait $GPU_MON_PID 2>/dev/null

echo ""
echo "=== TRAINING COMPLETE (exit code: $TRAIN_EXIT) ==="
echo "=== Time: $(date) ==="
exit $TRAIN_EXIT
