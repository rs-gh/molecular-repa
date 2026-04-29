#!/bin/bash
#!
#! P1 - per-bucket batch-size calibration for length-bucketed sampling plan.
#!
#! Measures max viable BS at 7 seq_len anchors (128, 192, 256, 320, 384,
#! 448, 512), baseline only, compile=True, no gradient checkpointing.
#! Output drives the choice of `bucket_batch_sizes` in the future
#! LengthBucketedBatchSampler config.
#!
#! Isolated: reuses batch_size_sweep.py (bench tooling, never imported by
#! training). Writes to a dedicated CSV path. Does not touch any training
#! config, model code, or checkpoint.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/bench/bs_sweep_p1_bucket_calibration.sh

#SBATCH -J p1-bucket-calib
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/p1-bucket-calib-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/p1-bucket-calib-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export LMDB_DIR="/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

rm -rf /tmp/torchinductor_${USER} 2>/dev/null

echo "=== P1: Per-bucket batch-size calibration ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo ""

cd "$REPO_DIR"

OUTPUT_DIR="evaluation/proteina/bench/results/batch_size_sweep"
mkdir -p "$OUTPUT_DIR"

# 7 seq_len anchors, baseline only, compile=True, no gc.
# No --all_variants: uses default --compile=True --gc_layers=0.
python -u hpc-scripts/proteina/bench/batch_size_sweep.py \
    --seq_lens 128 192 256 320 384 448 512 \
    --model_types baseline \
    --min_bs 1 --max_bs 128 \
    --num_steps 20 \
    --timeout 300 \
    --output_csv "$OUTPUT_DIR/p1_bucket_calibration.csv"

echo ""
echo "=== DONE: $(date) ==="
