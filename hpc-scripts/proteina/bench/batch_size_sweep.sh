#!/bin/bash
#!
#! Batch size sweep: find max BS for each (seq_len, model_type) on A100 80GB.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/bench/batch_size_sweep.sh
#!   sbatch hpc-scripts/proteina/bench/batch_size_sweep.sh --seq_lens 128 256 --max_bs 64
#!

#SBATCH -J bs-sweep
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/bs-sweep-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/bs-sweep-%j.err
#SBATCH -p ampere

EXTRA_ARGS="$@"

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

# Clear stale caches
rm -rf /tmp/torchinductor_${USER} 2>/dev/null

echo "=== BATCH SIZE SWEEP ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo ""

cd "$REPO_DIR"

OUTPUT_DIR="evaluation/proteina/results/batch_size_sweep"
mkdir -p "$OUTPUT_DIR"

python -u hpc-scripts/proteina/bench/batch_size_sweep.py \
    --seq_lens 128 256 512 \
    --model_types baseline repa \
    --min_bs 4 --max_bs 128 \
    --num_steps 20 \
    --timeout 300 \
    --all_variants \
    --output_csv "$OUTPUT_DIR/batch_size_results.csv" \
    $EXTRA_ARGS

echo ""
echo "=== SWEEP COMPLETE ==="
echo "=== Time: $(date) ==="
