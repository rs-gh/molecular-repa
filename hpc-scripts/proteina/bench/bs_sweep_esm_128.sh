#!/bin/bash
#!
#! Batch-size probe scoped to ESM (ESM2-650M) REPA at seq_len=128.
#! Uses intr QoS (1h max wall) for fast queueing.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/bench/bs_sweep_esm_128.sh
#!

#SBATCH -J bs-esm-128
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:55:00
#SBATCH --qos=intr
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/home/sr2173/sdpa_diag_tmp/bs-esm-128-%j.out
#SBATCH --error=/home/sr2173/sdpa_diag_tmp/bs-esm-128-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export LMDB_DIR="/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb"
export ESM_MODEL_PATH="/rds/user/sr2173/hpc-work/proteina/hf_cache/esm2_t33_650M_UR50D"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

rm -rf /tmp/torchinductor_${USER} 2>/dev/null

echo "=== ESM 128 BATCH-SIZE PROBE ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo ""

cd "$REPO_DIR"

OUTPUT_DIR="evaluation/proteina/bench/results/batch_size_sweep"
mkdir -p "$OUTPUT_DIR"

python -u hpc-scripts/proteina/bench/batch_size_sweep.py \
    --seq_lens 128 \
    --model_types repa \
    --encoder_type esm \
    --min_bs 4 --max_bs 128 \
    --num_steps 10 \
    --timeout 300 \
    --output_csv "$OUTPUT_DIR/batch_size_results_esm_128_wide.csv"
PY_EXIT=$?

echo ""
echo "=== DONE: $(date) (python exit=$PY_EXIT) ==="
exit $PY_EXIT
