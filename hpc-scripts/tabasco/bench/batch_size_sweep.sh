#!/bin/bash
#!
#! Batch-size sweep for tabasco on A100 (Wilkes3 ampere partition).
#!
#! Usage:
#!   sbatch hpc-scripts/tabasco/bench/batch_size_sweep.sh
#!   sbatch hpc-scripts/tabasco/bench/batch_size_sweep.sh --experiments geom/mild --max_bs 1024
#!
#! Diagnostics are short (≤ 1h) — use --qos=intr per feedback_smoketest_qos_intr.md.
#!

#SBATCH -J tabasco-bs-sweep
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --qos=intr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/tabasco/logs/bs-sweep-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/tabasco/logs/bs-sweep-%j.err
#SBATCH -p ampere

EXTRA_ARGS="$@"

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
module load python/3.11.0-icl

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export PROJECT_ROOT="$REPO_DIR/src/tabasco"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p /rds/user/sr2173/hpc-work/tabasco/logs

rm -rf /tmp/torchinductor_${USER} 2>/dev/null

echo "=== TABASCO BATCH SIZE SWEEP ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo ""

cd "$REPO_DIR"

OUTPUT_DIR="evaluation/tabasco/bench/results"
mkdir -p "$OUTPUT_DIR"

python -u hpc-scripts/tabasco/bench/batch_size_sweep.py \
    --experiments geom/mild geom/chemprop_tradeoff geom/mace_cached_tradeoff \
    --min_bs 64 --max_bs 1024 \
    --num_steps 20 --warmup_steps 5 \
    --timeout 300 \
    --compile_modes off reduce-overhead \
    --gc_variants 0 \
    --output_csv "$OUTPUT_DIR/batch_size_sweep.csv" \
    $EXTRA_ARGS

echo ""
echo "=== SWEEP COMPLETE ==="
echo "=== Time: $(date) ==="
