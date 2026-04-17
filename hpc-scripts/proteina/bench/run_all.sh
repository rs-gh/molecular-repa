#!/bin/bash
#!
#! Slurm wrapper: run all four proteina performance benchmarks back-to-back
#! on a single A100 node. CSVs land in evaluation/proteina/results/bench/.
#!
#! Total wall time budget: ~90 min
#!   - compile:  ~20 min (4 modes × 2 seq_lens × 100 steps + compile warmups)
#!   - io:       ~15 min (2 sources × 3 worker counts × 2 pin × 300 batches)
#!   - e2e:      ~35 min (4 compile-modes × 2 sources × 100 steps + copies)
#!   - sdpa:     ~20 min (3 backends × 2 compile × 100 steps)
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/bench/run_all.sh
#!   sbatch hpc-scripts/proteina/bench/run_all.sh --only compile
#!   sbatch hpc-scripts/proteina/bench/run_all.sh --only compile,io
#!

#SBATCH -J prot-bench
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/bench-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/bench-%j.err
#SBATCH -p ampere

ONLY="compile,io,e2e,sdpa"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

should_run() { [[ ",$ONLY," == *",$1,"* ]]; }

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

#! Clear stale inductor cache
rm -rf /tmp/torchinductor_${USER} 2>/dev/null

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: ${SLURM_JOB_ID:-none} ==="
echo "=== Start: $(date) ==="
echo "=== Benchmarks selected: $ONLY ==="
echo ""

cd "$REPO_DIR"
RESULTS_DIR="evaluation/proteina/results/bench"
mkdir -p "$RESULTS_DIR"

#! --- 1. Compile sweep (fake data, fast) ---
if should_run compile; then
    echo ""
    echo "############################################################"
    echo "### 1/4  COMPILE BENCHMARK"
    echo "############################################################"
    python -u hpc-scripts/proteina/bench/benchmark_compile.py \
        --seq_lens 256 512 \
        --model_types baseline \
        --compile_modes off default reduce-overhead \
        --num_steps 100 --warmup 30 \
        --output_csv "$RESULTS_DIR/compile.csv"
fi

#! --- 2. I/O: Lustre vs NVMe, worker sweep (no model) ---
if should_run io; then
    echo ""
    echo "############################################################"
    echo "### 2/4  I/O BENCHMARK"
    echo "############################################################"
    python -u hpc-scripts/proteina/bench/benchmark_io.py \
        --sources lustre nvme \
        --num_workers 0 2 4 \
        --pin_memory true false \
        --batch_size 6 --max_size 512 \
        --num_batches 300 --warmup 20 \
        --output_csv "$RESULTS_DIR/io.csv"
fi

#! --- 3. End-to-end: compile × lmdb_source ---
if should_run e2e; then
    echo ""
    echo "############################################################"
    echo "### 3/4  END-TO-END BENCHMARK"
    echo "############################################################"
    python -u hpc-scripts/proteina/bench/benchmark_e2e.py \
        --compile_modes off default \
        --lmdb_sources lustre nvme \
        --gc_layers 0 \
        --model_type baseline \
        --batch_size 6 --max_size 512 \
        --num_workers 2 \
        --num_steps 100 --warmup 30 \
        --output_csv "$RESULTS_DIR/e2e.csv"
fi

#! --- 4. SDPA kernel backends ---
if should_run sdpa; then
    echo ""
    echo "############################################################"
    echo "### 4/4  SDPA BACKEND BENCHMARK"
    echo "############################################################"
    python -u hpc-scripts/proteina/bench/benchmark_sdpa.py \
        --backends default efficient math flash \
        --compile_modes off default \
        --seq_len 512 --batch_size 6 \
        --num_steps 100 --warmup 30 \
        --output_csv "$RESULTS_DIR/sdpa.csv"
fi

echo ""
echo "=== End: $(date) ==="
echo "=== Results in: $RESULTS_DIR ==="
ls -la "$RESULTS_DIR"/
