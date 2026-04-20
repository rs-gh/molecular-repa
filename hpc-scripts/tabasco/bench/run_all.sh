#!/bin/bash
#!
#! SLURM wrapper: run all tabasco performance benchmarks back-to-back on one A100.
#! CSVs land in evaluation/tabasco/bench/results/.
#!
#! Approximate wall time:
#!   - compile:   ~15 min (3 modes × 2 precisions × 100 steps, fake data)
#!   - sdpa:      ~10 min (4 backends × 2 compile, fake data)
#!   - bs_sweep:  ~30 min (3 experiments × 2 compile × binary-search, fake data)
#!   - io:        ~5 min  (1 workers × 2 pin × 300 batches, real LMDB)
#!
#! Total ~60 min with --qos=intr (fast-start) per feedback_smoketest_qos_intr.md.
#!
#! Usage:
#!   sbatch hpc-scripts/tabasco/bench/run_all.sh
#!   sbatch hpc-scripts/tabasco/bench/run_all.sh --only compile
#!   sbatch hpc-scripts/tabasco/bench/run_all.sh --only compile,sdpa
#!

#SBATCH -J tabasco-bench
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --qos=intr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/tabasco/logs/bench-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/tabasco/logs/bench-%j.err
#SBATCH -p ampere

ONLY="compile,sdpa,bs_sweep,io"
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
module load python/3.11.0-icl

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export PROJECT_ROOT="$REPO_DIR/src/tabasco"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p /rds/user/sr2173/hpc-work/tabasco/logs
rm -rf /tmp/torchinductor_${USER} 2>/dev/null

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: ${SLURM_JOB_ID:-none} ==="
echo "=== Start: $(date) ==="
echo "=== Benchmarks selected: $ONLY ==="
echo ""

cd "$REPO_DIR"
RESULTS_DIR="evaluation/tabasco/bench/results"
mkdir -p "$RESULTS_DIR"

if should_run compile; then
    echo ""
    echo "############################################################"
    echo "### COMPILE BENCHMARK"
    echo "############################################################"
    python -u hpc-scripts/tabasco/bench/benchmark_compile.py \
        --experiments geom/mild \
        --compile_modes off default reduce-overhead \
        --precisions 16 bf16-mixed \
        --num_steps 100 --warmup 30 \
        --output_csv "$RESULTS_DIR/compile.csv"
fi

if should_run sdpa; then
    echo ""
    echo "############################################################"
    echo "### SDPA BACKEND BENCHMARK"
    echo "############################################################"
    python -u hpc-scripts/tabasco/bench/benchmark_sdpa.py \
        --backends default flash efficient math \
        --compile_modes off default \
        --experiment geom/mild \
        --precision bf16-mixed \
        --batch_size 256 \
        --num_steps 100 --warmup 30 \
        --output_csv "$RESULTS_DIR/sdpa.csv"

    echo ""
    echo "### SDPA DIAGNOSTIC (call shapes)"
    python -u hpc-scripts/tabasco/bench/diagnose_sdpa.py \
        > "$RESULTS_DIR/diagnose_sdpa.log" 2>&1 || true
    echo "saved: $RESULTS_DIR/diagnose_sdpa.log"
fi

if should_run bs_sweep; then
    echo ""
    echo "############################################################"
    echo "### BATCH SIZE SWEEP"
    echo "############################################################"
    python -u hpc-scripts/tabasco/bench/batch_size_sweep.py \
        --experiments geom/mild geom/chemprop_tradeoff geom/mace_cached_tradeoff \
        --min_bs 64 --max_bs 1024 \
        --num_steps 20 --warmup_steps 5 \
        --timeout 300 \
        --compile_modes off reduce-overhead \
        --gc_variants 0 \
        --output_csv "$RESULTS_DIR/batch_size_sweep.csv"
fi

if should_run io; then
    echo ""
    echo "############################################################"
    echo "### I/O BENCHMARK"
    echo "############################################################"
    python -u hpc-scripts/tabasco/bench/benchmark_io.py \
        --dataset geom \
        --num_workers 0 \
        --pin_memory true false \
        --batch_size 256 \
        --num_batches 300 --warmup 20 \
        --output_csv "$RESULTS_DIR/io.csv"
fi

echo ""
echo "=== End: $(date) ==="
echo "=== Results in: $RESULTS_DIR ==="
ls -la "$RESULTS_DIR"/ || true
