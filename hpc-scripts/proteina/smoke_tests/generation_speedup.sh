#!/bin/bash
#!
#! SLURM smoke / benchmark / validation wrapper for the generation-speedup playground.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/smoke_tests/generation_speedup.sh explore    # ~5 min
#!   sbatch hpc-scripts/proteina/smoke_tests/generation_speedup.sh bench      # ~30-60 min (A vs D default)
#!   sbatch hpc-scripts/proteina/smoke_tests/generation_speedup.sh validate   # ~20-40 min (4 modes x 50 samples)
#!

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --qos=intr
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/gen-speedup-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/gen-speedup-%j.err
#SBATCH -p ampere

set -e

MODE="${1:?Usage: sbatch generation_speedup.sh <explore|bench|validate> [extra args...]}"
shift  # rest are extra args forwarded to the python script
EXTRA_ARGS="$@"

case "$MODE" in
    explore)   SCRIPT="playground/proteina/generation_speedup/explore.py"  ;;
    bench)     SCRIPT="playground/proteina/generation_speedup/bench.py"    ;;
    validate)  SCRIPT="playground/proteina/generation_speedup/validate.py" ;;
    *)         echo "Unknown mode: $MODE (use explore|bench|validate)"; exit 1 ;;
esac

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$REPO_DIR"

rm -rf /tmp/torchinductor_${USER} 2>/dev/null
find "$REPO_DIR/src/proteina" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

echo "=== Generation speedup: $MODE ==="
echo "=== Script: $SCRIPT $EXTRA_ARGS ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== Time: $(date) ==="
echo ""

python -u "$SCRIPT" $EXTRA_ARGS
EXIT=$?
echo ""
echo "=== DONE (exit $EXIT) ==="
echo "=== Time: $(date) ==="
exit $EXIT
