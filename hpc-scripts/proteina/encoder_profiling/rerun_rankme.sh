#!/bin/bash
#! Re-run all 5 proteina encoder probes with the new RankMe metric.
#!
#! After lib.py was switched from "entropy on squared centered SVs" to
#! Roy-Vetterli on raw uncentered SVs (Garrido et al. 2023), every encoder's
#! results.json needs regenerating. The driver is encoder_profiling/proteina/
#! run_all.sh; this just wraps it for SLURM with intr-queue priority.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/encoder_profiling/rerun_rankme.sh smoke
#!   sbatch hpc-scripts/proteina/encoder_profiling/rerun_rankme.sh full

#SBATCH -J rerun-rankme
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --qos=intr
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/rerun-rankme-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/rerun-rankme-%j.err
#SBATCH -p ampere

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
cd "$REPO_DIR"

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

MODE="${1:-full}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
    echo "Usage: sbatch $0 <smoke|full>" >&2
    exit 1
fi

scontrol update JobId="$SLURM_JOB_ID" JobName="rerun-rankme-${MODE}" 2>/dev/null || true

echo "=== rerun-rankme MODE=$MODE ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== Time: $(date) ==="
echo "=== DATA_PATH=$DATA_PATH ==="
echo ""

bash "$REPO_DIR/encoder_profiling/proteina/run_all.sh" "$MODE"

echo ""
echo "=== Done at $(date) ==="
