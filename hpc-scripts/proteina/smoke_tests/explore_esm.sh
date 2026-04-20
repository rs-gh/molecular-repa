#!/bin/bash
#!
#! Run playground/proteina/esm/explore_esm.py on a GPU node.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/smoke_tests/explore_esm.sh          # full: 200 proteins
#!   QUICK=1 sbatch hpc-scripts/proteina/smoke_tests/explore_esm.sh  # 20-protein smoke
#!

#SBATCH -J esm-explore
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --qos=intr
#SBATCH --output=/home/sr2173/git/molecular-repa/.local_ckpts/esm-explore-%j.out
#SBATCH --error=/home/sr2173/git/molecular-repa/.local_ckpts/esm-explore-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

REPO_DIR="/home/sr2173/git/molecular-repa"

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export DATA_PATH="${DATA_PATH:-/rds/user/sr2173/hpc-work/proteina/data}"

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Time: $(date) ==="

cd "$REPO_DIR"

EXTRA_ARGS=""
if [[ "${QUICK:-0}" == "1" ]]; then
  EXTRA_ARGS="--quick"
fi

python -u playground/proteina/esm/explore_esm.py ${EXTRA_ARGS} "$@"
