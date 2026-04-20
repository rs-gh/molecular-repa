#!/bin/bash
#!
#! Run playground/proteina/gearnet/explore_gearnet.py on a GPU node.
#! GearNet itself runs fine on CPU, but we use a GPU node for parity with the
#! ESM playground and because the `intr` QOS has better queue behavior.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/smoke_tests/explore_gearnet.sh                        # default: 200 proteins
#!   sbatch hpc-scripts/proteina/smoke_tests/explore_gearnet.sh --n-proteins 500 --random-seed 0
#!

#SBATCH -J gearnet-explore
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --qos=intr
#SBATCH --output=/home/sr2173/git/molecular-repa/.local_ckpts/gearnet-explore-%j.out
#SBATCH --error=/home/sr2173/git/molecular-repa/.local_ckpts/gearnet-explore-%j.err
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

python -u playground/proteina/gearnet/explore_gearnet.py "$@"
