#!/bin/bash
#!
#! CA-GearNet characterization with RANDOM-INIT weights (no pretrained ckpt).
#! Isolates what the architecture alone contributes vs. what the pretrained
#! weights add. Compare against the pretrained run in playground/proteina/gearnet/FINDINGS.md.
#!
#! Usage: sbatch playground/proteina/gearnet/run_gearnet_random_init.sh
#!

#SBATCH -J gn-rand
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --qos=intr
#SBATCH --output=/home/sr2173/git/molecular-repa/.local_ckpts/gn-rand-%j.out
#SBATCH --error=/home/sr2173/git/molecular-repa/.local_ckpts/gn-rand-%j.err
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
export PYTHONUNBUFFERED=1

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo "=== MODE: CA-GearNet random-init ==="

cd "$REPO_DIR"
python -u playground/proteina/gearnet/explore_gearnet.py \
    --n-proteins 200 \
    --random-init \
    --init-seed 0

echo "=== DONE: $(date) ==="
