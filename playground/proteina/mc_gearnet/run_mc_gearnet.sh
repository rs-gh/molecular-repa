#!/bin/bash
#!
#! MC-GearNet-Edge characterization on 200 PDB proteins.
#! Mirrors the GearNet-CA and ESM2 analyses:
#!   playground/proteina/gearnet/explore_gearnet.py
#!   playground/proteina/esm/FINDINGS.md
#!
#! Usage: sbatch playground/proteina/mc_gearnet/run_mc_gearnet.sh
#!

#SBATCH -J mc-gn-char
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --qos=intr
#SBATCH --output=/home/sr2173/git/molecular-repa/.local_ckpts/mc-gn-char-%j.out
#SBATCH --error=/home/sr2173/git/molecular-repa/.local_ckpts/mc-gn-char-%j.err
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
echo "=== DATA_PATH: $DATA_PATH ==="

cd "$REPO_DIR"
python -u playground/proteina/mc_gearnet/explore_mc_gearnet.py --n-proteins 200

echo "=== DONE: $(date) ==="
