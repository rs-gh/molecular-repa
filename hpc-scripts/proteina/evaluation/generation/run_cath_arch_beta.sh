#!/bin/bash
#!
#! One-off: run exp_cath_arch_concentrated_beta.py on a single GPU.
#! Targets the β≥25 designable subset for the concentrated-fold cases and
#! emits a per-A-class breakdown. Short (~10-30 min); uses intr QOS for
#! fast scheduling.
#!

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --job-name=cath-arch-beta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --qos=intr
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/cath-arch-beta-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/cath-arch-beta-%j.err
#SBATCH -p ampere

set -e

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

REPO_DIR="/home/sr2173/git/molecular-repa"
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== cath-arch-beta ==="
echo "=== node: $(hostname) ==="
echo "=== gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== time: $(date) ==="

cd "$REPO_DIR"
python -u evaluation/proteina/generation/scripts/paper/exp_cath_arch_concentrated_beta.py "$@"

echo "=== done: $(date) ==="
