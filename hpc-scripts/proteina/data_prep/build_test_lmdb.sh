#!/bin/bash
#! Build the held-out test.lmdb for the Proteina probe suite.
#!
#! Usage:
#!   # Tier 2 (intr) smoke — n=50, ~5 min:
#!   sbatch --qos=intr --time=00:15:00 hpc-scripts/proteina/data_prep/build_test_lmdb.sh \
#!       --n_test 50 --output_path /tmp/test_smoke50.lmdb --commit_every 10 --skip_train_check
#!
#!   # Full build — n=500, ~10-20 min on 16 CPUs:
#!   sbatch --qos=intr --time=00:30:00 hpc-scripts/proteina/data_prep/build_test_lmdb.sh \
#!       --n_test 500 --commit_every 100 --skip_train_check --resume
#!
#! The --resume flag is defensive: picks up partial LMDBs without re-parsing
#! completed entries. First run will find no existing entries (behaves like
#! fresh build); restarts after crashes skip committed work.

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --job-name=build-test-lmdb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/build-test-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/build-test-%j.err
#SBATCH -p ampere

set -e

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH=/rds/user/sr2173/hpc-work/proteina/data
export OMP_NUM_THREADS=1

cd "$REPO_DIR"

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="
echo "=== Args: $@ ==="

python -u hpc-scripts/proteina/data_prep/build_test_lmdb.py --num_workers 16 "$@"

echo "=== DONE, $(date) ==="
