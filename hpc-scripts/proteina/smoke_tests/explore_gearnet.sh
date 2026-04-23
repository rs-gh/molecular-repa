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
#SBATCH --time=01:30:00
#SBATCH --qos=gpu1
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

SRC_DATA="${DATA_PATH:-/rds/user/sr2173/hpc-work/proteina/data}"
# Stage the LMDB (+ GearNet ckpt) to node-local NVMe — random reads over Lustre thrash.
# /local is NVMe (1.5TB); $TMPDIR is tmpfs (16GB, too small for 50GB LMDB).
STAGE="/local/$USER/$SLURM_JOB_ID"
mkdir -p "$STAGE/data/pdb_train/lmdb" "$STAGE/data/metric_factory/model_weights"
trap 'rm -rf "$STAGE"' EXIT

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo "=== Staging LMDB + GearNet ckpt to $STAGE ==="
t0=$SECONDS
cp "$SRC_DATA/pdb_train/lmdb/train.lmdb" "$STAGE/data/pdb_train/lmdb/"
cp "$SRC_DATA/pdb_train/lmdb/train_keys.pkl" "$STAGE/data/pdb_train/lmdb/"
cp "$SRC_DATA/metric_factory/model_weights/gearnet_ca.pth" "$STAGE/data/metric_factory/model_weights/"
echo "=== Staged in $((SECONDS-t0))s: $(du -sh $STAGE/data/) ==="

export DATA_PATH="$STAGE/data"

cd "$REPO_DIR"

python -u playground/proteina/gearnet/explore_gearnet.py "$@"
