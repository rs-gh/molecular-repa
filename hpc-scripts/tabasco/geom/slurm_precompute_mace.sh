#!/bin/bash
#! Pre-compute MACE embeddings for GEOM dataset (train, val, test splits)
#! Runs on a single A100 GPU. Expected time: ~1 hour for train, minutes for val/test.

#SBATCH -J precompute-mace-geom
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/tabasco/logs/slurm-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/tabasco/logs/slurm-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

source /home/sr2173/git/molecular-repa/.venv/bin/activate
cd /home/sr2173/git/molecular-repa
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PROJECT_ROOT=/home/sr2173/git/molecular-repa/src/tabasco

DATA_DIR=src/tabasco/data
OUT_DIR=/rds/user/sr2173/hpc-work/tabasco/data/mace_geom
mkdir -p "$OUT_DIR"

echo "=== Pre-computing MACE embeddings for GEOM ==="
echo "Start: $(date)"

echo ""
echo "--- Train split ---"
python encoder_profiling/tabasco/mace/precompute_embeddings.py \
    --lmdb-in  "$DATA_DIR/lmdb_geom/train.lmdb" \
    --lmdb-out "$OUT_DIR/train_embeddings.lmdb" \
    --batch-size 256 \
    --device cuda \
    --skip-existing

echo ""
echo "--- Val split ---"
python encoder_profiling/tabasco/mace/precompute_embeddings.py \
    --lmdb-in  "$DATA_DIR/lmdb_geom/val.lmdb" \
    --lmdb-out "$OUT_DIR/val_embeddings.lmdb" \
    --batch-size 256 \
    --device cuda

echo ""
echo "--- Test split ---"
python encoder_profiling/tabasco/mace/precompute_embeddings.py \
    --lmdb-in  "$DATA_DIR/lmdb_geom/test.lmdb" \
    --lmdb-out "$OUT_DIR/test_embeddings.lmdb" \
    --batch-size 256 \
    --device cuda

echo ""
echo "=== Done: $(date) ==="
ls -lh "$OUT_DIR"
