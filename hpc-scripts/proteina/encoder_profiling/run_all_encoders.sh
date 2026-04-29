#!/bin/bash
#!
#! Run the full proteina encoder-comparison sweep back-to-back on one GPU node:
#!   - CA-GearNet (trained)
#!   - CA-GearNet random-init (3 seeds)
#!   - ESM-2 650M
#!   - MC-GearNet-Edge
#!   - PW-GearNet-Edge (torsional)
#! then collate -> encoder_profiling/proteina/{comparison.csv,figures/}
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/encoder_profiling/run_all_encoders.sh
#!   sbatch hpc-scripts/proteina/encoder_profiling/run_all_encoders.sh --n-proteins 100
#!

#SBATCH -J enc-sweep
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00  # observed 12:33 on 2026-04-29 (job 28596016); 30m gives ~2.4x headroom
#SBATCH --qos=gpu1
#SBATCH --output=/home/sr2173/git/molecular-repa/.local_ckpts/enc-sweep-%j.out
#SBATCH --error=/home/sr2173/git/molecular-repa/.local_ckpts/enc-sweep-%j.err
#SBATCH -p ampere

set -euo pipefail

N_PROTEINS=200
SEED=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-proteins) N_PROTEINS="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

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
export PYTHONUNBUFFERED=1

SRC_DATA="${DATA_PATH:-/rds/user/sr2173/hpc-work/proteina/data}"
STAGE="/local/$USER/$SLURM_JOB_ID"
mkdir -p "$STAGE/data/pdb_train/lmdb" "$STAGE/data/metric_factory/model_weights"
trap 'rm -rf "$STAGE"' EXIT

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Start: $(date) ==="
echo "=== n_proteins=$N_PROTEINS  seed=$SEED ==="

t0=$SECONDS
echo "=== Staging LMDB + encoder weights to $STAGE ==="
cp "$SRC_DATA/pdb_train/lmdb/train.lmdb"           "$STAGE/data/pdb_train/lmdb/"
cp "$SRC_DATA/pdb_train/lmdb/train_keys.pkl"       "$STAGE/data/pdb_train/lmdb/"
cp "$SRC_DATA/metric_factory/model_weights/gearnet_ca.pth"        "$STAGE/data/metric_factory/model_weights/"
cp "$SRC_DATA/metric_factory/model_weights/mc_gearnet_edge.pth"   "$STAGE/data/metric_factory/model_weights/"
cp "$SRC_DATA/metric_factory/model_weights/pw_gearnet_torsional_denoising_ca_angles.ckpt" \
   "$STAGE/data/metric_factory/model_weights/"
echo "=== Staged in $((SECONDS-t0))s: $(du -sh $STAGE/data/) ==="
export DATA_PATH="$STAGE/data"

cd "$REPO_DIR"
HERE="encoder_profiling/proteina"
PW_CKPT="$STAGE/data/metric_factory/model_weights/pw_gearnet_torsional_denoising_ca_angles.ckpt"

step() { echo; echo "############# $1 (t+$((SECONDS-t0))s) #############"; }

step "1/5 CA-GearNet (trained)"
python -u "$HERE/gearnet/explore_gearnet.py" --n-proteins "$N_PROTEINS" --random-seed "$SEED"

step "2/5 CA-GearNet random-init x3 seeds"
python -u "$HERE/gearnet_random/explore_gearnet_random.py" \
    --n-proteins "$N_PROTEINS" --random-seed "$SEED" --init-seeds 0 1 2

step "3/5 ESM-2 650M"
python -u "$HERE/esm/explore_esm.py" --n-proteins "$N_PROTEINS" --random-seed "$SEED"

step "4/5 MC-GearNet-Edge"
python -u "$HERE/mc_gearnet/explore_mc_gearnet.py" --n-proteins "$N_PROTEINS" --random-seed "$SEED"

step "5/5 PW-GearNet-Edge (torsional)"
python -u "$HERE/pw_gearnet/explore_pw_gearnet.py" --n-proteins "$N_PROTEINS" --random-seed "$SEED" \
    --ckpt "$PW_CKPT" --variant torsional

step "Collating"
python -u "$HERE/collate.py"

echo
echo "=== End: $(date)  total $((SECONDS-t0))s ==="
echo "=== See $HERE/comparison.csv and $HERE/figures/ ==="
