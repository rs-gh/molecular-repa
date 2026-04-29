#!/bin/bash
#!
#! PW GearNet-Edge characterization on 200 PDB proteins.
#! Mirrors encoder_profiling/proteina/mc_gearnet/run_mc_gearnet.sh.
#! Runs both the torsional_denoising and structure_denoising checkpoints.
#!
#! Usage: sbatch encoder_profiling/proteina/pw_gearnet/run_pw_gearnet.sh
#!

#SBATCH -J pw-gn-char
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --qos=intr
#SBATCH --output=/home/sr2173/git/molecular-repa/.local_ckpts/pw-gn-char-%j.out
#SBATCH --error=/home/sr2173/git/molecular-repa/.local_ckpts/pw-gn-char-%j.err
#SBATCH -p ampere
#SBATCH --exclude=gpu-q-39

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
echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi failed"
echo "=== CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="
echo "=== Time: $(date) ==="
echo "=== DATA_PATH: $DATA_PATH ==="

cd "$REPO_DIR"

CKPT_DIR="$DATA_PATH/metric_factory/model_weights"

for variant in torsional structure; do
    CKPT="$CKPT_DIR/pw_gearnet_${variant}_denoising_ca_angles.ckpt"
    if [[ ! -f "$CKPT" ]]; then
        echo "MISSING: $CKPT - skipping $variant"
        continue
    fi
    echo ""
    echo "########################################################################"
    echo "# Variant: $variant  ($(date))"
    echo "########################################################################"
    python -u encoder_profiling/proteina/pw_gearnet/explore_pw_gearnet.py \
        --ckpt "$CKPT" --variant "$variant" --n-proteins 200
done

echo "=== DONE: $(date) ==="
