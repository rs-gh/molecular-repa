#!/bin/bash
#!
#! Profile MC-GearNet-Edge forward pass to identify the step-time bottleneck.
#! Times _local_idx, _build_edges, _build_line_graph, and GNN layers separately.
#! Also compares full encoder wrapper vs GearNet-CA at same batch size.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/smoke_tests/profile_mc_gearnet.sh
#!   sbatch hpc-scripts/proteina/smoke_tests/profile_mc_gearnet.sh --bs 40

#SBATCH -J profile-mc-gearnet
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --qos=intr
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/profile-mc-gearnet-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/profile-mc-gearnet-%j.err
#SBATCH -p ampere
#SBATCH --exclude=gpu-q-43

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
echo ""

cd "$REPO_DIR"
# Default to bs=40 (bs=80 OOMs in eager mode without torch.compile)
python -u playground/proteina/gearnet/profile_mc_gearnet.py --bs 40 "$@"

echo ""
echo "=== Done: $(date) ==="
