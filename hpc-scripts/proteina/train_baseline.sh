#!/bin/bash
#!
#! SLURM job script for Proteina baseline (60M, no REPA)
#! Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#!

#SBATCH -J prot-baseline
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=36:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk

#! Output logs on RDS to avoid filling /home:
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/slurm-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/slurm-%j.err

#! Do not change:
#SBATCH -p ampere

#! Environment setup
. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
module load python/3.11.0-icl

REPO_DIR="/home/sr2173/git/molecular-repa"
PROTEINA_DIR="$REPO_DIR/src/proteina/proteinfoundation"

#! Activate venv
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

#! Set data path (adjust to your setup)
export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"

export OMP_NUM_THREADS=1
export WANDB_INIT_TIMEOUT=120

#! Ensure log directory exists
mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

#! Run training
cd "$PROTEINA_DIR"
echo "Working directory: $(pwd)"
echo "Config: training_ca_baseline"
echo "Time: $(date)"

CMD="python train_repa.py --config_name training_ca_baseline --show_prog_bar"

echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD
