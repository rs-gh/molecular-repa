#!/bin/bash
#!
#! SLURM job script for unified REPA projector analysis
#! Runs encoder_analysis.py across CheMeleon, MACE, GearNet
#!

#SBATCH -J proj-analysis
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/tabasco/logs/proj-analysis-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/tabasco/logs/proj-analysis-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
module load python/3.11.0-icl

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

cd "$REPO_DIR"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PROJECT_ROOT="$REPO_DIR/src/tabasco"
export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "JobID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Directory: $(pwd)"
echo ""

echo "=== Phase 1: Computing embeddings (GPU) ==="
python -u playground/projector/encoder_analysis.py --phase embeddings --n-mols 200

echo ""
echo "=== Phase 2: Analysis (CPU) ==="
python -u playground/projector/encoder_analysis.py --phase analysis

echo ""
echo "=== Done ==="
echo "Time: $(date)"
echo "Results in playground/projector/results.json"
echo "Figures in playground/projector/figures/"
