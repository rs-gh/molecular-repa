#!/bin/bash
#! CPU job: whole-set vs designable structural diversity (Experiment A).
#! Pure biotite/numpy TM-align — CPU-bound, no GPU needed.
#SBATCH --job-name=wholeset-div
#SBATCH --partition=icelake
#SBATCH --account=computerlab-sl2-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/wholeset_div_%j.out

set -euo pipefail
cd /home/sr2173/git/molecular-repa
source .venv/bin/activate
export PROJECT_ROOT=$(pwd)/src/proteina
export SKIP_SMOKE_GATE=1

CAP="${1:-150}"
echo "Running whole-set vs designable diversity, cap=$CAP"
python evaluation/proteina/generation/scripts/paper/exp_wholeset_vs_designable_diversity.py --cap "$CAP"
