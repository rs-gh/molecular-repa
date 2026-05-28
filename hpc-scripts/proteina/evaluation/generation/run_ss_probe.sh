#!/bin/bash
#! GPU job: MPNN cross-dataset SS-probe (Experiment B).
#SBATCH --job-name=ss-probe
#SBATCH --partition=ampere
#SBATCH --account=lio-charm-sl2-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/ss_probe_%j.out

set -euo pipefail
cd /home/sr2173/git/molecular-repa
source .venv/bin/activate
export PROJECT_ROOT=$(pwd)/src/proteina
export SKIP_SMOKE_GATE=1

NPROT="${1:-600}"
echo "MPNN cross-dataset SS-probe, n_proteins=$NPROT per dataset"
python encoder_profiling/proteina/mpnn/ss_probe_cross_dataset.py --n-proteins "$NPROT" --max-res 60000
