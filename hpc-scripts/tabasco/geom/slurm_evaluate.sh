#!/bin/bash
#!
#! SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#! Post-hoc evaluation: generate 1000 molecules per checkpoint, compute all metrics
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J eval-geom
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A LIO-CHARM-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#SBATCH --gres=gpu:1
#! Number of CPUs per task:
#SBATCH --cpus-per-task=4
#! How much wallclock time will be required?
#SBATCH --time=02:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk

#! Output and error logs:
#SBATCH --output=/rds/user/sr2173/hpc-work/tabasco/logs/eval-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/tabasco/logs/eval-%j.err

#! Do not change:
#SBATCH -p ampere

#! ############################################################

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

echo "JobID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Directory: $(pwd)"
echo ""

echo "=== Evaluating all GEOM checkpoints ==="
echo "Checkpoints available:"
ls -lh evaluation/checkpoints/tabasco/geom/*.ckpt
echo ""

python evaluation/scripts/evaluate.py \
    --all \
    --num_mols 1000 \
    --num_steps 100 \
    --batch_size 256 \
    --no_bootstrap \
    --device cuda

echo ""
echo "=== Done ==="
echo "Time: $(date)"
echo "Results in evaluation/results/tabasco/geom/"
