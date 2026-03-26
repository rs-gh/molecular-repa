#!/bin/bash
#!
#! SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#! Evaluate MACE additive and tradeoff checkpoints
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J eval-mace
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A LIO-CHARM-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#SBATCH --gres=gpu:1
#! Number of CPUs per task:
#SBATCH --cpus-per-task=16
#! How much wallclock time will be required?
#SBATCH --time=04:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk

#! Output and error logs (on RDS to avoid filling /home):
#SBATCH --output=/rds/user/sr2173/hpc-work/tabasco/logs/slurm-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/tabasco/logs/slurm-%j.err

#! Do not change:
#SBATCH -p ampere

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by GPU number*walltime.

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

#! ############################################################
#! Modify the settings below to specify the application's environment, location
#! and launch method:

#! Optionally modify the environment seen by the application
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/ampere/base              # Ampere-native env (enables torch.compile)

#! Insert additional module load commands after this line if needed:
module load python/3.11.0-icl

#! Where the repo is cloned
REPO_DIR="/home/sr2173/git/molecular-repa"

set -euo pipefail

cd "$REPO_DIR"
source .venv/bin/activate
export PROJECT_ROOT=$(pwd)/src/tabasco

echo "=== Evaluating MACE additive ==="
python evaluation/scripts/evaluate.py \
    --checkpoint evaluation/checkpoints/tabasco/geom/mace_additive.ckpt \
    --output_dir evaluation/results/tabasco/geom/evaluation/mace_additive/ \
    --train_smiles evaluation/data/tabasco/geom/geom_train_smiles.txt \
    --num_mols 1000 --num_steps 100

echo ""
echo "=== Evaluating MACE tradeoff ==="
python evaluation/scripts/evaluate.py \
    --checkpoint evaluation/checkpoints/tabasco/geom/mace_tradeoff.ckpt \
    --output_dir evaluation/results/tabasco/geom/evaluation/mace_tradeoff/ \
    --train_smiles evaluation/data/tabasco/geom/geom_train_smiles.txt \
    --num_mols 1000 --num_steps 100

echo ""
echo "=== Compiling results ==="
python evaluation/scripts/compile_results.py evaluation/results/tabasco/geom/evaluation

echo "Done!"
