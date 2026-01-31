#!/bin/bash
#!
#! SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#! Run qm9_baseline experiment
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J qm9-baseline
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#! To find your account, run: sacctmgr show associations user=$USER format=Account%30
#SBATCH -A LIO-CHARM-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=09:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=ALL
#! Uncomment and set your email to receive notifications:
#SBATCH --mail-user=sr2173@cam.ac.uk

#! Output and error logs:
#SBATCH --output=/rds/user/sr2173/hpc-work/git/molecular-repa/logs/slurm-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/git/molecular-repa/logs/slurm-%j.err

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
module load rhel8/default-amp              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
module load python/3.11.0-icl

#! Work directory - where the repo is cloned on HPC
REPO_DIR="/rds/user/sr2173/hpc-work/git/molecular-repa"

#! Deactivate conda if active and activate the project venv
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

#! Run options for the application:
experiment="qm9/baseline"

#! Work directory (i.e. where the job will run):
workdir="$REPO_DIR"

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 128:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! Checkpoint auto-resume: find latest checkpoint for this experiment
CKPT=$(find "$REPO_DIR/outputs" -path "*${experiment}*" -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)

if [ -n "$CKPT" ]; then
    echo "Resuming from checkpoint: $CKPT"
    CMD="python scripts/train_tabasco.py experiment=$experiment ckpt_path=$CKPT datamodule.num_workers=0"
else
    echo "Starting fresh training run"
    CMD="python scripts/train_tabasco.py experiment=$experiment datamodule.num_workers=0"
fi

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD
