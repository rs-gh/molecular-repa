#!/bin/bash
#!
#! SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#! Run geom/mace_cached_tradeoff experiment (REPA with cached MACE encoder, tradeoff)
#! Seeded from live MACE run checkpoint on first submission.
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J geom-mace-cached-tradeoff
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#! To find your account, run: sacctmgr show associations user=$USER format=Account%30
#SBATCH -A LIO-CHARM-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#SBATCH --gres=gpu:1
#! Number of CPUs per task (workers + 1 for main process):
#SBATCH --cpus-per-task=16
#! How much wallclock time will be required?
#SBATCH --time=16:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=ALL
#! Uncomment and set your email to receive notifications:
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

#! Where the repo is cloned (code lives in /home, outputs go to /rds via configs/local/default.yaml)
REPO_DIR="/home/sr2173/git/molecular-repa"

#! Where Hydra writes outputs (set in src/tabasco/configs/local/default.yaml on the HPC machine)
OUTPUTS_DIR="/rds/user/sr2173/hpc-work/tabasco/outputs"

#! Deactivate conda if active and activate the project venv
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

#! Run options for the application:
experiment="geom/mace_cached_tradeoff"

#! Work directory (i.e. where the job will run):
workdir="$REPO_DIR"

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 128:
export OMP_NUM_THREADS=1
export WANDB_INIT_TIMEOUT=120

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! Safe fixed worker count (31 workers exhausts /dev/shm on HPC Ampere nodes)
num_workers=0

#! Copy MACE embedding LMDB to local NVMe for fast random reads.
#! The 14GB file on Lustre/NFS causes severe mmap thrashing.
MACE_LMDB_SRC="$REPO_DIR/src/tabasco/data/mace_geom/train_embeddings.lmdb"
LOCAL_MACE_LMDB="/tmp/mace_geom_train_embeddings.lmdb"
if [ ! -f "$LOCAL_MACE_LMDB" ]; then
    echo "Copying MACE embeddings to local NVMe..."
    cp "$MACE_LMDB_SRC" "$LOCAL_MACE_LMDB"
    echo "Done ($(du -h $LOCAL_MACE_LMDB | cut -f1))"
else
    echo "MACE embeddings already on local NVMe: $LOCAL_MACE_LMDB"
fi

#! Per-experiment output directory so checkpoint searches don't cross-contaminate
exp_safe=$(echo "$experiment" | tr '/' '_')_v2
EXP_OUTPUTS_DIR="$OUTPUTS_DIR/$exp_safe"
hydra_run_dir="$EXP_OUTPUTS_DIR/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"

#! Checkpoint auto-resume: find latest checkpoint for this experiment only
CKPT=$(find "$EXP_OUTPUTS_DIR" -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)

#! Seed checkpoint: on first run, resume from the live MACE additive run.
#! Once this experiment has its own checkpoints, auto-resume takes over.
SEED_CKPT="$OUTPUTS_DIR/geom_mace_tradeoff_v2/checkpoints/last.ckpt"
if [ -z "$CKPT" ] && [ -f "$SEED_CKPT" ]; then
    echo "First run: seeding from live MACE checkpoint: $SEED_CKPT"
    CKPT="$SEED_CKPT"
fi

#! WandB run resume: find most recent run that actually logged metrics (has wandb-summary.json)
WANDB_RUN_ID=""
if [ -n "$CKPT" ] && [ "$CKPT" != "$SEED_CKPT" ]; then
    WANDB_RUN_ID=$(find "$EXP_OUTPUTS_DIR/wandb" -name "wandb-summary.json" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2 | sed 's|.*/run-[0-9_]*-\([^/]*\)/.*|\1|')
fi

#! Common Hydra overrides (local LMDB path for fast reads)
COMMON_OVERRIDES="trainer=gpu model.compile=true trainer.precision=16 \
        datamodule.num_workers=$num_workers \
        model.repa_loss.encoder.lmdb_path=$LOCAL_MACE_LMDB"

if [ -n "$CKPT" ] && [ -n "$WANDB_RUN_ID" ]; then
    echo "Resuming from checkpoint: $CKPT (wandb run: $WANDB_RUN_ID)"
    CMD="python hpc-scripts/tabasco/train_tabasco.py experiment=$experiment ckpt_path=$CKPT \
        $COMMON_OVERRIDES \
        logger.wandb.id=$WANDB_RUN_ID logger.wandb.resume=must \
        hydra.run.dir=$hydra_run_dir"
elif [ -n "$CKPT" ]; then
    echo "Resuming from checkpoint: $CKPT (no wandb run found)"
    CMD="python hpc-scripts/tabasco/train_tabasco.py experiment=$experiment ckpt_path=$CKPT \
        $COMMON_OVERRIDES \
        hydra.run.dir=$hydra_run_dir"
else
    echo "Starting fresh training run"
    CMD="python hpc-scripts/tabasco/train_tabasco.py experiment=$experiment \
        $COMMON_OVERRIDES \
        hydra.run.dir=$hydra_run_dir"
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
