#!/bin/bash
#! End-to-end designability wall on a 20-PDB subset (5/length x 4 lengths)
#! with the TF32 patch in. Compares to historical 0.155*L estimate.

#SBATCH -J desig-e2e
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=intr
#SBATCH --time=00:45:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/desig-e2e-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/desig-e2e-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

CA_WEIGHTS_ROOT="/rds/user/sr2173/hpc-work/proteina/ProteinMPNN"
export PROTEINMPNN_DIR="$REPO_DIR/src/proteina/ProteinMPNN"
export PROTEINMPNN_WEIGHTS_DIR="$CA_WEIGHTS_ROOT"
export PYTHON_EXEC="$REPO_DIR/.venv/bin/python"

export CACHE_DIR="/rds/user/sr2173/hpc-work/proteina/hf_cache"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_HUB_DISABLE_XET=1
export TMPDIR="/rds/user/sr2173/hpc-work/proteina/tmp"
mkdir -p "$CACHE_DIR/hub" "$TMPDIR"

SAMPLES_DIR="$REPO_DIR/eval_output/inference_paper_inference_fid_60m_paper_sweep_pretrained_dfs_60m_n256_paper_step_1300000/samples_fid"

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== Time: $(date) ==="

cd "$REPO_DIR"

python -u hpc-scripts/proteina/bench/benchmark_designability_e2e.py \
    --samples_dir "$SAMPLES_DIR" \
    --lengths 100 150 200 250 \
    --per_length 5 \
    --tmp_root "$TMPDIR/desig_e2e_${SLURM_JOB_ID}"

echo ""
echo "=== DONE: $(date) ==="
