#!/bin/bash
#!
#! Resume-safety validator. Confirms a checkpoint can be loaded into a
#! model with a candidate optimization applied, and that a single
#! forward+backward+optimizer step doesn't produce NaN/Inf or state_dict
#! mismatches.
#!
#! Usage:
#!   sbatch validate_resume.sh <ckpt_path> <config_name> [config_subdir] [optimization]
#!
#! Examples:
#!   # Baseline 256, plain noop resume sanity:
#!   sbatch validate_resume.sh \
#!     /rds/user/sr2173/hpc-work/proteina/store/proteina_60m_baseline_256/checkpoints/last.ckpt \
#!     training_baseline_256 training/256 noop
#!
#!   # REPA l4 128, test EFFICIENT_ATTENTION backend is safe:
#!   sbatch validate_resume.sh \
#!     /rds/user/sr2173/hpc-work/proteina/store/proteina_60m_repa_l4_128_per_residue/checkpoints/last.ckpt \
#!     training_repa_l4_128_per_residue training/128/per_residue sdpa_efficient

#SBATCH -J validate-resume
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/validate-resume-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/validate-resume-%j.err
#SBATCH -p ampere

CKPT="${1:?Usage: $0 <ckpt> <config> [subdir] [optimization]}"
CONFIG="${2:?Usage: $0 <ckpt> <config> [subdir] [optimization]}"
SUBDIR="${3:-}"
OPT="${4:-noop}"

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

cd "$REPO_DIR/src/proteina/proteinfoundation"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export PYTHONUNBUFFERED=1

echo "=== RESUME VALIDATOR ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo "=== Checkpoint: $CKPT"
echo "=== Config: $CONFIG${SUBDIR:+ (subdir=$SUBDIR)}"
echo "=== Optimization: $OPT"
echo ""

SUBDIR_ARG=""
if [ -n "$SUBDIR" ]; then
    SUBDIR_ARG="--config_subdir $SUBDIR"
fi

python -u "$REPO_DIR/hpc-scripts/proteina/smoke_tests/validate_resume.py" \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    $SUBDIR_ARG \
    --optimization "$OPT"

EXIT_CODE=$?
echo ""
echo "=== DONE: $(date), exit=$EXIT_CODE ==="
exit $EXIT_CODE
