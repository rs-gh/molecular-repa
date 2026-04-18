#!/bin/bash
#!
#! P2 — verify torch.compile handles 4 distinct (B, N) shapes cleanly.
#!
#! Round-robin through the 4 planned bucket shapes for 3 passes (12 total
#! steps). First pass is expected to hit compile cost (30-60 s per shape);
#! subsequent passes should be fast (cache hit). Reports compile event
#! counts from torch._dynamo.
#!
#! Isolated: reuses bench tooling, no edits to training code paths.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/bench/p2_compile_multishape.sh

#SBATCH -J p2-compile-multi
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:40:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/p2-compile-multi-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/p2-compile-multi-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

cd "$REPO_DIR/src/proteina/proteinfoundation"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export PYTHONUNBUFFERED=1
# Surface dynamo recompile reasons in case the verdict is WARN
export TORCH_LOGS="recompiles"

echo "=== P2: torch.compile multi-shape compatibility ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo ""

python -u "$REPO_DIR/hpc-scripts/proteina/bench/p2_compile_multishape.py"
EXIT=$?

echo ""
echo "=== DONE: $(date), exit=$EXIT ==="
exit $EXIT
