#!/bin/bash
#!
#! Serial backfill of _res_novelty_foldseek_* into all paper-table jsonl rows.
#!
#! Iterates every row in evaluation/proteina/generation/results/paper/n*_paper_*/
#! sweep_results.jsonl, resolves its eval_output dir, and:
#!   - runs compute_novelty_foldseek for rows whose samples_fid + designability
#!     index are still on disk,
#!   - syncs already-filled CSV columns into the jsonl for rows we pre-filled
#!     elsewhere (e.g. the timing-test run),
#!   - copies foldseek columns from a sibling row when a pruned row shares
#!     (run, step) with a sibling that has the columns.
#!
#! Submission mode: pass --array=0-1 and the script will run two shards
#!   sbatch --array=0-1 backfill_novelty_foldseek.sh
#! Each shard handles ~17 of 33 usable rows on --qos=intr. After both shards
#! finish, run the finalize step on a login node (cached sync + pruned
#! sibling-copy) — that's cheap and doesn't need SLURM:
#!   python evaluation/proteina/generation/scripts/backfill_novelty_foldseek.py --finalize
#!
#! Per-shard wall estimate (16 threads, intr/icelake):
#!   - median 82 designable queries -> ~110 s/ckpt
#!   - 17 ckpts -> ~30 min wall, well under the 1h intr cap.

#SBATCH -J nov-foldseek-backfill
#SBATCH -A computerlab-sl2-cpu
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/nov-foldseek-backfill-%A_%a.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/nov-foldseek-backfill-%A_%a.err
#SBATCH -p icelake

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH=/rds/user/sr2173/hpc-work/tools/foldseek/bin:$PATH
mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-2}"

echo "=== Foldseek novelty backfill (shard $SHARD / $NSHARDS) ==="
echo "node:    $(hostname), threads=$SLURM_CPUS_PER_TASK, time=$(date)"
foldseek version
echo ""

cd "$REPO_DIR"

START=$(date +%s)
# Pass-through extra args (e.g. --force-rerun --profile-prefix n256_convergence)
# All sbatch args after the launcher are forwarded to the Python script.
python -u evaluation/proteina/generation/scripts/backfill_novelty_foldseek.py \
    --threads 16 \
    --shard "$SHARD" \
    --num-shards "$NSHARDS" \
    "$@"
END=$(date +%s)
echo ""
echo "=== Shard $SHARD wall: $((END - START))s, $(date) ==="
