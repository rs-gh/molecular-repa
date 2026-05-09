#!/bin/bash
#! Precompute SS reference distribution for proteina generation eval.
#! CPU-only, fast (~few min for 5000 structures).
#!
#! Usage:
#!   # PDB reference:
#!   sbatch --qos=intr --time=00:30:00 \
#!       hpc-scripts/proteina/data_prep/precompute_ss_reference.sh \
#!       --dataset pdb \
#!       --output_path /rds/user/sr2173/hpc-work/proteina/data/ss_reference_pdb.pt
#!
#!   # AFDB reference:
#!   sbatch --qos=intr --time=00:30:00 \
#!       hpc-scripts/proteina/data_prep/precompute_ss_reference.sh \
#!       --dataset afdb \
#!       --output_path /rds/user/sr2173/hpc-work/proteina/data/ss_reference_afdb.pt
#!
#! N=5000 chosen because per-bin counts (H/E/C) on a balanced 3-bin distribution
#! have binomial SE ≈ sqrt(p*(1-p)/N) ≈ 0.005 — well below per-run between-model
#! variation. Smoke-test first with --max_samples 200 to confirm pipeline works.

#SBATCH -A computerlab-sl2-cpu
#SBATCH --job-name=ss_ref
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/ss_ref-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/ss_ref-%j.err
#SBATCH -p icelake

set -e

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export PYTHONPATH="$REPO_DIR/src/proteina:$PYTHONPATH"
export OMP_NUM_THREADS=1

cd "$REPO_DIR"

DATASET=""
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            PASSTHROUGH+=("$1")
            shift
            ;;
    esac
done

case "$DATASET" in
    pdb)
        LMDB="/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb/train.lmdb"
        ;;
    afdb)
        LMDB="/rds/user/sr2173/hpc-work/proteina/data/afdb_swissprot/lmdb/train.lmdb"
        ;;
    *)
        echo "ERROR: --dataset must be 'pdb' or 'afdb' (got: '$DATASET')" >&2
        exit 1
        ;;
esac

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="
echo "=== Dataset: $DATASET, LMDB: $LMDB ==="
echo "=== Passthrough args: ${PASSTHROUGH[*]} ==="

python -u evaluation/proteina/generation/scripts/precompute_ss_reference.py \
    --lmdb_path "$LMDB" \
    "${PASSTHROUGH[@]}"

echo "=== DONE, $(date) ==="
