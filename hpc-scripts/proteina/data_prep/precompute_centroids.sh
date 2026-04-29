#!/bin/bash
#! Precompute training-set centroids for novelty evaluation.
#! CPU-only job (TM-score via biotite, no GPU needed).
#!
#! Usage:
#!   # Smoke test (PDB, ~10-15 min on intr):
#!   sbatch --qos=intr --time=00:30:00 \
#!       hpc-scripts/proteina/data_prep/precompute_centroids.sh \
#!       --dataset pdb --max_entries 5000 --max_per_length_group 50 \
#!       --output_path /tmp/centroids_pdb_smoke.pt
#!
#!   # Real PDB run (~1-2h):
#!   sbatch hpc-scripts/proteina/data_prep/precompute_centroids.sh \
#!       --dataset pdb \
#!       --output_path /rds/user/sr2173/hpc-work/proteina/data/centroids_pdb.pt
#!
#!   # Real AFDB SwissProt run (~1-2h):
#!   sbatch hpc-scripts/proteina/data_prep/precompute_centroids.sh \
#!       --dataset afdb \
#!       --output_path /rds/user/sr2173/hpc-work/proteina/data/centroids_afdb_swissprot.pt

#SBATCH -A computerlab-sl2-cpu
#SBATCH --job-name=centroids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/centroids-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/centroids-%j.err
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

# Parse --dataset flag (pdb | afdb), strip from $@, derive --lmdb_path
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

python -u hpc-scripts/proteina/data_prep/precompute_centroids.py \
    --lmdb_path "$LMDB" \
    "${PASSTHROUGH[@]}"

echo "=== DONE, $(date) ==="
