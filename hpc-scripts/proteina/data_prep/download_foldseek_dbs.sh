#!/bin/bash
#! Download Foldseek prebuilt target databases for novelty evaluation.
#!
#! Builds two DBs under $DEST_ROOT:
#!   pdb/        Foldseek's pre-clustered PDB database (~5-15 GB, ~720k chains)
#!   afdb_swissprot/   AlphaFold Swiss-Prot subset (~30-50 GB, ~540k chains)
#!
#! Both are auto-downloaded and indexed by `foldseek databases`. Takes ~1-2 h
#! per DB (download-bound). Run independently as two SLURM jobs.
#!
#! Usage:
#!   sbatch --export=DB=pdb hpc-scripts/proteina/data_prep/download_foldseek_dbs.sh
#!   sbatch --export=DB=afdb_swissprot hpc-scripts/proteina/data_prep/download_foldseek_dbs.sh
#!
#!   # Both at once (two array tasks):
#!   sbatch --array=0-1 hpc-scripts/proteina/data_prep/download_foldseek_dbs.sh
#!
#! Re-running is safe — `foldseek databases` skips work when the DB already exists.
#! NOTE: requires outbound HTTPS from compute nodes. If the icelake nodes block
#! external traffic, run this directly on the login node in a tmux session.

#SBATCH -A computerlab-sl2-cpu
#SBATCH --job-name=fs-dbs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/fs-dbs-%A_%a.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/fs-dbs-%A_%a.err
#SBATCH -p icelake

set -e

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

export PATH=/rds/user/sr2173/hpc-work/tools/foldseek/bin:$PATH

DEST_ROOT="/rds/user/sr2173/hpc-work/proteina/foldseek_dbs"
TMP_ROOT="/rds/user/sr2173/hpc-work/proteina/tmp/foldseek_dbs"
mkdir -p "$DEST_ROOT" "$TMP_ROOT" /rds/user/sr2173/hpc-work/proteina/logs

# Resolve which DB to fetch. Priority: --export=DB=... > SLURM_ARRAY_TASK_ID > error.
if [ -n "${DB:-}" ]; then
    SELECTED="$DB"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    case "$SLURM_ARRAY_TASK_ID" in
        0) SELECTED="pdb" ;;
        1) SELECTED="afdb_swissprot" ;;
        *) echo "ERROR: array task $SLURM_ARRAY_TASK_ID out of range (0-1)" >&2; exit 1 ;;
    esac
else
    echo "ERROR: set DB=pdb or DB=afdb_swissprot via --export, or use --array=0-1" >&2
    exit 1
fi

case "$SELECTED" in
    pdb)
        FS_NAME="PDB"
        OUT_DIR="$DEST_ROOT/pdb"
        ;;
    afdb_swissprot)
        FS_NAME="Alphafold/Swiss-Prot"
        OUT_DIR="$DEST_ROOT/afdb_swissprot"
        ;;
    *)
        echo "ERROR: DB must be 'pdb' or 'afdb_swissprot' (got: '$SELECTED')" >&2
        exit 1
        ;;
esac

mkdir -p "$OUT_DIR"
TMP_DIR="$(mktemp -d -p "$TMP_ROOT" "fs_${SELECTED}_XXXXXX")"

echo "=== NODE:    $(hostname)"
echo "=== Time:    $(date)"
echo "=== Job:     $SLURM_JOB_ID  Array: ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "=== DB:      $SELECTED  (Foldseek: $FS_NAME)"
echo "=== Out:     $OUT_DIR"
echo "=== Tmp:     $TMP_DIR"
echo "=== Threads: 8"
echo ""

# `foldseek databases <name> <outDB> <tmpDir>` is idempotent — re-runs are no-ops
# if the DB index files are present. We point at $OUT_DIR/db so the DB lives at a
# stable path that the novelty pipeline can reference.
foldseek databases \
    "$FS_NAME" \
    "$OUT_DIR/db" \
    "$TMP_DIR" \
    --threads 8 \
    -v 3

# Foldseek leaves a tmp/latest symlink; clean it up to free disk + inodes.
rm -rf "$TMP_DIR"

echo ""
echo "=== DONE for $SELECTED at $(date)"
echo "=== DB files under $OUT_DIR:"
ls -lh "$OUT_DIR" | head -40
