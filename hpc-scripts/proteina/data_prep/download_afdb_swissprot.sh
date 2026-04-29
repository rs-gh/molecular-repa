#!/bin/bash
#! Download AFDB Swiss-Prot PDB tar (v6, ~27 GB) from EBI FTP.
#! The LMDB builder streams directly from this tar - no extraction needed.
#! Resumable: wget --continue picks up an interrupted download.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/data_prep/download_afdb_swissprot.sh
#!   # or interactively on a login node:
#!   bash hpc-scripts/proteina/data_prep/download_afdb_swissprot.sh
#!
#! After completion, run:
#!   sbatch hpc-scripts/proteina/data_prep/build_afdb_lmdb.sh

#SBATCH -J prot-afdb-dl
#SBATCH -A computerlab-sl2-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/afdb-dl-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/afdb-dl-%j.err
#SBATCH -p icelake

set -euo pipefail

DEST_DIR="${DATA_PATH:-/rds/user/sr2173/hpc-work/proteina/data}/afdb_swissprot"
TAR_PATH="$DEST_DIR/swissprot_pdb_v6.tar"
URL="https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v6.tar"

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="
echo "=== Dest: $TAR_PATH ==="
echo ""

mkdir -p "$DEST_DIR"
mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

###############################################################
### Download (resumable)                                    ###
###############################################################

echo "=== Downloading swissprot_pdb_v6.tar (~27 GB) ==="
wget \
    --continue \
    --show-progress \
    --progress=dot:giga \
    --tries=5 \
    --waitretry=30 \
    -O "$TAR_PATH" \
    "$URL"

echo "=== Downloaded: $(du -h "$TAR_PATH" | cut -f1) ==="
echo ""

###############################################################
### Integrity check                                         ###
###############################################################

echo "=== Verifying tar integrity ==="
# List the first and last few members - catches truncated or corrupt archives
# without reading all 27 GB (tar --list streams headers only)
MEMBER_COUNT=$(tar -t -f "$TAR_PATH" | wc -l)
echo "=== Members in tar: $MEMBER_COUNT ==="

if [ "$MEMBER_COUNT" -lt 100000 ]; then
    echo "ERROR: Expected ~550k members, got $MEMBER_COUNT - tar may be corrupt or incomplete"
    exit 1
fi

echo "=== Tar looks healthy ==="
echo ""

###############################################################
### Done                                                    ###
###############################################################

echo "=== DOWNLOAD COMPLETE ==="
echo "=== Time: $(date) ==="
echo ""
echo "Next: build LMDB (streams directly from tar, no extraction):"
echo "  sbatch hpc-scripts/proteina/data_prep/build_afdb_lmdb.sh"
