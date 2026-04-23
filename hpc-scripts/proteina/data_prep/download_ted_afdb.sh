#!/bin/bash
#! Download TED AFDB CATH domain summary from Zenodo (19.9 GB compressed).
#! Decompresses in-place to TSV after download.
#!
#! Source: https://zenodo.org/records/13908086
#! File:   ted_365m.domain_summary.cath.globularity.taxid.tsv.gz
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/data_prep/download_ted_afdb.sh
#!   # or on login node (long download — use screen/tmux):
#!   bash hpc-scripts/proteina/data_prep/download_ted_afdb.sh
#!
#! After completion, run:
#!   sbatch hpc-scripts/proteina/data_prep/bake_cath_labels.sh afdb

#SBATCH -J prot-ted-download
#SBATCH -A computerlab-sl3-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/ted-download-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/ted-download-%j.err
#SBATCH -p icelake

set -euo pipefail

DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
OUT_DIR="$DATA_PATH/afdb_swissprot"
GZ_FILE="$OUT_DIR/ted_365m.domain_summary.cath.globularity.taxid.tsv.gz"
TSV_FILE="$OUT_DIR/ted_365m.domain_summary.cath.globularity.taxid.tsv"
URL="https://zenodo.org/records/13908086/files/ted_365m.domain_summary.cath.globularity.taxid.tsv.gz"

mkdir -p "$OUT_DIR"
mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="
echo "=== Downloading TED AFDB CATH summary ==="
echo "    Destination: $GZ_FILE"
echo "    Size: ~19.9 GB compressed"
echo ""

if [ -f "$TSV_FILE" ]; then
    echo "TSV already exists at $TSV_FILE — skipping download and decompression."
    echo "Delete it to re-download."
    exit 0
fi

if [ ! -f "$GZ_FILE" ]; then
    echo "Downloading..."
    # --continue resumes partial downloads; --progress=bar:force shows progress in logs
    wget --continue --progress=dot:giga \
        -O "$GZ_FILE" \
        "$URL"
    echo "Download complete: $(du -sh "$GZ_FILE" | cut -f1)"
else
    echo "GZ file already present ($GZ_FILE) — skipping download, proceeding to decompress."
fi

echo ""
echo "Decompressing (this will take several minutes and ~60 GB disk)..."
gunzip -k "$GZ_FILE"   # -k keeps the .gz file
echo "Decompressed: $(du -sh "$TSV_FILE" | cut -f1)"

echo ""
echo "=== TED DOWNLOAD + DECOMPRESS COMPLETE ==="
echo "=== Time: $(date) ==="
echo ""
echo "Next step:"
echo "  sbatch hpc-scripts/proteina/data_prep/bake_cath_labels.sh afdb"
