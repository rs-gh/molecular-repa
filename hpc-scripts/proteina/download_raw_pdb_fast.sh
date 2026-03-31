#!/bin/bash
#! Fast PDB download using parallel wget (replaces Python urllib approach).
#! ~780 files/min vs ~170 files/min with Python multiprocessing.
#!
#! Requires the full dataset CSV to already exist (created by first run of
#! download_raw_pdb.sh or by the dataselector). Reads PDB codes from CSV,
#! skips existing files, downloads remaining with parallel wget.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/download_raw_pdb_fast.sh

#SBATCH -J prot-pdb-dl-fast
#SBATCH -A COMPUTERLAB-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/pdb-dl-fast-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/pdb-dl-fast-%j.err
#SBATCH -p icelake

module load rhel8/default

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
RAW_DIR="$DATA_PATH/pdb_train/raw"
mkdir -p "$RAW_DIR"

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="
echo ""

# Step 1: Find the full dataset CSV (fraction=1.0, max_length=512)
CSV=$(ls "$DATA_PATH/pdb_train/"df_pdb_f1.0_*maxl512*.csv 2>/dev/null | head -1)
if [ -z "$CSV" ]; then
    echo "ERROR: No full dataset CSV found. Run download_raw_pdb.sh first to create it."
    exit 1
fi
echo "=== Using CSV: $(basename $CSV) ==="

# Step 2: Extract unique PDB codes from CSV
CODES_FILE=$(mktemp)
tail -n +2 "$CSV" | cut -d',' -f2 | sort -u > "$CODES_FILE"
N_TOTAL=$(wc -l < "$CODES_FILE")
echo "=== Unique PDB codes in CSV: $N_TOTAL ==="

# Step 3: Build list of codes to download (skip existing)
EXISTING_FILE=$(mktemp)
ls "$RAW_DIR" 2>/dev/null | sed 's/\.cif.*//g' | sort -u > "$EXISTING_FILE"
N_EXISTING=$(wc -l < "$EXISTING_FILE")
echo "=== Already downloaded: $N_EXISTING ==="

DOWNLOAD_LIST=$(mktemp)
comm -23 "$CODES_FILE" "$EXISTING_FILE" > "$DOWNLOAD_LIST"
N_TO_DOWNLOAD=$(wc -l < "$DOWNLOAD_LIST")
echo "=== To download: $N_TO_DOWNLOAD ==="
echo "=== Start: $(date) ==="
echo ""

rm -f "$CODES_FILE" "$EXISTING_FILE"

if [ "$N_TO_DOWNLOAD" -eq 0 ]; then
    echo "Nothing to download!"
    rm -f "$DOWNLOAD_LIST"
    exit 0
fi

# Step 4: Download with parallel wget (32 concurrent)
download_one() {
    local code="$1"
    local outfile="$RAW_DIR/${code}.cif.gz"
    wget -q --retry-connrefused --waitretry=2 --tries=3 \
        -O "$outfile" \
        "https://files.rcsb.org/download/${code}.cif.gz" 2>/dev/null
    if [ $? -ne 0 ]; then
        rm -f "$outfile"
    fi
}
export -f download_one
export RAW_DIR

cat "$DOWNLOAD_LIST" | xargs -P 32 -I {} bash -c 'download_one "$@"' _ {}

rm -f "$DOWNLOAD_LIST"

N_FINAL=$(ls "$RAW_DIR" 2>/dev/null | wc -l)
echo ""
echo "=== DOWNLOAD COMPLETE ==="
echo "=== Total raw files: $N_FINAL / $N_TOTAL unique PDB codes ==="
echo "=== Time: $(date) ==="
echo ""
echo "Next step: sbatch hpc-scripts/proteina/convert_to_lmdb.sh pdb"
