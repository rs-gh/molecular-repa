#!/bin/bash
#! Download InterPro protein2ipr.dat.gz and filter to Gene3D (CATH) entries
#! for our Swiss-Prot AFDB accessions only.
#!
#! Source: https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/protein2ipr.dat.gz
#! Size:   ~17 GB compressed
#! Format: UniProt_acc | InterPro_ID | description | db:entry | start | end
#! Gene3D entries have db:entry starting with "G3DSA:" followed by CATH code.
#!
#! Output: $DATA_PATH/afdb_swissprot/interpro_gene3d_swissprot.tsv
#!   Columns: uniprot_acc, cath_code, start, end
#!   Only rows matching our 550k Swiss-Prot accessions, Gene3D only.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/data_prep/download_interpro_gene3d.sh
#!
#! After completion, run:
#!   sbatch hpc-scripts/proteina/data_prep/analyse_cath_distribution.sh   (AFDB)
#!   sbatch hpc-scripts/proteina/data_prep/bake_cath_labels.sh afdb

#SBATCH -J prot-interpro-dl
#SBATCH -A computerlab-sl2-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/interpro-dl-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/interpro-dl-%j.err
#SBATCH -p icelake

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="

GZ_FILE="$DATA_PATH/afdb_swissprot/protein2ipr.dat.gz"
OUT_TSV="$DATA_PATH/afdb_swissprot/interpro_gene3d_swissprot.tsv"
SPLITS_JSON="$DATA_PATH/afdb_swissprot/splits.json"
URL="https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/protein2ipr.dat.gz"

echo "Step 1: Download protein2ipr.dat.gz (~17 GB)"
if [ ! -f "$GZ_FILE" ]; then
    wget --continue --progress=dot:giga -O "$GZ_FILE" "$URL"
    echo "Download complete: $(du -sh "$GZ_FILE" | cut -f1)"
else
    echo "Already present: $GZ_FILE ($(du -sh "$GZ_FILE" | cut -f1)) - skipping download"
fi

echo ""
echo "Step 2: Filter to Gene3D entries for our Swiss-Prot accessions"

python3 - <<'EOF'
import json, gzip, time
from pathlib import Path

data_path = Path("/rds/user/sr2173/hpc-work/proteina/data/afdb_swissprot")
gz_file  = data_path / "protein2ipr.dat.gz"
out_tsv  = data_path / "interpro_gene3d_swissprot.tsv"
splits   = data_path / "splits.json"

# Build target accession set: AF-Q9RUA0-F1-model_v6.pdb -> Q9RUA0
with open(splits) as f:
    d = json.load(f)
our_accs = set(x.replace(".pdb", "").split("-")[1]
               for x in d["train"] + d["val"] + d["test"])
print(f"Target accessions: {len(our_accs):,}", flush=True)

t0 = time.time()
kept = total = 0
with gzip.open(gz_file, "rt") as fin, open(out_tsv, "w") as fout:
    fout.write("uniprot_acc\tcath_code\tstart\tend\n")
    for line in fin:
        total += 1
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6:
            continue
        acc, _, _, db_entry, start, end = parts[:6]
        if acc in our_accs and db_entry.startswith("G3DSA:"):
            cath_code = db_entry[6:]  # strip "G3DSA:"
            fout.write(f"{acc}\t{cath_code}\t{start}\t{end}\n")
            kept += 1
        if total % 10_000_000 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {total:,} lines | kept {kept:,} | {elapsed:.0f}s", flush=True)

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s", flush=True)
print(f"  Total lines : {total:,}", flush=True)
print(f"  Kept        : {kept:,} Gene3D rows", flush=True)

# Report unique accession coverage
with open(out_tsv) as f:
    next(f)  # skip header
    covered = set(line.split("\t")[0] for line in f)
print(f"  Unique accs with Gene3D: {len(covered):,} / {len(our_accs):,} "
      f"({100*len(covered)/len(our_accs):.1f}%)", flush=True)
EOF

echo ""
echo "Output: $(du -sh "$OUT_TSV" | cut -f1) at $OUT_TSV"
echo ""
echo "=== INTERPRO DOWNLOAD + FILTER COMPLETE ==="
echo "=== Time: $(date) ==="
echo ""
echo "Next steps:"
echo "  Update analyse_cath_distribution.sh --interpro_tsv flag and resubmit for AFDB"
echo "  sbatch hpc-scripts/proteina/data_prep/bake_cath_labels.sh afdb"
