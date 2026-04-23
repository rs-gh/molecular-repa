#!/bin/bash
#! Pre-filter the 120 GB TED TSV to only rows matching our Swiss-Prot AFDB accessions.
#! Reduces the file from ~120 GB to ~120 MB, making subsequent CATH lookups fast.
#!
#! Match key: TED row ID starts with AF-XXXXXX (e.g. AF-Q9RUA0-F1-model_v4_TED01).
#! Our splits.json contains AF-XXXXXX-F1-model_v6.pdb entries.
#! We match on the AF-XXXXXX accession prefix, version-agnostic.
#!
#! Output: ted_swissprot_filtered.tsv alongside the full TSV.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/data_prep/filter_ted_swissprot.sh
#!
#! After completion, resubmit:
#!   sbatch hpc-scripts/proteina/data_prep/analyse_cath_distribution.sh
#!   sbatch hpc-scripts/proteina/data_prep/bake_cath_labels.sh afdb

#SBATCH -J prot-ted-filter
#SBATCH -A computerlab-sl2-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/ted-filter-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/ted-filter-%j.err
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

IN_TSV="$DATA_PATH/afdb_swissprot/ted_365m.domain_summary.cath.globularity.taxid.tsv"
OUT_TSV="$DATA_PATH/afdb_swissprot/ted_swissprot_filtered.tsv"
SPLITS_JSON="$DATA_PATH/afdb_swissprot/splits.json"

echo "Input : $IN_TSV ($(du -sh "$IN_TSV" | cut -f1))"
echo "Output: $OUT_TSV"
echo ""

python3 - <<'EOF'
import json, sys, time
from pathlib import Path

data_path = Path("/rds/user/sr2173/hpc-work/proteina/data/afdb_swissprot")
in_tsv    = data_path / "ted_365m.domain_summary.cath.globularity.taxid.tsv"
out_tsv   = data_path / "ted_swissprot_filtered.tsv"
splits    = data_path / "splits.json"

# Build accession set from splits.json: AF-Q9RUA0-F1-model_v6.pdb -> AF-Q9RUA0
with open(splits) as f:
    d = json.load(f)
all_ids = d["train"] + d["val"] + d["test"]
accessions = set()
for x in all_ids:
    parts = x.replace(".pdb", "").split("-")
    # parts = ['AF', 'Q9RUA0', 'F1', 'model_v6'] -> take first two
    accessions.add(f"AF-{parts[1]}")
print(f"Target accessions: {len(accessions):,}", flush=True)

t0 = time.time()
kept = skipped = 0
with open(in_tsv) as fin, open(out_tsv, "w") as fout:
    for i, line in enumerate(fin):
        # Row ID is first field, e.g. AF-Q9RUA0-F1-model_v4_TED01
        # Extract AF-XXXXXX: split on '-', take first two parts
        row_id = line[:line.index('\t')]
        parts  = row_id.split("-")
        acc    = f"AF-{parts[1]}" if len(parts) >= 2 else ""
        if acc in accessions:
            fout.write(line)
            kept += 1
        else:
            skipped += 1
        if (i + 1) % 10_000_000 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {i+1:,} rows | kept {kept:,} | {elapsed:.0f}s", flush=True)

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s", flush=True)
print(f"  Kept   : {kept:,} rows", flush=True)
print(f"  Skipped: {skipped:,} rows", flush=True)
EOF

echo ""
echo "Output size: $(du -sh "$OUT_TSV" | cut -f1)"
echo ""
echo "=== TED FILTER COMPLETE ==="
echo "=== Time: $(date) ==="
echo ""
echo "Next steps:"
echo "  sbatch hpc-scripts/proteina/data_prep/analyse_cath_distribution.sh"
echo "  sbatch hpc-scripts/proteina/data_prep/bake_cath_labels.sh afdb"
