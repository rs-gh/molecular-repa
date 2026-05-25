#!/bin/bash
set -e
MMSEQS=/usr/local/software/mmseqs2/mmseqs/bin/mmseqs
TH=4
PAIRS=(
  "pdb_train_le128.fasta afdb_train_le128.fasta pdb_vs_afdb_le128"
  "afdb_train_le128.fasta pdb_train_le128.fasta afdb_vs_pdb_le128"
  "pdb_train_le256.fasta afdb_train_le256.fasta pdb_vs_afdb_le256"
  "afdb_train_le256.fasta pdb_train_le256.fasta afdb_vs_pdb_le256"
)
for pair in "${PAIRS[@]}"; do
  read q t name <<<"$pair"
  echo "===> $name : Q=$q T=$t" >&2
  rm -rf tmp_$name
  mkdir -p tmp_$name
  /usr/bin/time -v $MMSEQS easy-search "$q" "$t" "${name}.m8" tmp_$name \
    --min-seq-id 0.3 -c 0.8 --cov-mode 0 --threads $TH \
    --format-output 'query,target,pident,evalue' >${name}.log 2>&1
  rm -rf tmp_$name
  echo "===> $name DONE rows=$(wc -l <${name}.m8)" >&2
done
echo ALL_DONE
