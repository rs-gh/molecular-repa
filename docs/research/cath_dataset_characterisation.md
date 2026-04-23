# CATH Dataset Characterisation & Conditioning Plan

## Goal

Characterise the PDB and AFDB Swiss-Prot training datasets by CATH structural family,
and use those annotations for fold-class conditioning during proteina training.

---

## Current status (2026-04-23)

### ✅ What's been accomplished

1. **Characterisation scripts written and verified**
   - `hpc-scripts/proteina/data_prep/analyse_cath_distribution.py` (read-only)
   - `hpc-scripts/proteina/data_prep/analyse_cath_pdb.sh` (slurm wrapper, PDB)
   - `hpc-scripts/proteina/data_prep/analyse_cath_afdb.sh` (slurm wrapper, AFDB)
   - Smoke-tested and debugged (two bugs found: id-transform mismatch for InterPro,
     JSON clobbering between parallel jobs — both fixed)

2. **PDB CATH annotation (SIFTS)**
   - Source: `pdb_chain_cath_uniprot.tsv.gz` (SIFTS) + `cath-b-newest-all.gz` (CATH hierarchy)
   - Both files were already on disk at `/rds/.../pdb_train/`
   - Coverage: **44.2% of train (187,920 / 425,100), 44.6% of val (2,229 / 4,999)**

3. **AFDB CATH annotation (InterPro Gene3D)**
   - Source: `protein2ipr.dat.gz` from EBI InterPro, filtered to Swiss-Prot accessions
   - Downloaded + filtered via `download_interpro_gene3d.sh`
   - Output: `/rds/.../afdb_swissprot/interpro_gene3d_swissprot.tsv` (~few MB)
   - Built lookup for 363,718 Swiss-Prot accessions
   - Coverage: **61.8% of train (141,862 / 229,670), 61.7% of val (2,791 / 4,521)**

4. **Characterisation results** → `evaluation/proteina/cath_distribution/`
   - `cath_distribution_summary.json` — all 4 splits, top-50 architectures, top-50 topologies
   - `cath_class_distribution.png` — C-level bar chart
   - `cath_architecture_distribution.png` — top-20 A-level per dataset
   - `cath_topology_distribution.png` — top-20 T-level per dataset
   - `cath_coverage.png` — coverage comparison

5. **LMDB enrichment scripts ready (not yet run)**
   - `hpc-scripts/proteina/data_prep/bake_cath_labels.py` + `.sh` wrapper
   - Supports `--interpro_tsv` for AFDB (updated to use InterPro instead of TED)
   - Backcompat-safe: only adds `cath_code` field, never modifies others
   - Idempotent: skips entries already enriched
   - Dry-run mode for verification

### Key findings from characterisation

| Dataset | Entries | CATH coverage | Dominant class | Top topology |
|---------|---------|---------------|----------------|--------------|
| PDB train | 425,100 | 44.2% | Alpha/Beta (55%) | `3.40.50` Rossmann (33,709) |
| PDB val | 4,999 | 44.6% | Alpha/Beta (54%) | `3.40.50` Rossmann (402) |
| AFDB train | 229,670 | 61.8% | (tbd from fig) | (tbd from fig) |
| AFDB val | 4,521 | 61.7% | (tbd from fig) | (tbd from fig) |

PDB val distribution matches train closely — random split preserved family balance.

### ✗ What we tried that didn't work

- **TED file** (`ted_365m.domain_summary.cath.globularity.taxid.tsv`, 120 GB uncompressed)
  - Covers AFDB v4 via Foldseek clustering
  - Uses TrEMBL-style unreviewed accessions (`A0A000`...) — near-zero overlap with
    our Swiss-Prot v6 set (reviewed `Q`/`P`/`O` accessions)
  - **Deleted** to reclaim 139 GB of disk space

- **SIFTS `uniprot_pdb.tsv.gz`** (UniProt → PDB → CATH chain)
  - Only 6.6% of our Swiss-Prot accessions have PDB structures
  - Not useful at scale

---

## What needs to happen next

### Immediate

- [ ] **Update plan doc** ← DONE (this file)

- [ ] **Inspect the characterisation figures** and confirm they look sensible
  ```bash
  ls evaluation/proteina/cath_distribution/
  # Open the .png files in an image viewer
  ```

- [ ] **Bake CATH labels into PDB LMDB** (writes in-place, backcompat-safe)
  ```bash
  # Dry-run first:
  python hpc-scripts/proteina/data_prep/bake_cath_labels.py \
      --dataset pdb --lmdb_dir /rds/.../pdb_train/lmdb \
      --cath_dir /rds/.../pdb_train --splits train --max_entries 500 --dry_run

  # Then full run:
  sbatch hpc-scripts/proteina/data_prep/bake_cath_labels.sh pdb
  ```

- [ ] **Bake CATH labels into AFDB LMDB**
  ```bash
  sbatch hpc-scripts/proteina/data_prep/bake_cath_labels.sh afdb
  ```

### After both LMDBs are enriched

- [ ] **Smoke-test dataloader with baked labels** — confirm `cath_code` is on the
  `Data` object when iterating the LMDB, without any transform needed

- [ ] **Wire up fold conditioning in training configs**
  - Dataset YAML (`pdb_lmdb.yaml`, `afdb_swissprot_lmdb_256.yaml`): transform lines
    can stay commented out since `cath_code` is baked in
  - Training config: `fold_cond: True`
  - Model config: confirm `feats_cond_seq: ["time_emb", "fold_emb"]`
    and `cath_code_dir` points to CATH vocabulary files

- [ ] **Launch a fold-conditioned training run**
  - Start with one dataset (PDB, residue-length 256) for speed
  - Compare FID + fold-class diversity vs unconditional baseline

- [ ] **Add CATH distribution figures to eventual writeup**

### Open questions to revisit

- **Conditioning level**: T-level (~1400 classes, fine-grained) vs C-level (4 classes,
  coarse). T-level is what the inference configs default to; C-level might be a
  simpler first experiment.
- **Multi-domain proteins**: some proteins have 2+ CATH domains. The model's
  `multilabel_mode` config has three options (`sample`, `average`, `transformer`) —
  worth confirming which we want.
- **Coverage gaps**: 44% (PDB) and 62% (AFDB) of proteins have CATH labels. For the
  rest, the model will see a null token. Should be fine for classifier-free guidance
  but worth measuring whether uncovered proteins are structurally biased.
- **Stratified splits**: should train/val be split so fold families don't leak?
  Would require a separate data-prep pass. Not a blocker for the first conditioned run.

---

## Key file locations

| File | Purpose |
|------|---------|
| `hpc-scripts/proteina/data_prep/analyse_cath_distribution.py` | Core characterisation logic |
| `hpc-scripts/proteina/data_prep/analyse_cath_pdb.sh` | Slurm: PDB characterisation |
| `hpc-scripts/proteina/data_prep/analyse_cath_afdb.sh` | Slurm: AFDB characterisation |
| `hpc-scripts/proteina/data_prep/bake_cath_labels.py` | LMDB enrichment logic |
| `hpc-scripts/proteina/data_prep/bake_cath_labels.sh` | Slurm: run enrichment (PDB or AFDB) |
| `hpc-scripts/proteina/data_prep/download_interpro_gene3d.sh` | Download + filter InterPro TSV |
| `evaluation/proteina/cath_distribution/` | Characterisation outputs (JSON + 4 PNGs) |
| `/rds/.../pdb_train/pdb_chain_cath_uniprot.tsv.gz` | SIFTS PDB→CATH (already on disk) |
| `/rds/.../pdb_train/cath-b-newest-all.gz` | CATH ID→code hierarchy (already on disk) |
| `/rds/.../afdb_swissprot/interpro_gene3d_swissprot.tsv` | InterPro Gene3D filtered to Swiss-Prot |
| `src/proteina/proteinfoundation/datasets/transforms.py` | `CATHLabelTransform`, `TEDLabelTransform` |
| `src/proteina/proteinfoundation/nn/feature_factory.py` | `FoldEmbeddingSeqFeat` (consumes `cath_code`) |
| `src/proteina/configs/experiment_config/training_ca.yaml` | `fold_cond` toggle |
