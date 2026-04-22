# CATH Dataset Characterisation & Conditioning Plan

## Goal

Characterise the PDB and AFDB Swiss-Prot training datasets by structural family using CATH
annotations, and eventually use those annotations for fold-class conditioning during proteina
training.

---

## Background

### Datasets

| Dataset | Path | Entries (est.) | Format |
|---------|------|----------------|--------|
| PDB | `/rds/.../pdb_train/lmdb/` | ~590k chains (train) | LMDB, PyG `Data` objects |
| AFDB Swiss-Prot | `/rds/.../afdb_swissprot/lmdb/` | ~540k proteins (train) | LMDB, PyG `Data` objects |

### CATH annotation sources

| Source | Covers | Files | Status |
|--------|--------|-------|--------|
| SIFTS + CATH | PDB | `pdb_chain_cath_uniprot.tsv.gz`, `cath-b-newest-all.gz` | **On disk** at `/rds/.../pdb_train/` |
| TED (Transfer of Evolutionary Data) | AFDB | `ted_365m.domain_summary.cath.globularity.taxid.tsv.gz` | **Not yet downloaded** (~19.9 GB gz, ~60 GB uncompressed) |

CATH hierarchy levels: **C** (class) → **A** (architecture) → **T** (topology) → **H** (homology).
PDB coverage expected ~70-80% of chains; AFDB Swiss-Prot expected high coverage.

### Existing codebase infrastructure

The codebase already has full fold-conditioning infrastructure:
- `CATHLabelTransform` (`transforms.py`) — attaches `graph.cath_code` at dataloader time for PDB
- `TEDLabelTransform` (`transforms.py`) — same for AFDB, reads TED TSV with pickle caching
- `FoldEmbeddingSeqFeat` (`feature_factory.py`) — embeds C/A/T levels into conditioning vectors
- Masking + classifier-free guidance in `model_trainer_base.py` (progressive level masking)
- Inference configs for fold-conditional sampling already exist

**Decision**: rather than running transforms at every training job startup (slow, RAM-heavy),
we bake `cath_code` directly into each LMDB entry once. The field is an optional list; old
code that doesn't read it is unaffected (backcompat guaranteed).

---

## What has been done

### 1. Investigation & design (this session)
- Confirmed SIFTS + CATH files are already on disk for PDB.
- Confirmed TED file is **not** on disk; Zenodo download is 19.9 GB compressed.
- Audited `CATHLabelTransform` and `TEDLabelTransform` — understood their lookup logic and
  output format (`graph.cath_code = ["1.10.150.10", ...]`).
- Audited `FoldEmbeddingSeqFeat` and training masking — confirmed the end-to-end conditioning
  pipeline is already implemented; just needs data + config toggles.
- Decided on "bake into LMDB" approach rather than runtime transform.

### 2. Scripts written

#### Characterisation (read-only, login-node friendly)
- **`hpc-scripts/proteina/data_prep/analyse_cath_distribution.py`**
  - Builds CATH lookup from SIFTS/TED files, scans LMDB splits, reports coverage and
    C/A/T-level distributions, saves JSON summary + matplotlib figures.
  - PDB works now; AFDB requires `--ted_tsv` argument.
  - Output dir: `evaluation/proteina/cath_distribution/`

#### LMDB enrichment (one-time write)
- **`hpc-scripts/proteina/data_prep/bake_cath_labels.py`**
  - Adds `graph.cath_code` to every LMDB entry in-place.
  - Idempotent: skips entries already enriched.
  - Backcompat: only adds the new field, never modifies or deletes others.
- **`hpc-scripts/proteina/data_prep/bake_cath_labels.sh`**
  - Slurm wrapper: `sbatch bake_cath_labels.sh pdb` or `sbatch bake_cath_labels.sh afdb`.

#### TED download
- **`hpc-scripts/proteina/data_prep/download_ted_afdb.sh`**
  - Downloads from Zenodo, decompresses to TSV.
  - Destination: `/rds/.../afdb_swissprot/ted_365m.domain_summary.cath.globularity.taxid.tsv`

---

## What to do next

### Immediate (no new data needed)

- [ ] **Smoke-test characterisation on PDB** (in progress — 2000-entry test running)
  ```bash
  source .venv/bin/activate
  python hpc-scripts/proteina/data_prep/analyse_cath_distribution.py \
      --max_entries 2000 --splits train
  ```
- [ ] **Run full PDB characterisation** once smoke-test passes
  ```bash
  python hpc-scripts/proteina/data_prep/analyse_cath_distribution.py \
      --out_dir evaluation/proteina/cath_distribution
  ```
- [ ] **Smoke-test LMDB enrichment** (dry-run, no writes)
  ```bash
  python hpc-scripts/proteina/data_prep/bake_cath_labels.py \
      --dataset pdb \
      --lmdb_dir /rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb \
      --cath_dir /rds/user/sr2173/hpc-work/proteina/data/pdb_train \
      --splits train --max_entries 500 --dry_run
  ```
- [ ] **Run PDB LMDB enrichment** once dry-run looks good
  ```bash
  sbatch hpc-scripts/proteina/data_prep/bake_cath_labels.sh pdb
  ```

### Requires TED download

- [ ] **Download TED AFDB file** (~19.9 GB, ~12h job)
  ```bash
  sbatch hpc-scripts/proteina/data_prep/download_ted_afdb.sh
  ```
- [ ] **Run AFDB characterisation** (add `--ted_tsv` to analyse script)
- [ ] **Run AFDB LMDB enrichment**
  ```bash
  sbatch hpc-scripts/proteina/data_prep/bake_cath_labels.sh afdb
  ```

### After both LMDBs are enriched

- [ ] **Wire up conditioning in training configs**
  - In dataset YAML (e.g. `pdb_lmdb.yaml` / `afdb_swissprot_lmdb_256.yaml`):
    remove the commented-out transform lines — transforms are no longer needed since
    `cath_code` is baked in, but confirm the dataloader still picks up the field correctly
    (it should, since it's already on the `Data` object).
  - In training config: set `fold_cond: True`.
  - In model config: confirm `feats_cond_seq: ["time_emb", "fold_emb"]` and
    `cath_code_dir` points to a dir with the CATH vocabulary files.
- [ ] **Launch a baseline fold-conditioned training run** and compare FID/diversity vs
  unconditional baseline.
- [ ] **Add CATH distribution figures to writeup** — the JSON + PNG outputs from the
  characterisation script can go directly into the report.

### Open questions

- Should we condition at T-level (topology, ~1400 classes) or C-level (4 classes) first?
  T-level is what the existing inference configs use; C-level is simpler to validate.
- AFDB Swiss-Prot coverage with TED: unknown until we run it. TED covers 365M AFDB entries
  but Swiss-Prot is ~550k — expect near-complete coverage for well-characterised proteins.
- Do we want to add CATH-stratified train/val splits (ensure fold families don't leak across
  splits)? The codebase supports MMseqs2 sequence-identity splitting already; CATH-stratified
  splitting would require a separate pass.

---

## Key file locations

| File | Purpose |
|------|---------|
| `hpc-scripts/proteina/data_prep/analyse_cath_distribution.py` | Characterisation script |
| `hpc-scripts/proteina/data_prep/bake_cath_labels.py` | LMDB enrichment script |
| `hpc-scripts/proteina/data_prep/bake_cath_labels.sh` | Slurm wrapper for enrichment |
| `hpc-scripts/proteina/data_prep/download_ted_afdb.sh` | TED download job |
| `evaluation/proteina/cath_distribution/` | Output: JSON + figures (created on first run) |
| `/rds/.../pdb_train/pdb_chain_cath_uniprot.tsv.gz` | SIFTS PDB→CATH mapping (on disk) |
| `/rds/.../pdb_train/cath-b-newest-all.gz` | CATH ID→code hierarchy (on disk) |
| `/rds/.../afdb_swissprot/ted_365m.domain_summary.cath.globularity.taxid.tsv` | TED file (to download) |
| `src/proteina/proteinfoundation/datasets/transforms.py` | `CATHLabelTransform`, `TEDLabelTransform` |
| `src/proteina/proteinfoundation/nn/feature_factory.py` | `FoldEmbeddingSeqFeat` |
| `src/proteina/configs/datasets_config/pdb/original/pdb_train.yaml` | Dataset config (transform toggle) |
| `src/proteina/configs/experiment_config/training_ca.yaml` | Training config (`fold_cond` toggle) |
