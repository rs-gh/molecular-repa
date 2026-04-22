# Datasets

One-stop reference for all training data used in this project — sources, filters, processing, and augmentations. Covers both **Proteina** (protein structure generation) and **Tabasco** (small-molecule generation).

---

## Proteina

### PDB (Protein Data Bank)

**What it is**
The Protein Data Bank is the global archive of experimentally determined 3D protein structures, determined by X-ray crystallography, cryo-EM, and NMR. All structures are peer-reviewed and deposited by the research community. PDB is the standard training set for protein structure generation models, including the original Proteina paper.

**Source**
Raw CIF files downloaded from RCSB PDB. Download scripts: [`download_raw_pdb.sh`](../../hpc-scripts/proteina/data_prep/download_raw_pdb.sh) (sequential, ~170 files/min) and [`download_raw_pdb_fast.sh`](../../hpc-scripts/proteina/data_prep/download_raw_pdb_fast.sh) (parallel wget, ~780 files/min).

**How our dataset size compares to the literature**
Many protein structure generation papers (FrameDiff, Chroma, FoldingDiff, and Proteina itself) cite training on "~20k structures" from the PDB. Our 425k train chains (~600k including all splits) is dramatically larger. The difference comes from several deliberate choices:

| Factor | Typical paper | Our setup |
|---|---|---|
| Sequence deduplication | MMseqs2/CD-HIT at 30–40% identity → ~20–30k clusters | None — random split on raw chains |
| Resolution cutoff | ≤ 2.0–3.0 Å (X-ray only or strict cryo-EM) | ≤ 5.0 Å (includes lower-quality cryo-EM) |
| Unit counted | Structure or sequence cluster | Chain (multi-chain structures expand ~3×) |
| NMR | Excluded | Excluded (same) |

The "~20k" figure refers to **sequence-similarity clusters**, not raw structures. After 30% identity clustering, many highly similar proteins (same enzyme, different organism, point mutations) collapse to a single representative — that's the main driver. We include all of them. Whether training on a larger, more redundant dataset helps or hurts generation quality is an open question; the papers use clustering to avoid over-representing common protein families, which can bias the model's generation distribution.

**Filters (applied post-download, before LMDB build)**
All filters are applied by `PDBDataSelector` during LMDB construction (`convert_to_lmdb.py`):

| Filter | Value |
|---|---|
| Molecule type | protein only |
| Experiment type | diffraction, EM (no NMR) |
| Min residues | 50 |
| Max residues | 512 (full), 256 or 128 for shorter-context runs |
| Resolution | 0.0–5.0 Å |
| Ligands | none required, none excluded (ligands stripped) |
| Non-standard residues | removed |
| Unavailable PDB files | removed |
| Oligomeric state | no filter |

**Structure vs chain**
The CSV and LMDB are indexed by **chain** (e.g. `1abc_A`), not by structure (PDB code). A single structure can contain multiple chains — a homodimer has one PDB code but two chain entries. The 194,090 unique structures in the CSV yield ~591,632 chain rows. Each chain is filtered independently, so some structures contribute all their chains and others only a subset (e.g. one chain passes the 512-residue limit while another doesn't).

**Split**
Random 98% / 1.9% / 0.1% train/val/test (chain-level). Split type is `random` — MMseqs2 sequence-similarity clustering is not used on CSD3. The splitter uses a fixed internal seed (42) for determinism; the same CSV + ratios always produces the same partition.

**Dataset sizes (LMDB, max_residues=512, as built 2026-04-22)**

| Split | Entries | Notes |
|---|---|---|
| train | 425,100 chains | 98% of parsed chains |
| val | 5,000 chains | held-out probe suite |
| test | 500 chains | held-out, never used during development |

Source CSV: 591,632 chain rows from 194,090 unique PDB structures (all downloaded). Gap from ~591k → ~430k total LMDB entries is due to the 98/1.9/0.1 split plus parse failures (malformed CIFs, missing atoms, etc.).

**Processing (LMDB build)**
Each CIF file is parsed with graphein → PyG `Data` object:
- atom37 coordinate tensor (OpenFold convention, reordered from PDB atom ordering)
- `coord_mask`: bool tensor marking filled vs. padded atom positions
- `residue_type`: integer index per residue (OpenFold `resname_to_idx`)
- `bfactor_avg`: mean B-factor per residue
- `residue_pdb_idx`: original PDB residue numbering
- `seq_pos`: sequential position index

**Runtime augmentations (applied per batch during training)**
- `GlobalRotationTransform`: random SO(3) rotation of all coordinates
- `ChainBreakPerResidueTransform`: marks chain breaks in the sequence position encoding
- `PaddingTransform`: zero-pads to fixed `max_size` (128, 256, or 512 depending on config) for constant tensor shapes required by `torch.compile`

**Configs**
Dataset configs live at [`src/proteina/configs/datasets_config/pdb/`](../../src/proteina/configs/datasets_config/pdb/). Key configs:
- `pdb_lmdb.yaml` — max 512 residues, batch 7
- `pdb_lmdb_256.yaml` — max 256 residues, batch 24
- `pdb_lmdb_128.yaml` — max 128 residues, batch 80

---

### AFDB Swiss-Prot

**What it is**
The AlphaFold Database (AFDB) is EBI's public repository of AlphaFold2-predicted protein structures covering essentially all of UniProtKB. We use the Swiss-Prot subset specifically.

**UniProt / Swiss-Prot background**
UniProt is the canonical protein sequence registry. It has two tiers:
- **Swiss-Prot** (~570k proteins): manually curated by experts — function, domains, post-translational modifications all experimentally verified. Every entry has been reviewed by a human curator.
- **TrEMBL** (~250M proteins): auto-annotated from genome sequencing; much larger but lower confidence and high redundancy.

AlphaFold predicted structures for essentially all of UniProtKB. AFDB v6 is the current version. The structures themselves are *predicted*, not experimentally determined — but the underlying sequences are Swiss-Prot quality (manually verified).

**pLDDT confidence score**
Every AFDB structure includes a per-residue pLDDT (predicted Local Distance Difference Test) confidence score stored in the B-factor column. Values 0–100: >90 = very high confidence, 70–90 = high, 50–70 = low, <50 = very low. We preserve this as `bfactor_avg` in the LMDB, mirroring the PDB B-factor field.

**Why Swiss-Prot (not the paper's d_FS dataset)**
The Proteina paper trains on `d_FS`, a Foldseek-clustered subset of 588k AFDB TrEMBL structures. This dataset is broken: the NVIDIA-provided index references UniProt accessions that no longer exist in AFDB v6 (AlphaFold DB pruned large numbers of TrEMBL entries between v4 and v6). All 178,560 "downloaded" files from the original script were 127-byte XML error pages. Replicating d_FS exactly would require downloading all 214M AFDB structures and re-running Foldseek + MMseqs2 clustering — not feasible.

Swiss-Prot is a practical alternative: `swissprot_pdb_v6.tar` (27 GB) is a confirmed live EBI FTP download containing all ~550k Swiss-Prot AlphaFold v6 structures in a single file.

**Source**
Downloaded as a single tar archive from EBI FTP: [`download_afdb_swissprot.sh`](../../hpc-scripts/proteina/data_prep/download_afdb_swissprot.sh).
Path on RDS: `/rds/user/sr2173/hpc-work/proteina/data/afdb_swissprot/swissprot_pdb_v6.tar`
Format: `.pdb.gz` files inside the tar (gzip-compressed PDB format).
Downloaded: 2026-04-22, job 28190177.

**Filters**

| Filter | Where applied | Value |
|---|---|---|
| Max residues | Build-time (LMDB builder) | 256 |
| Min residues | None | — |
| Sequence similarity split | Not applied | AFDB Swiss-Prot has low redundancy by design; random split used |
| Resolution / experiment type | Not applicable | Predicted structures, no experimental metadata |

Filters are applied during LMDB construction by [`build_afdb_lmdb.py`](../../hpc-scripts/proteina/data_prep/build_afdb_lmdb.py), which streams the tar without extracting to disk.

**Split**
Random 98% / 1.9% / 0.1% train/val/test. Split assignments computed on first run and saved to `splits.json` alongside the tar for reproducibility. No sequence-similarity clustering (unlike PDB) — Swiss-Prot proteins are already manually curated and have low within-dataset redundancy.

**Dataset sizes (LMDB, max_residues=256, built 2026-04-22)**

| Split | Entries written | Filtered (>256 res) | LMDB size |
|---|---|---|---|
| train | 229,670 | 309,450 | 19 GB |
| val | 4,521 | 5,931 | 379 MB |
| test | 235 | 315 | 19 MB |
| **total** | **234,426** | **315,696** | **~19.4 GB** |

~57% of the 550,122 raw structures were filtered out for exceeding 256 residues. Total members in source tar: 550,122. 0 parse failures.

**Processing (LMDB build)**
[`build_afdb_lmdb.py`](../../hpc-scripts/proteina/data_prep/build_afdb_lmdb.py) streams the tar in two passes:
1. **Pass 1** (~30–60s): read tar headers only to collect all protein ID stems, compute split assignments, save `splits.json`
2. **Pass 2** (hours): stream tar data blocks, decompress each `.pdb.gz` in memory, write to `NamedTemporaryFile`, parse with graphein → PyG `Data`, delete temp file, write to LMDB. Commits every 500 entries (durable progress). SIGTERM handler flushes partial batch for graceful SLURM preemption.

Per-structure processing mirrors PDB:
- `protein_to_pyg` (graphein) parses PDB → PyG Data with atom37 coords
- `num_nodes` set explicitly as `coords.shape[0]` (graphein does not set this automatically)
- `coord_mask`, `residue_type`, `bfactor_avg` (= pLDDT here), `residue_pdb_idx`, `seq_pos` all added
- Coordinate reorder: PDB → OpenFold atom37 convention

**Runtime augmentations**
Same as PDB: `GlobalRotationTransform`, `ChainBreakPerResidueTransform`, `PaddingTransform` (max_size=256).

**Config**
[`src/proteina/configs/datasets_config/afdb/afdb_swissprot_lmdb_256.yaml`](../../src/proteina/configs/datasets_config/afdb/afdb_swissprot_lmdb_256.yaml) — max 256 residues, batch 24.

---

## Tabasco

### QM9

**What it is**
QM9 is a standard small-molecule benchmark dataset containing 133,885 organic molecules with up to 9 heavy atoms (C, N, O, F), with DFT-computed 3D geometries and quantum chemical properties.

**Source**
Pre-processed `.pt` files from HuggingFace (`carlosinator/tabasco-qm9`), stored locally at `src/tabasco/data/`:
- `processed_qm9_train.pt` — 95,793 molecules
- `processed_qm9_val.pt`
- `processed_qm9_test.pt`

LMDB at `src/tabasco/data/lmdb_qm9/`.

**Filters**
- Max heavy atoms: 9 (QM9 dataset property — all molecules have ≤9 heavy atoms)
- Atom types: C, N, O, F, S, Cl, Br, I (+ padding dummy `*`)
- Hydrogens: removed at load time (`remove_hydrogens: true`)

**Dataset size**
- Train: 95,793 molecules
- `ATOM_DIM = 9`, `max_mol_num_atoms = 9`

**Processing and augmentations**
- `remove_hydrogens`: heavy atoms only
- `add_random_rotation`: random SO(3) rotation per sample
- `reorder_to_smiles_order`: atom ordering matched to RDKit canonical SMILES
- Padding: variable-length molecules padded to `max_mol_num_atoms=9` with dummy `*` atom (one-hot index 8, not all-zero)

---

### GEOM-Drugs

**What it is**
GEOM (Geometric Ensemble Of Molecules) is a large-scale dataset of drug-like molecules with DFT-refined 3D conformers. The "drugs" subset covers larger, more complex molecules than QM9.

**Source**
Pre-processed `.pt` files from HuggingFace (`carlosinator/tabasco-geom-drugs`), stored locally at `src/tabasco/data/`:
- `processed_geom_train.pt` — 1,142,099 molecules
- `processed_geom_val.pt`
- `processed_geom_test.pt`

LMDB at `src/tabasco/data/lmdb_geom/`.

**Filters**
- Atom types: C, N, O, F, S, Cl, Br, I (+ padding dummy `*`)
- Hydrogens: removed at load time

**Dataset size**
- Train: 1,142,099 molecules
- Batch size: 256 (typical); ~58 unique SMILES per batch (high intra-batch SMILES cache hit rate for ChemPropEncoder)

**Processing and augmentations**
- `remove_hydrogens`: heavy atoms only
- `add_random_rotation`: random SO(3) rotation per sample
- `reorder_to_smiles_order`: atom ordering matched to RDKit canonical SMILES
- `add_random_permutation`: off by default
- Padding: variable-length molecules padded to `max_mol_num_atoms` (dataset-level maximum, computed at startup) with dummy `*` atom (one-hot index 8)
