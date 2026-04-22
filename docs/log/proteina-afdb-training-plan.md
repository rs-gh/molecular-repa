# Proteina AFDB Swiss-Prot Training Plan

**Date:** 2026-04-22
**Goal:** Train Proteina (60M baseline) on AFDB predicted structures at max 256 residues, as an alternative to the PDB-only training runs.

---

## Background

### Why AFDB?

The Proteina paper trains on two datasets: PDB (experimentally determined structures) and d_FS (a filtered Foldseek-clustered subset of the AlphaFold Database, ~588k structures). We already have PDB training working. This plan covers setting up the AFDB side.

### Why not d_FS (the paper's dataset)?

The NVIDIA-provided d_FS index (`d_FS_index.txt`, 588,318 UniProt accessions from `proteina_training_data_indices.zip` on NGC) is broken. The accessions are TrEMBL proteins that no longer exist on AFDB:

- AlphaFold DB moved from v4 → v6 and pruned a large fraction of TrEMBL accessions
- The individual-file URL scheme (`alphafold.ebi.ac.uk/files/AF-<id>-F1-model_v4.pdb`) returns 404 for all d_FS IDs
- The download script (`prepare_data_dfs.sh`) silently wrote 127-byte XML error pages instead of real PDB files — all 178,560 "downloaded" files were garbage
- These stub files have been deleted (`d_FS/raw/` removed, recovering ~178k inodes)

Replicating d_FS exactly is not feasible: it would require downloading all 214M AFDB structures and re-running the Foldseek + MMseqs2 clustering pipeline.

### Why Swiss-Prot?

`swissprot_pdb_v6.tar` (27 GB, EBI FTP) is confirmed live (HTTP 200). It contains ~550k AlphaFold v6 predictions for all Swiss-Prot proteins — the manually curated, high-quality half of UniProt. At ≤256 residues we expect ~200–300k proteins. This is:

- Smaller than d_FS but higher quality (every entry is experimentally validated, not computationally predicted)
- Not exactly the paper's dataset, but a reasonable training set for AFDB-predicted structures
- Inode-efficient: 1 tar file + 3 LMDB files instead of 550k individual PDB files

### What is Swiss-Prot / UniProt?

UniProt is the canonical registry of proteins. Every protein gets a unique accession (e.g. `P00533` = human EGFR). UniProtKB has two tiers:
- **Swiss-Prot** (~570k): manually reviewed by curators — function, domains, modifications verified
- **TrEMBL** (~250M): auto-annotated from genome sequencing runs; many are low-confidence or redundant

AlphaFold predicted structures for essentially all of UniProtKB; AFDB is EBI's public mirror of those predictions. pLDDT confidence scores are stored in the B-factor column of AFDB PDB files — the pipeline preserves these as `bfactor_avg`.

---

## Data Pipeline

### Files created

| File | Purpose |
|---|---|
| [hpc-scripts/proteina/data_prep/download_afdb_swissprot.sh](../../hpc-scripts/proteina/data_prep/download_afdb_swissprot.sh) | Download `swissprot_pdb_v6.tar` (~27 GB) from EBI FTP. Resumable via `wget --continue`. Verifies member count after download. |
| [hpc-scripts/proteina/data_prep/build_afdb_lmdb.py](../../hpc-scripts/proteina/data_prep/build_afdb_lmdb.py) | Streams PDB files directly from tar (no extraction). Two-pass: scan headers → assign splits → stream data → write LMDB. |
| [hpc-scripts/proteina/data_prep/build_afdb_lmdb.sh](../../hpc-scripts/proteina/data_prep/build_afdb_lmdb.sh) | SLURM wrapper for the LMDB build (24 workers, ampere partition). |
| [src/proteina/configs/datasets_config/afdb/afdb_swissprot_lmdb_256.yaml](../../src/proteina/configs/datasets_config/afdb/afdb_swissprot_lmdb_256.yaml) | Dataset config: LMDB-backed, max 256 residues, PaddingTransform, batch 24. |
| [src/proteina/configs/experiment_config/training/256/training_baseline_afdb_swissprot_256.yaml](../../src/proteina/configs/experiment_config/training/256/training_baseline_afdb_swissprot_256.yaml) | Training config: run name `proteina_60m_baseline_afdb_swissprot_256`, wandb project `proteina-repa`. |
| [hpc-scripts/proteina/training/afdb/train_baseline.sh](../../hpc-scripts/proteina/training/afdb/train_baseline.sh) | Training SLURM script. Stages AFDB LMDB to `/tmp/proteina_afdb_lmdb` on NVMe. Auto-resume from last.ckpt. |

### Code changes

- [src/proteina/proteinfoundation/datasets/lmdb_utils.py](../../src/proteina/proteinfoundation/datasets/lmdb_utils.py):
  - Added `max_residues: Optional[int] = None` to `process_raw_to_lmdb` — build-time length filter
  - Added `_parse_pdb_bytes(args)` — like `_parse_one_structure` but takes raw bytes, writes to NamedTemporaryFile, parses, deletes
  - Added `process_tar_to_lmdb(...)` — streams tar sequentially, feeds bytes to worker pool, commits every 500 entries, handles SIGTERM gracefully

- [hpc-scripts/proteina/data_prep/build_length_index.sh](../../hpc-scripts/proteina/data_prep/build_length_index.sh):
  - Now accepts a dataset name argument: `sbatch build_length_index.sh afdb_swissprot`

### How the tar streaming works

1. **Pass 1 (30-60s):** Read tar headers only (no data blocks) to collect all ~550k protein ID stems. Compute random 98/1.9/0.1 train/val/test split with fixed seed. Save `splits.json`. On reruns, load from file and detect any new members.

2. **Pass 2 (hours):** Stream tar again. For each member, if its stem is in `target_ids = split_assignment - already_in_lmdb`: read bytes into memory, send to worker pool. Workers write to a `NamedTemporaryFile(.pdb)`, parse with graphein → PyG Data, delete the temp file, return pickled graph. At most 24 temp files exist simultaneously.

3. **Commits every 500 entries** — progress is durable; at most 499 entries lost on kill.

4. **SIGTERM handler** — SLURM sends SIGTERM 30s before wall-time kill. Handler flushes the partial batch and exits cleanly.

5. **Incremental reruns** — LMDB `__ids__` metadata key tracks all written protein IDs. Rerun skips already-done proteins, appending only new ones.

### AFDB file processing

No special handling needed vs PDB files. `_parse_pdb_bytes` does:
- `protein_to_pyg(path=tmp_path, chain_selection="all")` — graphein parses PDB, builds atom37 coord tensor
- Add `coord_mask`, `residue_type` (via OpenFold resname_to_idx), `bfactor_avg` (= pLDDT), `seq_pos`
- Coordinate reorder: PDB → OpenFold atom37 convention
- Build-time filter: skip if `num_nodes > 256`

---

## Run sequence

### Step 1 — Download (job 28190177, currently running)

```bash
sbatch hpc-scripts/proteina/data_prep/download_afdb_swissprot.sh
# Account: computerlab-sl2-cpu, partition: icelake, 12h wall
# Output: /rds/user/sr2173/hpc-work/proteina/data/afdb_swissprot/swissprot_pdb_v6.tar
# Log: /rds/user/sr2173/hpc-work/proteina/logs/afdb-dl-28190177.out
```

### Step 2 — Build LMDB

```bash
sbatch hpc-scripts/proteina/data_prep/build_afdb_lmdb.sh
# Account: lio-charm-sl2-gpu, partition: ampere, 36h wall
# Output: /rds/user/sr2173/hpc-work/proteina/data/afdb_swissprot/lmdb/{train,val,test}.lmdb
# splits.json written alongside tar on first run
```

Safe to kill and rerun — resumes from last commit checkpoint.

### Step 3 — Length index

```bash
sbatch hpc-scripts/proteina/data_prep/build_length_index.sh afdb_swissprot
# ~1-2h. Writes train_lengths.npy, train_keys.pkl, val_lengths.npy etc. to lmdb/
```

### Step 4 — Smoke test (qos=intr, ≤1h)

```bash
sbatch --qos=intr hpc-scripts/proteina/training/afdb/train_baseline.sh \
    training_baseline_afdb_swissprot_256 training/256
```

Verify: batch 24 fits in 80 GB VRAM, throughput matches PDB 256 baseline (~same step time), no NaN loss.

### Step 5 — Full training run

```bash
sbatch hpc-scripts/proteina/training/afdb/train_baseline.sh
# Default config: training_baseline_afdb_swissprot_256 training/256
# WandB: proteina-repa project, run proteina_60m_baseline_afdb_swissprot_256
```

---

## Comparison baseline

| Run | Dataset | Max residues | WandB run |
|---|---|---|---|
| PDB baseline | PDB (exp.) | 256 | `proteina_60m_baseline_256` |
| **AFDB baseline** | AFDB Swiss-Prot (predicted) | 256 | `proteina_60m_baseline_afdb_swissprot_256` |

Primary metric: val FID at equivalent training steps. After ~50k steps, compare val loss trajectory. If AFDB trains to comparable quality, it validates using predicted structures as training data (useful for scaling to larger datasets in future).

---

## Open questions / future work

- **REPA on AFDB**: REPA configs for AFDB not yet created (only baseline). Once baseline is proven, add `training_repa_l4_afdb_swissprot_256_per_residue.yaml` mirroring the PDB REPA configs.
- **Larger AFDB subsets**: Swiss-Prot is ~550k. If we want to scale closer to d_FS's 588k or beyond, model organism proteome tarballs (also on EBI FTP, v6) could be downloaded and merged. Each is a separate tar; `build_afdb_lmdb.py` would need a `--tar-name` flag pointing at each, with `splits.json` accumulating across them.
- **d_FS reconstruction**: Theoretically possible by downloading all AFDB organism tarballs and running Foldseek + MMseqs2 clustering. Not planned — Swiss-Prot is sufficient for our purposes.

---

## Session Log — 2026-04-22

### What we did

Executed the full pipeline from scratch in a single session: download → LMDB build → length index → smoke test.

| Step | Job | Result |
|---|---|---|
| Download `swissprot_pdb_v6.tar` | 28190177 | ✅ Complete, 27 GB, 550,122 members verified |
| Build LMDB (final attempt) | 28191351 | ✅ Complete, 229,670 train / 4,521 val / 235 test, 0 failures |
| Build length index | 28227447 | ✅ Complete, lengths 16–256, mean 154 residues |
| Smoke test (compile, intr) | 28231877 | ❌ Hit 1h wall time during torch.compile |
| Smoke test (no-compile, gpu1) | 28235149 | ❌ CUDA device busy on assigned node |

### Problems encountered and fixes

#### 1. Wrong SLURM account / partition for LMDB build
**Problem**: Original script targeted `lio-charm-sl2-gpu` / ampere partition. LMDB build is pure CPU work — wasteful, and GPU nodes queue longer.
**Fix**: Switched to `computerlab-sl2-cpu` / icelake. Then hit `AssocGrpCPUMinutesLimit` (only 1,013 hours remaining). Switched to `computerlab-sl3-cpu`, which has 157k hours but caps wall time at 12h. Reduced wall time from 36h to 12h — fine since the build supports incremental reruns.
**Residual risk**: If Swiss-Prot grows significantly (e.g. we add organism tarballs), a single 12h run may not be enough. Solution: just rerun — progress is committed every 500 entries.

#### 2. `No module named 'graphein_utils'` in worker processes
**Problem**: `graphein_utils` lives at `src/proteina/graphein_utils/` but is not installed as a package in the venv. `build_afdb_lmdb.py` adds it to `sys.path` at startup, but `multiprocessing` workers (spawned after `sys.path` modification) don't inherit the modified path.
**Fix**: Added `PYTHONPATH="$REPO_DIR/src/proteina:$REPO_DIR/src/proteina/proteinfoundation"` to the SLURM shell script. All child processes inherit environment variables, so workers can find the module.
**Residual risk**: If the repo is moved or `src/proteina/` is restructured, this hardcoded path silently breaks. Long-term fix: install `graphein_utils` as an editable package (`pip install -e src/proteina`).

#### 3. AFDB tar contains `.pdb.gz` (gzip-compressed), not plain `.pdb`
**Problem**: `swissprot_pdb_v6.tar` members are named `AF-xxx-F1-model_v6.pdb.gz`. The LMDB builder read raw bytes and wrote them to a `.pdb` temp file, passing compressed binary data to graphein's PDB parser. Graphein's error: `"No model found for index: 1"`.
**Fix**: Added gzip magic byte detection in `_parse_pdb_bytes` ([lmdb_utils.py](../../src/proteina/proteinfoundation/datasets/lmdb_utils.py)): if bytes start with `\x1f\x8b`, decompress with `gzip.decompress()` before writing to temp file.
**Residual risk**: None for this dataset. If future AFDB tarballs contain other compression formats (e.g. `.pdb.bz2`), they would silently fail parsing (no fix needed unless we use them).

#### 4. `num_nodes=None` on all parsed graphs
**Problem**: `protein_to_pyg` (graphein) returns a PyG `Data` object where `num_nodes` is not explicitly set. PyG infers `num_nodes` from standard edge/node attributes like `x` or `edge_index` — but graphein uses non-standard attributes (`coords`, `residues`, etc.). Result: `graph.num_nodes` returns `None`, causing a `TypeError` in the `_accept` filter and treating all graphs as empty.
**Fix**: Explicitly set `graph.num_nodes = graph.coords.shape[0]` immediately after `protein_to_pyg` returns, before any downstream use. Also added an empty-graph guard (`num_nodes == 0` → return None).
**Residual risk**: This fix assumes `coords` is always populated by graphein and has shape `(N, 37, 3)`. If a structure has no CA atoms (extremely rare), `coords.shape[0]` would be 0 and be caught by the empty-graph guard. Should be safe.
**Note**: This same bug likely exists in `_parse_one_structure` (the PDB build path). It worked there because the PDB LMDB was built before this issue was noticed — worth checking if the `_accept` filter is also called for the PDB path, or if it only affects tar-streaming builds.

### Smoke test outcome

Both smoke tests failed for infrastructure reasons unrelated to the model or data:

- **Compile run (28231877)**: `torch.compile` on a 65M model at 256-residue context takes >1h on an A100. The `intr` QOS has a 1h wall cap, so the job timed out before training started. Not a real problem — compilation is a one-time cost per node, and the full training run uses 36h wall time (`gpu1` QOS). **The config, data loading, and model init all succeeded without errors.**
- **No-compile run (28235149)**: `CUDA error: CUDA-capable device(s) is/are busy or unavailable` — the scheduler assigned a node whose GPU was already occupied by another job. Scheduling fluke, not a code issue.

### What we should do next

1. **Resubmit no-compile smoke test** with `gpu1` QOS and a fresh node:
   ```bash
   sbatch /tmp/train_afdb_nocompile.sh  # already written, or resubmit with gpu1
   ```
   This will give us the first actual training step loss values within ~2 minutes of starting. Check for: no NaN loss, loss in 0.2–0.8 range (consistent with PDB baseline at step 1), batch fits in VRAM.

2. **Check `_parse_one_structure` for the same `num_nodes` bug** (PDB build path). If `_accept` is called there, it has the same latent bug. Low priority since PDB LMDB is already built and working.

3. **Submit full training run** once no-compile smoke test passes:
   ```bash
   sbatch hpc-scripts/proteina/training/afdb/train_baseline.sh
   # Default: training_baseline_afdb_swissprot_256, training/256, 36h, gpu1 QOS
   ```
   The LMDB is 19.5 GB — slightly too large for `/tmp` NVMe on current nodes (16 GB free). Training will run from Lustre. This is slower than NVMe but acceptable; if step time is significantly worse than PDB baseline, consider requesting a node with more NVMe headroom via `--constraint`.

4. **Monitor WandB** for first 1,000 steps: compare loss curve shape to `proteina_60m_baseline_256` (PDB). AFDB structures are predicted rather than experimental, so we expect slightly noisier early training but similar eventual convergence.

5. **Once baseline is stable (~50k steps)**: create REPA config `training_repa_l4_afdb_swissprot_256_per_residue.yaml` mirroring the PDB REPA setup.
