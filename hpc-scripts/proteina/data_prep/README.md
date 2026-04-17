# data_prep

Everything that builds or moves training data. Run in order (download → LMDB → length index → centroids); each step is resumable and checks for existing outputs.

## Scripts

| File | Role |
|---|---|
| `download_raw_pdb.sh` | Download raw PDB CIF files with retry logic. Runs on the `ampere` partition because `icelake` nodes can't resolve PDB servers. |
| `download_raw_pdb_fast.sh` | Parallel-wget version, ~780 files/min vs ~170 for the urllib path. Reads the CSV produced by the first run of `download_raw_pdb.sh`. |
| `convert_to_lmdb.sh` | Slurm wrapper: convert raw CIF/PDB → LMDB for the `pdb` or `d_FS` datasets. |
| `convert_to_lmdb.py` | The actual conversion logic invoked by the wrapper. |
| `prepare_data_dfs.sh` | Two-stage pipeline for the D_FS (AlphaFold DB clusters) dataset: download 588k AFDB files, then process to `.pt`. |
| `build_length_index.sh` | Slurm wrapper that calls `build_lmdb_length_index.py` on the training LMDB. One-off (~90 min for 425k entries). |
| `build_lmdb_length_index.py` | Scans LMDB splits, saves `*_lengths.npy` + `*_keys.pkl` so the dataset can filter by `max_num_residues` without re-scanning at startup. |
| `precompute_centroids.py` | Greedy TM-score clustering over training LMDB to produce novelty-evaluation centroids. No Slurm wrapper — run interactively on a GPU node. |

## Dependencies

- `DATA_PATH` env var must point to `/rds/.../proteina/data` (or equivalent).
- `download_raw_pdb*.sh` must run before `convert_to_lmdb.sh`.
- `convert_to_lmdb.sh` must run before `build_length_index.sh`.
