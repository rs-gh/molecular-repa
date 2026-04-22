# HPC scripts — proteina

Slurm wrappers and supporting Python for running the proteina pipeline on Wilkes3 (A100 80GB). Each subdir has its own README with per-script details.

## Layout

| Subdir | Purpose |
|---|---|
| [`data_prep/`](data_prep/) | Download raw PDB/AFDB, build LMDBs, precompute length index and novelty centroids |
| [`training/`](training/) | Launch baseline and REPA training runs |
| [`evaluation/`](evaluation/) | Full and lite FID evaluation, bundle/transfer lite runs to external servers |
| [`bench/`](bench/) | Performance benchmarks: torch.compile, LMDB I/O (Lustre vs NVMe), SDPA backends, max-batch-size sweep |
| [`smoke_tests/`](smoke_tests/) | "Does it run" sanity checks (fake data, in-memory, wandb, compile, spawn workers, …) |

## Typical pipeline

```
data_prep/download_raw_pdb{,_fast}.sh      # 1. Pull CIF files
data_prep/convert_to_lmdb.sh               # 2. Pack into LMDB
data_prep/build_length_index.sh            # 3. Precompute length index (one-off, ~90 min)
data_prep/precompute_centroids.py          # 4. Precompute novelty centroids (one-off)
training/pdb/train_baseline.sh             # 5. Train
evaluation/eval_fid_lite_sweep.sh          # 6. Lite FID across checkpoints (convergence curve)
evaluation/eval_fid.sh                     # 7. Full FID on final ckpt
```
