# bench

Performance benchmarks for proteina training. Each variant runs in a fresh spawn subprocess (compile cache wiped, OOM isolated) and reports steady-state `steps/sec` after dropping a configurable warmup window — so compile overhead doesn't poison the mean.

Results land in `evaluation/proteina/results/bench/*.csv`.

## Scripts

| File | Role |
|---|---|
| `_bench_common.py` | Shared helpers: subprocess runner, step-timer Lightning callback, warmup-skip summarizer, fake PDB datamodule. |
| `benchmark_compile.py` | `compile_mode ∈ {off, default, reduce-overhead, max-autotune}` × `seq_len` × `model_type` on fake data. Isolates the compile question. |
| `benchmark_io.py` | Real LMDB, no model. `source ∈ {lustre, nvme}` × `num_workers` × `pin_memory`. Auto-copies LMDB to `/tmp` when `nvme` requested. |
| `benchmark_e2e.py` | Full training loop against real LMDB. `compile × lmdb_source × gc_layers`. Validates that compile + I/O wins translate end-to-end. |
| `benchmark_sdpa.py` | Forces each SDPA backend via `torch.nn.attention.sdpa_kernel`. `backend ∈ {default, flash, efficient, math, cudnn}` × `compile`. |
| `batch_size_sweep.py` | Binary-searches max batch size before OOM for each `(seq_len, model_type, compile, gc_layers)` combination. |
| `batch_size_sweep.sh` | Slurm wrapper for `batch_size_sweep.py`. |
| `run_all.sh` | Slurm wrapper that runs the four per-variant benchmarks sequentially. Supports `--only compile,io`. |

## Usage

```bash
# Everything at once (~90 min)
sbatch hpc-scripts/proteina/bench/run_all.sh

# One benchmark
sbatch hpc-scripts/proteina/bench/run_all.sh --only compile

# Max-batch-size search (separate)
sbatch hpc-scripts/proteina/bench/batch_size_sweep.sh
```
