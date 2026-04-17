# smoke_tests

"Does it run" checks for isolated parts of the pipeline. Fast, stateless, meant to catch regressions (import breakage, CUDA issues, config drift) without a full training run. For throughput/perf measurements see `../bench/` instead.

## Scripts

| File | What it verifies |
|---|---|
| `smoke_test.sh` | End-to-end pipeline on a tiny PDB subset — the catch-all sanity check. |
| `smoke_test_lmdb.sh` | Training pipeline reads from LMDB correctly. |
| `smoke_test_fake_data.sh` | Fake-data path — measures GPU-only throughput ceiling (no I/O). |
| `smoke_test_inmemory.sh` | `in_memory=True` path, real shapes but no I/O. |
| `smoke_test_compile.sh` | `torch.compile` produces a working compiled model. |
| `smoke_test_compile_ab.sh` | A/B: compile vs no-compile at `max_size=512`, compares `s/epoch`. |
| `smoke_test_sdpa.sh` | SDPA attention path passes its unit tests. |
| `smoke_test_sdpa_ab.sh` | 2×2 matrix of `(manual, sdpa) × (no-compile, compile)`. |
| `smoke_test_spawn_workers.sh` | `num_workers ∈ {1,2,4}` with spawn start method (guards against fork+CUDA segfaults). |
| `smoke_test_wandb.sh` | WandB logging pipeline works. |
| `smoke_test_diagnostic.sh` | Baseline timing + GPU util + `/dev/shm` checks. |
| `smoke_256_batchsize.sh` | Find max BS for REPA at `max_size=256`, 20 steps. |
| `smoke_genmetrics.sh` | `ProteinGenerationMetricsCallback` fires correctly during training. |

> **Note:** `build_length_index.sh` used to live here but is actually a one-off data-prep step — it now lives in `../data_prep/`.
