# Tabasco performance benchmarks

Mirror of `hpc-scripts/proteina/bench/` — a scaffold for measuring tabasco's
training performance before changing anything. Every script runs each
variant in a **spawn subprocess** so torch.compile cache, CUDA state, and
peak-memory stats start clean for each trial. Warmup steps are dropped so
steady-state throughput is reported instead of compile-warmup-skewed
averages.

## Scripts

| Script | What it measures | Fake data? |
|---|---|---|
| `benchmark_compile.py` | `compile_mode ∈ {off, default, reduce-overhead, max-autotune}` × `precision ∈ {16, bf16-mixed}` × experiment | yes |
| `benchmark_sdpa.py` | Forces each SDPA backend via `torch.nn.attention.sdpa_kernel`; reports throughput + which backends the inputs are accepted by | yes |
| `diagnose_sdpa.py` | Standalone probe — reproduces tabasco's MHA call shapes (B=256, N=71, H=8, D=16, bf16/fp16, with/without padding mask), and also a "bare SDPA" variant, to see which backends each input shape unlocks | synth tensors |
| `batch_size_sweep.py` | Binary-searches max batch size before OOM per `(experiment, compile_mode, gc_layers)` | yes |
| `benchmark_io.py` | Real LMDB dataloader latency: `num_workers` × `pin_memory`. Probes `num_workers>0` cautiously (known to segfault on this cluster — see `feedback_num_workers.md`) | no (real LMDB) |

All scripts share [_bench_common.py](_bench_common.py) (`StepTimer`,
subprocess runner, cache wiper, CSV writer, grad-ckpt helper,
`load_tabasco_cfg`, `build_fake_tabasco_datamodule`).

## Output

CSVs land in [evaluation/tabasco/bench/results/](../../../evaluation/tabasco/bench/results/).

## Running

One-shot (sweeps all four benchmarks; ~60 min on A100 via `--qos=intr`):

```bash
sbatch hpc-scripts/tabasco/bench/run_all.sh
```

Single benchmark:

```bash
sbatch hpc-scripts/tabasco/bench/run_all.sh --only compile
sbatch hpc-scripts/tabasco/bench/run_all.sh --only sdpa,bs_sweep
sbatch hpc-scripts/tabasco/bench/batch_size_sweep.sh
```

Local / interactive:

```bash
source .venv/bin/activate
export PROJECT_ROOT=$(pwd)/src/tabasco
python hpc-scripts/tabasco/bench/benchmark_compile.py \
    --experiments geom/mild \
    --compile_modes off reduce-overhead \
    --precisions 16 bf16-mixed \
    --num_steps 100 --warmup 30
```

## Experiments covered

- `geom/mild` — baseline flow model, no REPA
- `geom/chemprop_tradeoff` — REPA with CheMeleon (ChemPropEncoder)
- `geom/mace_cached_tradeoff` — REPA with MACE cached embeddings

QM9 is intentionally excluded — it's not a production workload.

## Methodology

- **Warmup drop**: 30 steps (`WARMUP_STEPS_DEFAULT`). Compile warmup
  contaminates the first batch of iterations with graph recording; we
  report median, p10, p90, p99 over the remaining steps.
- **Subprocess isolation**: each variant runs in a fresh `spawn` process.
  Benefits: OOM kills the child cleanly (parent reports `oom_or_crash`);
  inductor cache is cleared between trials; torch / hydra state is not
  tainted from the previous trial.
- **CUDA sync at step boundaries**: `StepTimer` calls
  `torch.cuda.synchronize()` before and after each step so async kernel
  launches are not charged to the next step.
- **Fake data** (for compile/sdpa/bs_sweep): minimal TensorDict batches
  matching tabasco's real LMDB output shape (coords, atomics, padding_mask,
  optional `smiles`/`lmdb_key` non-tensor fields for REPA variants).
  Isolates the model question from I/O noise.
- **Peak memory**: `torch.cuda.max_memory_allocated()` after
  `trainer.fit()`, reset at the start of each trial.

## Interpreting results

- `status=ok`: trial completed, timing columns populated.
- `status=oom`: CUDA OOM. Peak mem at crash is not recorded (subprocess
  gone); use the previous-best trial's peak for memory headroom.
- `status=oom_or_crash`: subprocess exit code non-zero and no result in the
  queue. Usually means the OS killed the process (OOM-kill, segfault). Log
  files have context.
- `status=unsupported`: an SDPA backend explicitly rejected the inputs
  ("no available kernel" / "not supported"). Run `diagnose_sdpa.py` to see
  which call-shape features are blocking dispatch.

## Baseline context

Current production defaults (see [../../../src/tabasco/configs/](../../../src/tabasco/configs/)):

- `model.compile=true`, `model.compile_mode=reduce-overhead`
- `trainer.precision=16` (fp16 AMP)
- `datamodule.batch_size=256`, `datamodule.num_workers=0`
- `model.num_random_augmentations=7` (8× batch expansion post-dataloader)

Current production timings (see
[`docs/research/tabasco_training_runs.md`](../../../docs/research/tabasco_training_runs.md)):
baseline 0.38 s/step @ 59% GPU util; CheMeleon 0.74 s/step @ 54%; MACE
cached 0.78 s/step @ 30%. The low MACE util is the loudest signal for
where gains exist.
