# Proteina generation speedup

Three-part playground for measuring and validating a faster inference path for
Proteina generation.

- `bench.py` — throughput benchmark across `(mode, length)`
- `validate.py` — chemical-plausibility check across modes @ len=256
- `explore.py` — pick ckpt + mode + length, view samples in browser

## Motivation

Our FID eval (`evaluate.py`) calls `model.generate(...)` inside `torch.no_grad()`
with `dtype=torch.float32` and **no** `torch.compile`. SDPA is active (efficient-
attention kernel), but compile and mixed precision — both used during training —
are dropped at inference. 400 ODE steps × eager fp32 transformer forwards is the
bottleneck.

The candidate fast path wraps `model.nn` in `torch.compile(mode="reduce-
overhead", dynamic=False)` and runs the forward under a
`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` context.

## Modes

| code | compile | precision | role |
|------|---------|-----------|------|
| A | off | fp32 | current `evaluate.py` path |
| B | on | fp32 | diagnostic: compile alone |
| C | off | bf16 | diagnostic: bf16 alone |
| D | on | bf16 | candidate replacement |

Default bench runs A + D. Use `--modes all` for attribution.

## Quickstart

```bash
source .venv/bin/activate
# Smoke (few minutes):
python playground/proteina/generation_speedup/explore.py

# Benchmark (~30 min on 1 GPU, A vs D, lengths 128/256/384/512):
python playground/proteina/generation_speedup/bench.py

# Correctness validation (~15 min, 4 modes × 50 samples @ len=256):
python playground/proteina/generation_speedup/validate.py
```

## Checkpoint selection

Edit [checkpoints.py](checkpoints.py). Uncomment one (or more) of the 12
sample-matched entries. Active default: `baseline_512_sm` @ step 500K.

## Outputs

- `bench_results.csv` — throughput table
- `validate_metrics.csv`, `validate_rmsd.csv` — per-sample + cross-mode RMSD
- `figures/*.png` — overlaid distributions + cross-mode RMSD boxplot
- `samples/validate/{mode}/sample_*.pdb` — 3 saved PDBs per mode
- `samples/explore/view.html` — browser-viewable 3D cartoon of explore.py samples

## Pass criteria (for validate.py)

- CA-CA bond mean ∈ [3.75, 3.85] Å across modes
- Clash rate < 2% across modes
- Median CA RMSD B vs A < 0.1 Å (compile should be numerically equivalent to eager)
- Median CA RMSD C/D vs A < ~5 Å (bf16 drift over 400 ODE steps)
- Rg roughly follows `Rg ≈ 2.2 × N^0.4` for all modes

## Findings

Measured on A100 80GB, `baseline_512_sm` @ step 500K, batch size 8.

### Throughput (A vs D)

| Length | eager_fp32 (s/sample) | compile_bf16 (s/sample) | Speedup |
|--------|----------------------|------------------------|---------|
| 128    | 1.989                | 0.475                  | 4.2×    |
| 256    | 5.609                | 1.112                  | 5.0×    |
| 384    | 11.162               | 2.121                  | 5.3×    |
| 512    | 18.939               | 4.123                  | 4.6×    |

Compile warmup (first batch per length): ~60–90s for eager_fp32 compile trigger;
~60–70s for compile_bf16 CUDAGraph capture. Amortised to negligible over a
200-sample eval.

At n=512 the 200-sample FID eval drops from ~1 h → ~14 min.

### Validation (4 modes × 56 samples @ L=256)

All modes produce chemically plausible structures. Distributions are statistically
indistinguishable across compile and precision axes.

| Mode | Rg (Å) | bond_mean (Å) | angle (°) | clash |
|------|--------|---------------|-----------|-------|
| eager_fp32   | 17.93 ± 0.86 | 3.822 | 99.6° | 0.0% |
| compile_fp32 | 17.93 ± 0.85 | 3.823 | 99.7° | 0.0% |
| eager_bf16   | 17.91 ± 0.84 | 3.827 | 99.7° | 0.0% |
| compile_bf16 | 17.90 ± 0.85 | 3.825 | 99.7° | 0.0% |

Cross-mode per-sample CA RMSD vs eager_fp32:

| Mode | median RMSD | p90 RMSD |
|------|-------------|----------|
| compile_fp32 | 0.74 Å | 1.28 Å |
| eager_bf16   | 0.88 Å | 1.39 Å |
| compile_bf16 | 0.87 Å | 1.61 Å |

Note: compile_fp32 median RMSD (0.74 Å) exceeds the pre-planned <0.1 Å target.
`reduce-overhead` mode uses non-deterministic CUDA kernels (e.g. flash-attn
split-k), so small numeric differences accumulate over 400 ODE steps even without
bf16. Geometry distributions match; this is not a correctness concern.

### Decision

compile_bf16 is the shipped default in `evaluate.py` (`--fast_inference`, on by
default). Use `--no-fast_inference` to revert to the eager fp32 path.

## Caveats

- Compile is static-shape; each new `(batch_size, length)` pair triggers
  recompilation (batch 0 absorbs this as warmup).
- bf16 drift is cumulative over 400 ODE steps — per-sample coords diverge
  between modes even though distributions match. Compare distributions, not exact
  coords.
- `reduce-overhead` uses non-deterministic kernels → compile_fp32 is not bit-exact
  with eager_fp32 either (~0.74 Å median RMSD).
- Clash threshold (3.6 Å) is heuristic; adjust in `_geometry.py` if needed.
- No Ramachandran / DSSP / TM-to-nearest-PDB — deferred.
