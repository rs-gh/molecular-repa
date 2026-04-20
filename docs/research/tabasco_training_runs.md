# Tabasco Training Runs

WandB project: [`sr2173-university-of-cambridge/tabasco`](https://wandb.ai/sr2173-university-of-cambridge/tabasco)

## REPA Pipeline Configuration History

| Parameter | Runs below (pre-audit) | Current default (post-audit) | Reference paper |
|---|---|---|---|
| Projector layers | 2 | 3 | 3 |
| Averaging | per_atom (global) | **per_atom** (project choice) | per-patch mean_flat (≡ per_sample at fixed patch count; no guidance for variable length) |
| Similarity | cosine | cosine | cosine (normalize+dot) |
| Lambda | 0.5-0.8 | 0.5-0.8 | 0.5 |
| Combination | additive or tradeoff | additive or tradeoff | additive |

> **Note (2026-04-16 audit, updated 2026-04-17)**: All runs below were trained with
> `projector num_layers: 2` (default) and global per-atom averaging. The projector-layers
> default was updated to 3 to match the reference REPA paper; the averaging default was
> reverted to `per_atom` on 2026-04-17 to preserve continuity with existing checkpoints.
> `averaging: per_sample` remains available via explicit override — note neither option is
> "the paper default" since the paper averages per-patch at fixed patch count per image, so
> per-sample and per-patch coincide there and the paper gives no variable-length guidance.
> See [repa-codeflow.md](repa-codeflow.md) for audit details.

## GEOM Dataset — Production Runs

All production runs trained on GEOM-drugs (1,142,099 molecules, batch size 256, 4461 steps/epoch).
Checkpoints on RDS: `/rds/user/sr2173/hpc-work/tabasco/outputs/`
Stripped checkpoints (15MB): `evaluation/checkpoints/tabasco/geom/`

### Run Index

| Model | WandB Run ID(s) | WandB URL | Epochs | Steps | Phase | Checkpoint Path (RDS) |
|---|---|---|---|---|---|---|
| **Baseline (no REPA)** | `s105bkm0`, `yy363ps7` | [part1](https://wandb.ai/sr2173-university-of-cambridge/tabasco/runs/s105bkm0), [part2](https://wandb.ai/sr2173-university-of-cambridge/tabasco/runs/yy363ps7) | 33 | 151,899 | baseline | `geom_mild/checkpoints/last.ckpt` |
| **CheMeleon additive (same proj)** | `0fbrr8vx` | [link](https://wandb.ai/sr2173-university-of-cambridge/tabasco/runs/0fbrr8vx) | 15 | 73,264 | chemeleon | `geom_chemprop_additive/checkpoints/last.ckpt` |
| **CheMeleon additive (fused proj)** | `x3c4vid0` | [link](https://wandb.ai/sr2173-university-of-cambridge/tabasco/runs/x3c4vid0) | 16 | 77,099 | chemeleon | `geom_chemprop_additive_v2/checkpoints/last.ckpt` |
| **CheMeleon tradeoff (same proj)** | `cqjant8r` | [link](https://wandb.ai/sr2173-university-of-cambridge/tabasco/runs/cqjant8r) | 15 | 73,264 | chemeleon | `geom_chemprop_tradeoff/checkpoints/last.ckpt` |
| **CheMeleon tradeoff (fused proj)** | `7u3l0zpy` | [link](https://wandb.ai/sr2173-university-of-cambridge/tabasco/runs/7u3l0zpy) | 16 | 77,843 | chemeleon | `geom_chemprop_tradeoff_v2/checkpoints/last.ckpt` |
| **MACE additive** | `7kuaxjk4`, `1cj5gk44` | [GPU live](https://wandb.ai/sr2173-university-of-cambridge/tabasco/runs/7kuaxjk4), [cached](https://wandb.ai/sr2173-university-of-cambridge/tabasco/runs/1cj5gk44) | 15 | 73,249 | mace-gpu + mace-cached | `geom_mace_cached_additive_v2/checkpoints/last.ckpt` |
| **MACE tradeoff** | `uq02ccie`, `5s25bbx3` | [GPU live](https://wandb.ai/sr2173-university-of-cambridge/tabasco/runs/uq02ccie), [cached](https://wandb.ai/sr2173-university-of-cambridge/tabasco/runs/5s25bbx3) | 15 | 73,249 | mace-gpu + mace-cached | `geom_mace_cached_tradeoff_v2/checkpoints/last.ckpt` |

**Multi-part runs**: Baseline was split across two WandB runs (epochs 0-7 then 8-32). MACE runs had an initial GPU-live phase (epochs 0-3, encoder on GPU) then switched to cached embeddings (epochs 3-15) for speed.

### REPA Run Configurations

| Model | Encoder | Encoder Dim | Lambda | Combination | Projector Layers | Projector Hidden | Averaging |
|---|---|---|---|---|---|---|---|
| CheMeleon additive (same) | ChemPropEncoder | 2048 | 0.5 | additive | 2 | hidden_dim | per_atom |
| CheMeleon additive (fused) | ChemPropEncoder | 2048 | 0.5 | additive | 2 | hidden_dim | per_atom |
| CheMeleon tradeoff (same) | ChemPropEncoder | 2048 | 0.5 | tradeoff | 2 | hidden_dim | per_atom |
| CheMeleon tradeoff (fused) | ChemPropEncoder | 2048 | 0.5 | tradeoff | 2 | hidden_dim | per_atom |
| MACE additive | MACEEncoder (small) | 192 | 0.8 | additive | 2 | hidden_dim | per_atom |
| MACE tradeoff | MACEEncoder (small) | 192 | 0.8 | tradeoff | 2 | hidden_dim | per_atom |

**Notes**:
- "same proj" = single projector matching coord hidden dim; "fused proj" = projector sized for concatenated coord+atom heads (cross_attention=True)
- `hidden_dim` refers to `model.net.hidden_dim` (128 for GEOM mild config)
- All runs used `time_weighting: false` and `similarity_type: cosine`
- CheMeleon is 2D-only (same embeddings for all conformers); MACE is 3D-aware

### WandB Display Names (for API queries)

These are the display names used in `collect_training_perf.py` for matching runs via the WandB API:

| Display Name | Label |
|---|---|
| `final-tabasco-mild-geom-part2` | Baseline (no REPA) |
| `0224-1849-tabasco-geom-chemprop-additive-fused-projector` | CheMeleon additive |
| `0224-1848-tabasco-geom-chemprop-tradeoff-fused-projector` | CheMeleon tradeoff |
| `0320-1409-tabasco-geom-mace-additive` | MACE add (CPU, f64) — early run |
| `0320-1409-tabasco-geom-mace-tradeoff` | MACE trade (CPU, f64) — early run |
| `0321-0006-tabasco-geom-mace-additive-1` | MACE add (GPU, f32) |
| `0321-0010-tabasco-geom-mace-tradeoff-1` | MACE trade (GPU, f32) |
| `0321-1401-tabasco-geom-mace-cached-additive-3` | MACE cached add |
| `0321-1401-tabasco-geom-mace-cached-tradeoff-3` | MACE cached trade |

### Evaluation Results (1000 generated molecules, 100 Euler steps)

| Model | Epoch | Validity | Connectivity | Novelty | PB Bond Lengths | PB Bond Angles | PB Steric Clash | PB Intersection | FCD |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 32 | 0.980 | 0.998 | 0.966 | 0.974 | 0.961 | 0.933 | 0.917 | 5.61 |
| CheMeleon add (same) | 15 | 0.967 | 1.000 | 0.950 | 0.947 | 0.940 | 0.900 | 0.868 | 5.83 |
| CheMeleon add (fused) | 15 | 0.972 | 0.999 | 0.961 | 0.959 | 0.952 | 0.920 | 0.900 | 7.43 |
| CheMeleon trade (same) | 15 | 0.976 | 0.999 | 0.964 | 0.958 | 0.954 | 0.916 | 0.896 | 6.49 |
| CheMeleon trade (fused) | 16 | 0.960 | 0.999 | 0.942 | 0.939 | 0.928 | 0.885 | 0.850 | 6.24 |
| MACE additive | 14 | 1.000 | 0.995 | 0.977 | 0.977 | 0.973 | 0.921 | — | 6.81 |
| MACE tradeoff | 14 | 1.000 | 0.999 | 0.982 | 0.982 | 0.972 | 0.941 | — | 6.32 |

### Training Performance

| Model | s/step | Steps/hr | Runtime (hr) | GPU Util (mean) |
|---|---|---|---|---|
| Baseline | 0.376 | 9,567 | 15.88 | 59.3% |
| CheMeleon additive | 0.737 | 4,884 | 15.78 | 56.7% |
| CheMeleon tradeoff | 0.735 | 4,900 | 15.89 | 51.5% |
| MACE cached add | 0.780 | 4,618 | 15.86 | 30.1% |
| MACE cached trade | 0.779 | 4,619 | 15.86 | 30.5% |

## Performance engineering audit (2026-04-19)

Benchmarks run via [hpc-scripts/tabasco/bench/](../../hpc-scripts/tabasco/bench/) on A100 80GB (Wilkes3 ampere, `gpu-q-18`). Each variant in a spawn subprocess; steady-state `steps/sec` after dropping 30 warmup steps. Input shapes match GEOM drugs (N=71, BS=256 unless noted, fake data so the model is isolated from I/O).

### torch.compile: 1.4× speedup, no meaningful memory savings

[evaluation/tabasco/bench/results/compile.csv](../../evaluation/tabasco/bench/results/compile.csv) — baseline (no REPA), BS=256, N=71, fp16 vs bf16:

| compile | precision | s/step | steps/s | peak GB | speedup |
|---|---|---:|---:|---:|---:|
| off             | 16         | 0.268 | 3.74  | 13.5 | 1.00× |
| off             | bf16-mixed | 0.271 | 3.69  | 13.5 | 0.99× |
| default         | 16         | 0.191 | 5.23  | 13.3 | 1.40× |
| default         | bf16-mixed | 0.194 | 5.16  | 13.3 | 1.38× |
| **reduce-overhead** | **16**     | **0.189** | **5.31** | **13.2** | **1.42×** |
| reduce-overhead | bf16-mixed | 0.192 | 5.22  | 13.2 | 1.40× |

- Current prod (`compile_mode=reduce-overhead`, `precision=16`) is optimal by a razor margin. `default` mode is within 1.4%.
- **fp16 narrowly beats bf16** on this workload (~1.7%) — reverse of proteina. Keep fp16.
- Compile saves **~0.3 GB** vs proteina's 5–14 GB. Tabasco's N=71 leaves little activation memory for inductor to elide.

### SDPA: already on EFFICIENT, no kernel unlock available

[evaluation/tabasco/bench/results/sdpa.csv](../../evaluation/tabasco/bench/results/sdpa.csv). Forced each backend via `torch.nn.attention.sdpa_kernel([...])`, bf16-mixed:

| backend | compile | s/step | peak GB | status |
|---|---|---:|---:|---|
| default (auto-dispatch) | off     | 0.271 | 13.5 | OK — resolves to EFFICIENT |
| efficient               | off     | 0.271 | 13.5 | OK (identical to default) |
| **flash**               | off     | —     | —    | **unsupported** |
| math                    | off     | 0.441 | 22.4 | OK (fallback disaster) |
| default                 | default | 0.194 | 13.3 | OK |
| efficient               | default | 0.195 | 13.3 | OK |
| flash                   | default | —     | —    | unsupported |
| math                    | default | 0.255 | 19.1 | OK |

Tabasco's attention path: [Attention](../../src/tabasco/src/tabasco/models/components/attention.py#L33) wraps `nn.MultiheadAttention(batch_first=True)`, which lowers to `F.scaled_dot_product_attention` internally when `need_weights=False` + bool `key_padding_mask`. Both the `reimplemented` and `pytorch` transformer implementations end up at the same MHA kernel.

- **Auto-dispatch already picks EFFICIENT_ATTENTION**. Identical timings to forcing efficient directly. No `.contiguous()`-style unlock available (unlike proteina, which was stuck on MATH).
- **FLASH is hard-blocked by torch 2.9.1** via *"Flash Attention does not support non-null attn_mask"* ([diagnose_sdpa.log](../../evaluation/tabasco/bench/results/diagnose_sdpa.log)). MHA lowers `key_padding_mask` to an additive float mask that FA's kernel rejects.
- **MATH is the disaster fallback**: 1.6× slower, 1.7× more memory (22.4 vs 13.5 GB). Worth knowing the blast radius if EFFICIENT ever gets runtime-disabled.

### Kernel-level diagnose: MHA wrapper adds ~45% attention overhead

[diagnose_sdpa.log](../../evaluation/tabasco/bench/results/diagnose_sdpa.log) — per-call microbenchmark (A100 bf16, B=256, N=71, H=8, D=16):

| path | mask | FLASH | EFFICIENT | MATH | CUDNN |
|---|---|---:|---:|---:|---:|
| MHA wrapper | w/ padding | blocked | 1.21 ms | 2.92 ms | 1.16 ms |
| MHA wrapper | no mask    | 1.00 ms | 1.12 ms | 2.80 ms | 1.05 ms |
| bare SDPA   | additive   | blocked | 0.67 ms | 2.42 ms | 0.71 ms |
| bare SDPA   | no mask    | 0.42 ms | 0.58 ms | 2.30 ms | 0.48 ms |

- MHA costs ~0.54 ms/layer extra vs bare SDPA (projections + mask conversion + reshape). Over 16 layers × `8×` augmented batch, that's ~8.6 ms — **~3% of step time**. Not worth rewriting `Attention` to bypass MHA.
- **CUDNN_ATTENTION is slightly faster than EFFICIENT** at these shapes (1.16 vs 1.21 ms MHA; 0.71 vs 0.67 ms bare) and supports the attn_mask, but "cuDNN attention has been runtime disabled" in this torch build. Forcing `sdpa_kernel([CUDNN])` is a cheap follow-up to test in production.
- To unlock FLASH we'd need to restructure the forward to run unmasked and zero out padding post-hoc. At N=71 FLASH is only ~20-30% faster than EFFICIENT per kernel call — **not worth the correctness risk**.

### Batch-size ceiling: BS ≤ 1456 fits; throughput is flat from BS=256

[evaluation/tabasco/bench/results/batch_size_sweep.csv](../../evaluation/tabasco/bench/results/batch_size_sweep.csv) — baseline, `reduce-overhead`, fp16, binary-searched in [256, 2048]:

| BS | peak GB | steps/s | samples/s | % of 80 GB |
|---:|---:|---:|---:|---:|
| **256 (prod)** | 13.2 | **5.31** | **1,358** | 16% |
| 1152 | 59.0 | 1.12 | 1,290 | 74% |
| 1376 | 70.4 | 0.95 | 1,307 | 88% |
| **1432** | 73.2 | 0.96 | **1,375** | 92% |
| 1446 | 74.0 | 0.91 | 1,316 | 93% |
| 1456 (max) | 74.5 | 0.91 | 1,325 | 93% |
| 1460+ | OOM | — | — | — |

#### What BS actually means in tabasco

BS=256 refers to the **dataloader** batch size — the number of unique molecules drawn from LMDB per step. Two multipliers sit between that and what actually flows through the transformer:

1. **Rotation augmentation** ([flow_model.yaml `num_random_augmentations: 7`](../../src/tabasco/configs/model/flow_model.yaml#L18)) → [`apply_random_rotation`](../../src/tabasco/src/tabasco/data/transforms.py#L49) runs in-GPU post-dataloader and expands each batch **8×** (7 augmented copies + original). So **2,048 rows of shape `(N=71, 3)` hit the transformer per step** at production BS=256.
2. **DDP replicas** (where used) — [hpc-scripts/tabasco/geom/multi-gpu/](../../hpc-scripts/tabasco/geom/multi-gpu/) scripts exist (2×A100, `trainer=ddp`, lr scaled to 0.004) but **none of the runs in the production run index above used them** — all current GEOM entries are single-GPU, dataloader-BS=256 / model-BS=2048.

So the "13.2 GB at BS=256" peak memory is for **2,048 augmented rows**, not 256. The 1,456 ceiling means the transformer can handle **~11,648 augmented rows per step** before OOM.

All GEOM production configs ([mild.yaml](../../src/tabasco/configs/experiment/geom/mild.yaml#L18), [chemprop_*.yaml](../../src/tabasco/configs/experiment/geom/chemprop_tradeoff.yaml), [mace_*.yaml](../../src/tabasco/configs/experiment/geom/mace_cached_tradeoff.yaml)) inherit BS=256 from `geom/mild`. QM9 baseline also uses BS=256. Only `local_baseline` variants drop to 32 for dev.

#### Throughput interpretation

- Max dataloader BS before OOM: **1456** (74.5 GB). Production BS=256 leaves **~61 GB on the table**.
- But **raw samples/s is essentially flat** from BS=256 to BS=1432 (1,358 → 1,375, +1.2%). The GPU is already compute-bound at BS=256 — bigger batches just fit more work per step while each sample takes proportionally longer.
- The **59% GPU utilization** in the production table isn't compute headroom you can fill with bigger batches; it's kernel-launch overhead between steps, which compile partially hides.
- Scaling dataloader BS to 1432 spends 5× the memory for essentially zero throughput gain. The *real* use for this headroom is matching a larger effective optimizer batch without gradient accumulation, or per-GPU batch in future multi-GPU runs — not wall-clock speedup in the single-GPU case.

### Decisions and non-changes

Confirmed by measurement — no code changes land in this PR:

| Lever | Current | Decision | Rationale |
|---|---|---|---|
| `compile_mode` | `reduce-overhead` | **keep** | 1.42× win; 1.4% edge over `default` |
| `trainer.precision` | `16` (fp16 AMP) | **keep** | fp16 edges bf16 by ~1.7% on this workload |
| SDPA backend | auto (EFFICIENT) | **keep** | already on the fused kernel |
| Attention wrapper | `nn.MultiheadAttention` | **keep** | bare-SDPA rewrite yields ~3% step time; not worth refactor risk |
| `datamodule.batch_size` | 256 (→ 2048 via 8× rotation augmentation) | **keep** | no throughput gain from raising BS on single A100 |
| `num_workers` | 0 | **keep** | per [feedback_num_workers.md](../../../.claude/projects/-home-sr2173-git-molecular-repa/memory/feedback_num_workers.md); dataloader is not the bottleneck |

Follow-ups worth a small run in the future:
- Force `sdpa_kernel([CUDNN])` — microbench suggests a 4-5% attention speedup if it works end-to-end
- Profile the augmentation expansion in [`FlowMatchingModel.forward`](../../src/tabasco/src/tabasco/models/flow_model.py#L144) (`apply_random_rotation` with `n_augmentations=7` → 8× batch). MACE cached at 30% GPU util suggests encoder-side stalls; hasn't been characterized yet.
- REPA variants (CheMeleon, MACE cached) weren't BS-swept — the CachedMACEEncoder needs real LMDB to init. Repeat the probe on a compute node with the cache LMDB mounted when those encoders change.

## GEOM Dataset — Development/Early Runs

These were intermediate runs during development, before the final production configuration.

| Description | WandB Run ID | WandB URL | Notes |
|---|---|---|---|
| MACE CPU f64 additive | (by display name) | — | Crashed at epoch 0 (649 steps). 50s/step, 1.8% GPU util. |
| MACE CPU f64 tradeoff | (by display name) | — | Crashed at epoch 0 (699 steps). 50s/step, 0.8% GPU util. |
| MACE GPU f32 additive | (by display name) | — | Crashed at epoch 3. 2.67s/step, 23.8% GPU util. |
| MACE GPU f32 tradeoff | (by display name) | — | Crashed at epoch 3. 2.63s/step, 22% GPU util. |

## Dev Cluster Runs (2026-01-31)

Early proof-of-concept runs on the dev GPU cluster, before HPC training.

| Description | WandB Run ID | WandB URL |
|---|---|---|
| ChemProp (dev) | `tg4a8h91` | [link](https://wandb.ai/sr2173-university-of-cambridge/tabasco/runs/tg4a8h91) |
| Baseline (dev) | `oc3eb4x4` | [link](https://wandb.ai/sr2173-university-of-cambridge/tabasco/runs/oc3eb4x4) |

## Related Scripts

- **Pull validation curves**: `evaluation/scripts/tabasco/geom/compile_wandb_curves.py`
- **Collect training perf**: `evaluation/scripts/tabasco/geom/collect_training_perf.py`
- **Strip checkpoints**: `evaluation/scripts/strip_checkpoint.py`
- **Compile evaluation results**: `evaluation/scripts/compile_results.py`
- **Evaluate checkpoint**: `evaluation/scripts/evaluate.py`
