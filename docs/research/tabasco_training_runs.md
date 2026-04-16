# Tabasco Training Runs

WandB project: [`sr2173-university-of-cambridge/tabasco`](https://wandb.ai/sr2173-university-of-cambridge/tabasco)

## REPA Pipeline Configuration History

| Parameter | Runs below (pre-audit) | Current default (post-audit) | Reference paper |
|---|---|---|---|
| Projector layers | 2 | 3 | 3 |
| Averaging | per_atom (global) | per_sample | per_sample (mean_flat) |
| Similarity | cosine | cosine | cosine (normalize+dot) |
| Lambda | 0.5-0.8 | 0.5-0.8 | 0.5 |
| Combination | additive or tradeoff | additive or tradeoff | additive |

> **Note (2026-04-16 audit)**: All runs below were trained with `projector num_layers: 2` (default)
> and global per-atom averaging. The codebase has since been updated to default to
> `num_layers: 3` and `averaging: per_sample` to match the reference REPA paper.
> Future runs should use the new defaults. See [repa-codeflow.md](repa-codeflow.md) for audit details.

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
