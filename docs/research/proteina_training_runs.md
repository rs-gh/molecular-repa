# Proteina Training Runs

WandB project: [`sr2173-university-of-cambridge/proteina-repa`](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa)

## Dataset

PDB LMDB (579,823 CIF files). Train/val/test split: 98%/1.9%/0.1%.
Sequence similarity threshold: 0.5. Resolution: 0.0-5.0 A. Min length: 50, max length: 512.
Protein-only, standard residues, no ligands. Experiment types: diffraction, EM.

## Model Architecture

CAFlow, 60M parameters:
- Transformer: 10 layers, 512 token dim, 8 heads, 256 pair dim
- No triangle multiplicative layers
- EMA: decay=0.999
- torch.compile: enabled
- Optimizer: Adam, lr=0.0001

## REPA Configuration

All REPA runs use:
- **Encoder**: GearNet CA-only (frozen, 512-dim output, `gearnet_ca.pth`)
- **Similarity**: cosine
- **Combination mode**: additive (FM + lambda * REPA)
- **Lambda**: 0.5
- **Projector**: 512 hidden dim, **3-layer MLP** (Linear->SiLU->Linear->SiLU->Linear)
- **Averaging**: per_residue (global) for all runs below. `per_sample` option added 2026-04-16 (matches paper).

> **Note (2026-04-16 audit)**: All runs below were trained with `projector_num_layers: 2` and
> `averaging: per_residue` (global). The codebase has since been updated to default to
> `projector_num_layers: 3` and `averaging: per_sample` to match the reference REPA paper.
> Future runs should use the new defaults. See [repa-codeflow.md](repa-codeflow.md) for audit details.

## Run Index

Checkpoints on RDS: `/rds/user/sr2173/hpc-work/proteina/store/`

| Model | WandB Run Name | REPA Layer | Batch Size | Epochs | Steps | Checkpoint Path (RDS) |
|---|---|---|---|---|---|---|
| **Baseline** | `proteina_60m_baseline_v2` | -- | 6 | 10 | 742,000 | `proteina_60m_baseline_v2/checkpoints/last-EMA.ckpt` |
| **REPA L0** | `proteina_60m_repa_layer0_v2` | [0] | 4 | 7 | 836,500 | `proteina_60m_repa_layer0_v2/checkpoints/last-EMA.ckpt` |
| **REPA L4** | `proteina_60m_repa_v2` | [4] | 4 | 7 | 840,000 | `proteina_60m_repa_v2/checkpoints/last-EMA.ckpt` |
| **REPA L9** | `proteina_60m_repa_layer9_v2` | [9] | 4 | 7 | 847,000 | `proteina_60m_repa_layer9_v2/checkpoints/last-EMA.ckpt` |

**Batch size difference**: Baseline uses batch 6; REPA uses batch 4 because the frozen GearNet encoder adds ~10GB GPU memory overhead on A100 80GB.

**Periodic checkpoints**: Stored every 10,000 steps as `step={step:012d}-EMA.ckpt`.

## Evaluation Results (6,125 samples, 100 Euler steps, SDE sc_scale_noise=0.45)

### Latest Checkpoints

| Model | Step | Epoch | PDB FID | PDB fJSD_C | PDB fJSD_A | PDB fJSD_T | AFDB FID | fS_C | fS_A |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 742k | 10 | 648.0 | 1.288 | 1.567 | 3.794 | 622.0 | 2.44 | 4.23 |
| REPA L0 | 420k | 3 | 599.1 | 0.842 | 1.394 | 3.149 | 611.1 | 1.84 | 3.06 |
| REPA L4 | 420k | 3 | 657.0 | 1.075 | 1.570 | 3.445 | 659.7 | 1.78 | 3.38 |
| REPA L9 | -- | -- | 879.2 | 0.094 | 1.152 | 3.951 | 887.1 | 2.11 | 3.38 |

### At ~840k Steps (Epoch 7)

| Model | Step | PDB FID | PDB fJSD_C | PDB fJSD_A | PDB fJSD_T | AFDB FID | fS_C | fS_A |
|---|---|---|---|---|---|---|---|---|
| **Baseline (best, epoch 7)** | 535k | **518.4** | **0.218** | **0.830** | **2.660** | **532.9** | **2.21** | **4.72** |
| REPA L0 | 836k | **401.8** | **0.533** | **1.035** | **2.536** | **393.8** | 1.93 | 4.26 |
| REPA L4 | 840k | 580.1 | 0.896 | 1.362 | 2.934 | 569.8 | 1.82 | 3.48 |
| REPA L9 | 847k | 614.2 | 0.237 | 0.988 | 2.791 | 677.0 | 2.15 | 4.93 |

**Key observations**:
- Baseline peaks at epoch 7 (535k steps) then degrades (mode collapse, see below)
- REPA L0 at 840k achieves best PDB FID (401.8) and AFDB FID (393.8) overall
- REPA L4 at 840k beats the baseline's final checkpoint but not its epoch 7 peak
- REPA L9 has worst FID but best fJSD_C (0.094 at latest; 0.237 at 840k) — interesting pattern
- All runs used `projector_num_layers: 2` and `averaging: per_residue` (pre-audit defaults)

## Baseline Overfitting Analysis

Detailed in: `playground/proteina/baseline_overfitting/baseline_training_analysis.md`

- Model peaks at **epoch 7 (535k steps)** with PDB FID = 518.4
- Degrades at epoch 10 (742k steps): PDB FID = 648.0 (+25%)
- Fold diversity (fJSD_C) worsens 6x: 0.218 -> 1.288
- Training loss poorly correlates with generation quality
- Epoch 7 checkpoint was overwritten — only periodic 10k-step checkpoints remain

## GearNet Encoder Characterization

Detailed in: `playground/proteina/gearnet/FINDINGS.md`

- CA-only, 8 layers, 512-dim output
- 0% sparsity (LeakyReLU)
- Effective rank: 82.6/512
- Strong 3D sensitivity (cos=0.36 at 0.5A perturbation)
- Near-perfect rotation invariance (0.9997)
- Weak AA-identity encoding (15% linear probe accuracy) — encodes structural context, not sequence
- Projector saturates at 0.78 cosine similarity (vs 0.46 random baseline)

## Validation Loss Breakdown

Script: `playground/proteina/val_loss_breakdown/backcalc_repa_val_loss.py`
Data: `playground/proteina/val_loss_breakdown/data/`
Figure: `playground/proteina/val_loss_breakdown/figures/val_loss_breakdown.png`

## REPA Pipeline Configuration History

| Parameter | Runs above (pre-audit) | Current default (post-audit) | Reference paper |
|---|---|---|---|
| Projector layers | 2 | 3 | 3 |
| Averaging | per_residue (global) | per_sample | per_sample (mean_flat) |
| Similarity | cosine | cosine | cosine (normalize+dot) |
| Combination | additive | additive | additive |
| Lambda | 0.5 | 0.5 | 0.5 |

## Hardware

- Partition: Wilkes3 ampere (CSD3)
- GPU: A100 80GB (1 per node)
- CPUs: 24 per task
- Time limit: 36 hours
- LMDB copied to local NVMe (`/tmp/proteina_pdb_lmdb`)
- Account: LIO-CHARM-SL2-GPU

## Training Configs

| Config | Description |
|---|---|
| `src/proteina/configs/experiment_config/training_baseline.yaml` | Baseline (no REPA) |
| `src/proteina/configs/experiment_config/training_repa.yaml` | REPA L4 (full dataset) |
| `src/proteina/configs/experiment_config/training_repa_layer0.yaml` | REPA L0 (full dataset) |
| `src/proteina/configs/experiment_config/training_repa_layer9.yaml` | REPA L9 (full dataset) |
| `src/proteina/configs/experiment_config/training_repa_l0_256.yaml` | REPA L0 (256 max len) |
| `src/proteina/configs/experiment_config/training_repa_l4_256.yaml` | REPA L4 (256 max len) |
| `src/proteina/configs/experiment_config/training_repa_l9_256.yaml` | REPA L9 (256 max len) |

## Evaluation Configs

| Config | Description |
|---|---|
| `src/proteina/configs/experiment_config/inference_fid_60m_baseline.yaml` | Baseline FID eval |
| `src/proteina/configs/experiment_config/inference_fid_60m_repa.yaml` | REPA L4 FID eval |
| `src/proteina/configs/experiment_config/inference_fid_60m_repa_layer0.yaml` | REPA L0 FID eval |
| `src/proteina/configs/experiment_config/inference_fid_60m_repa_layer9.yaml` | REPA L9 FID eval |
| Lite versions: `*_lite.yaml` | For convergence curves (300 samples, faster) |

## HPC Scripts

| Script | Description |
|---|---|
| `hpc-scripts/proteina/train_baseline.sh` | Train baseline model |
| `hpc-scripts/proteina/train_repa.sh` | Train REPA models (configurable layer) |
| `hpc-scripts/proteina/eval_fid.sh` | Full FID evaluation (6,125 samples) |
| `hpc-scripts/proteina/eval_fid_lite_sweep.sh` | Convergence curves across checkpoints |

## Related Analysis

| Location | Description |
|---|---|
| `playground/proteina/baseline_overfitting/` | Baseline overfitting analysis |
| `playground/proteina/gearnet/` | GearNet encoder characterization |
| `playground/proteina/val_loss_breakdown/` | Validation loss decomposition |
| `docs/research/repa-codeflow.md` | REPA pipeline code flow + audit findings |
