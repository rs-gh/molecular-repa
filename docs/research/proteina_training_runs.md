# Proteina Training Runs

WandB project: [`sr2173-university-of-cambridge/proteina-repa`](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa)

## Training-config layout

```
configs/experiment_config/training/<seq_len>/[<encoder>/[<averaging>/]]<config>.yaml
```

- **Baselines** live at `training/<seq_len>/training_baseline*.yaml` — no encoder subdir (they don't use REPA).
- **REPA configs** live at `training/<seq_len>/<encoder>/[<averaging>/]<config>.yaml`, where `<encoder>` ∈ `{gearnet, esm2, ...}`.
- The encoder also appears in-config at `repa.encoder.type` — single source of truth; the factory in [`proteina_repa.py::_build_encoder`](../../src/proteina/proteinfoundation/repa/proteina_repa.py) matches on it. Legacy flat-schema configs (`repa.gearnet_ckpt_path` at top of `repa:`) are still accepted via backward-compat.

**Adding a new encoder**: drop a config under `training/<seq_len>/<new_encoder>/<averaging>/...yaml`, set `repa.encoder.type: <new_encoder>`, and add a branch in `_build_encoder` that returns the encoder instance. WandB will auto-group the run by `encoder_<type>` tag/group ([`train_repa.py`](../../src/proteina/proteinfoundation/train_repa.py) derives this from `cfg_exp.repa.encoder.type`).

**SLURM**:
```bash
# REPA (gearnet)
sbatch hpc-scripts/proteina/training/train_repa.sh training_repa_l4_256_per_residue training/256/gearnet/per_residue
# REPA (ESM-2)
sbatch hpc-scripts/proteina/training/train_repa.sh training_repa_l9_256_per_residue training/256/esm2/per_residue
# Baseline (no encoder subdir)
sbatch hpc-scripts/proteina/training/train_baseline.sh training_baseline_256 training/256
```

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
- **Averaging**: per_residue (global) — project default. `per_sample` (each protein weighted equally regardless of length) available via explicit config override. The REPA paper averages per-patch; at fixed patches-per-image that equals per_sample, so it gives no guidance on variable-length inputs.

> **Note (2026-04-16 audit, updated 2026-04-17)**: All runs below were trained with `projector_num_layers: 2` and
> `averaging: per_residue` (global). The projector-layers default was updated to 3 to match the reference
> REPA paper; the averaging default was reverted to `per_residue` on 2026-04-17 to preserve continuity with
> existing v2 checkpoints. `averaging: per_sample` remains available via explicit override — note neither
> can claim to be "the paper default" since the paper averages per-patch at fixed patch count per image, so
> per-sample and per-patch coincide there and the paper gives no variable-length guidance.
> See [repa-codeflow.md](repa-codeflow.md) for audit details.

## Production Runs (v2, full PDB, max_len=512)

Checkpoints on RDS: `/rds/user/sr2173/hpc-work/proteina/store/`

| Model | WandB Run ID | WandB URL | REPA Layer | Batch Size | Epochs | Steps | Checkpoint Path (RDS) |
|---|---|---|---|---|---|---|---|
| **Baseline** | `proteina_60m_baseline_v2` | [link](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa/runs/proteina_60m_baseline_v2) | -- | 6 | 10 | 742,000 | `proteina_60m_baseline_v2/checkpoints/last-EMA.ckpt` |
| **REPA L0** | `proteina_60m_repa_layer0_v2` | [link](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa/runs/proteina_60m_repa_layer0_v2) | [0] | 4 | 7 | 836,500 | `proteina_60m_repa_layer0_v2/checkpoints/last-EMA.ckpt` |
| **REPA L4** | `proteina_60m_repa_v2` | [link](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa/runs/proteina_60m_repa_v2) | [4] | 4 | 7 | 840,000 | `proteina_60m_repa_v2/checkpoints/last-EMA.ckpt` |
| **REPA L9** | `proteina_60m_repa_layer9_v2` | [link](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa/runs/proteina_60m_repa_layer9_v2) | [9] | 4 | 7 | 847,000 | `proteina_60m_repa_layer9_v2/checkpoints/last-EMA.ckpt` |

**Batch size difference**: Baseline uses batch 6; REPA uses batch 4 because the frozen GearNet encoder adds ~10GB GPU memory overhead on A100 80GB.

**Periodic checkpoints**: Stored every 10,000 steps as `step={step:012d}-EMA.ckpt`.

**Auto-resume**: Proteina uses `run_name_` as the WandB run ID. Multiple SLURM jobs append to the same WandB run (multiple local `wandb/` directories map to one run).

## Development Runs (v1, max_len=512)

Early runs before production configuration was finalized.

| Model | WandB Run ID | WandB URL | Notes |
|---|---|---|---|
| Baseline (v1) | `proteina_60m_baseline` | [link](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa/runs/proteina_60m_baseline) | Pre-v2 config |
| REPA (v1) | `proteina_60m_repa` | [link](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa/runs/proteina_60m_repa) | Pre-v2 config |

## Short-Sequence Runs (max_len=256)

Smaller runs for faster iteration (256 max residues instead of 512).

Config layout: `src/proteina/configs/experiment_config/training/256/gearnet/{per_residue,per_sample}/training_repa_l{0,4,9}_256_{per_residue,per_sample}.yaml`.

ESM-2 REPA variant (added 2026-04-18): `.../training/256/esm2/per_residue/training_repa_l9_256_per_residue.yaml`.

### Active runs (post-2026-04-17 rename)

| Model | WandB Run ID | REPA Layer | Averaging | Notes |
|---|---|---|---|---|
| Baseline 256 | [`proteina_60m_baseline_256`](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa/runs/proteina_60m_baseline_256) | -- | -- | Shorter proteins |
| REPA L0 256 (per_residue) | [`proteina_60m_repa_l0_256_per_residue`](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa/runs/proteina_60m_repa_l0_256_per_residue) | [0] | per_residue | Resumed from `_perres` ckpt on rename |
| REPA L4 256 (per_residue) | [`proteina_60m_repa_l4_256_per_residue`](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa/runs/proteina_60m_repa_l4_256_per_residue) | [4] | per_residue | Resumed from `l4_256` ckpt on rename |
| REPA L9 256 (per_residue) | [`proteina_60m_repa_l9_256_per_residue`](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa/runs/proteina_60m_repa_l9_256_per_residue) | [9] | per_residue | Resumed from `_perres` ckpt on rename |
| REPA L0 256 (per_sample) | [`proteina_60m_repa_l0_256_per_sample`](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa/runs/proteina_60m_repa_l0_256_per_sample) | [0] | per_sample | Fresh — never ran prior (run_name collision bug) |
| REPA L4 256 (per_sample) | [`proteina_60m_repa_l4_256_per_sample`](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa/runs/proteina_60m_repa_l4_256_per_sample) | [4] | per_sample | Resumed from `l4_256_persamp` ckpt on rename |
| REPA L9 256 (per_sample) | [`proteina_60m_repa_l9_256_per_sample`](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa/runs/proteina_60m_repa_l9_256_per_sample) | [9] | per_sample | Fresh — never ran prior (run_name collision bug) |

### 2026-04-17 rename history

Configs and checkpoint store dirs were reorganized for clarity: `per_residue/` and `per_sample/` subdirs, fully-qualified run_names (`_per_residue`/`_per_sample` suffixes).

| Old wandb run | Old store dir | New wandb run | New store dir | Notes |
|---|---|---|---|---|
| `proteina_60m_repa_l0_256_perres` | `proteina_60m_repa_l0_256_perres` | `proteina_60m_repa_l0_256_per_residue` | `proteina_60m_repa_l0_256_per_residue` | Canonical per_residue l0 (live) |
| `proteina_60m_repa_l4_256` | `proteina_60m_repa_l4_256` | `proteina_60m_repa_l4_256_per_residue` | `proteina_60m_repa_l4_256_per_residue` | Canonical per_residue l4 (live) |
| `proteina_60m_repa_l9_256_perres` | `proteina_60m_repa_l9_256_perres` | `proteina_60m_repa_l9_256_per_residue` | `proteina_60m_repa_l9_256_per_residue` | Canonical per_residue l9 (live) |
| — | `proteina_60m_repa_l4_256_persamp` | `proteina_60m_repa_l4_256_per_sample` | `proteina_60m_repa_l4_256_per_sample` | Canonical per_sample l4 (live) |
| `proteina_60m_repa_l0_256` | `proteina_60m_repa_l0_256` | *(archived)* | `proteina_60m_repa_l0_256_DEPRECATED_20260417` | Earlier duplicate per_residue l0, superseded by `_perres` |
| `proteina_60m_repa_l9_256` | `proteina_60m_repa_l9_256` | *(archived)* | `proteina_60m_repa_l9_256_DEPRECATED_20260417` | Earlier duplicate per_residue l9, superseded by `_perres` |

Wandb runs for renamed trainings start fresh on relaunch (wandb run ID = `run_name_`; renaming `run_name_` forks a new wandb run). Checkpoint continuity is preserved via the physical store-dir rename. `train_repa.py` was updated to use `resume="allow"` so relaunches after rename don't crash on missing wandb history.

Prior collision bug: `training_repa_l{0,9}_256_persamp.yaml` had `run_name_` pointing at the per_residue run name. Those configs are now removed, and l0/l9 per_sample runs start fresh from scratch under their new canonical names.

## Short-short-sequence runs (max_len=128)

Added 2026-04-17 for faster iteration than the 256 tier — ~4× lower attention cost. Intended for overnight convergence experiments rather than multi-day production runs.

Config layout: `src/proteina/configs/experiment_config/training/128/{training_baseline_128.yaml, gearnet/per_residue/training_repa_l{0,4,9}_128_per_residue.yaml}`. Dataset: `pdb_lmdb_128` (batch_size 24 placeholder, `lmdb_max_num_residues: 128`, `PaddingTransform max_size: 128`, `dataselector.max_length: 128`).

| Model | WandB Run ID (= run_name) | REPA Layer | Averaging | Projector depth |
|---|---|---|---|---|
| Baseline 128 | `proteina_60m_baseline_128` | -- | -- | -- |
| REPA L0 128 (per_residue) | `proteina_60m_repa_l0_128_per_residue` | [0] | per_residue | **3** |
| REPA L4 128 (per_residue) | `proteina_60m_repa_l4_128_per_residue` | [4] | per_residue | **3** |
| REPA L9 128 (per_residue) | `proteina_60m_repa_l9_128_per_residue` | [9] | per_residue | **3** |

**Noteworthy — projector depth diverges from 256/512**. The 128 REPA configs are the first in the repo to use `projector_num_layers: 3` (reference REPA paper depth). The 256/512 configs all still use 2 with a TODO. A direct comparison between e.g. `l4_128_per_residue` and `l4_256_per_residue` therefore is **not** a clean length-only ablation; it also varies projector depth. Keep this in mind when reading convergence curves across tiers.

**Batch size**: 24 is a placeholder based on L² attention scaling from 256 (B=12). First launch may OOM or leave headroom — adjust `pdb_lmdb_128.yaml` if needed.

**per_sample variants**: not shipped initially. Add via the 256 pattern (`training/128/gearnet/per_sample/training_repa_l{0,4,9}_128_per_sample.yaml`) if needed.

**Submission:**
```bash
sbatch hpc-scripts/proteina/training/train_baseline.sh training_baseline_128 training/128
sbatch hpc-scripts/proteina/training/train_repa.sh training_repa_l4_128_per_residue training/128/gearnet/per_residue
# (same pattern for l0, l9)
```

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

## Lite Eval Convergence Sweep (300 samples, 100 Euler steps)

Figures: `evaluation/proteina/generation/figures/fid_convergence.png`, `fjsd_convergence.png`, `feature_scores_convergence.png`
Data: `evaluation/proteina/generation/results/pdb/fid/lite_convergence_all.csv`
Scripts: `evaluation/proteina/generation/scripts/collect_lite_results.py`, `plot_fid_convergence.py`

### Baseline

| Step | Samples Seen | PDB FID | fJSD_C | fJSD_A | AFDB FID |
|-----:|-------------:|--------:|-------:|-------:|---------:|
| 10k | 60k | 78,491 | 1.196 | 5.335 | 78,811 |
| 20k | 120k | 180,900 | 1.150 | 4.096 | 181,171 |
| 40k | 240k | 37,432 | 0.506 | 2.769 | 37,495 |
| 80k | 480k | 15,679 | 0.358 | 2.438 | 15,689 |
| 150k | 900k | 2,961 | 0.091 | 1.630 | 2,937 |
| 250k | 1.5M | 850 | 0.284 | 1.244 | 848 |
| 350k | 2.1M | 719 | 0.308 | 0.999 | 734 |
| **450k** | **2.7M** | **488** | 0.691 | 0.937 | **489** |
| 550k | 3.3M | 922 | 1.812 | 2.427 | 825 |
| 650k | 3.9M | 766 | 1.751 | 2.187 | 690 |
| 740k | 4.4M | 576 | 0.911 | 1.206 | 561 |

### REPA L4

| Step | Samples Seen | PDB FID | fJSD_C | fJSD_A | AFDB FID |
|-----:|-------------:|--------:|-------:|-------:|---------:|
| 10k | 40k | 85,230 | 1.529 | 4.922 | 85,387 |
| 20k | 80k | 213,772 | 1.023 | 4.081 | 214,071 |
| 40k | 160k | 32,021 | 0.482 | 2.708 | 32,082 |
| 80k | 320k | 2,107 | 0.042 | 1.749 | 2,104 |
| 150k | 600k | 863 | 1.387 | 2.242 | 885 |
| 250k | 1.0M | 739 | 1.306 | 1.695 | 755 |
| 350k | 1.4M | 724 | 1.041 | 1.701 | 723 |
| **450k** | **1.8M** | **689** | 1.013 | 1.465 | **682** |
| 550k | 2.2M | 692 | 0.391 | 0.947 | 681 |
| 650k | 2.6M | 767 | 0.556 | 1.113 | 760 |
| 750k | 3.0M | 746 | 1.185 | 1.646 | 728 |
| 840k | 3.4M | 635 | 0.908 | 1.401 | 630 |

### REPA L0

| Step | Samples Seen | PDB FID | fJSD_C | fJSD_A | AFDB FID |
|-----:|-------------:|--------:|-------:|-------:|---------:|
| 10k | 40k | 120,725 | 1.406 | 4.917 | 120,949 |
| 20k | 80k | 170,108 | 0.975 | 3.904 | 170,357 |
| 40k | 160k | 28,917 | 0.360 | 2.622 | 28,976 |
| 80k | 320k | 2,227 | 0.026 | 1.669 | 2,241 |
| 150k | 600k | 936 | 0.343 | 1.442 | 973 |
| 250k | 1.0M | 680 | 0.042 | 0.459 | 710 |
| 350k | 1.4M | 610 | 0.731 | 1.179 | 629 |
| 450k | 1.8M | 523 | 0.911 | 1.121 | 517 |
| 550k | 2.2M | 440 | 0.454 | 0.915 | 427 |
| 650k | 2.6M | 457 | 0.369 | 0.941 | 448 |
| **750k** | **3.0M** | **431** | 0.118 | 0.783 | **429** |
| 830k | 3.3M | 436 | 0.394 | 0.931 | 429 |

### REPA L9

| Step | Samples Seen | PDB FID | fJSD_C | fJSD_A | AFDB FID |
|-----:|-------------:|--------:|-------:|-------:|---------:|
| 10k | 40k | 96,896 | 1.528 | 4.954 | 97,090 |
| 20k | 80k | 205,376 | 1.066 | 4.095 | 205,664 |
| 40k | 160k | 41,078 | 0.502 | 2.814 | 41,161 |
| 80k | 320k | 2,247 | 0.068 | 1.759 | 2,238 |
| 150k | 600k | 964 | 0.328 | 1.615 | 976 |
| 250k | 1.0M | 2,390 | 0.357 | 2.157 | 2,354 |
| 350k | 1.4M | 986 | 0.042 | 1.295 | 980 |
| 450k | 1.8M | 837 | 0.186 | 1.143 | 838 |
| 550k | 2.2M | 827 | 0.137 | 0.963 | 875 |
| 650k | 2.6M | 837 | 0.772 | 1.390 | 873 |
| 750k | 3.0M | 771 | 0.305 | 1.327 | 825 |
| **840k** | **3.4M** | **673** | 0.214 | 1.070 | 726 |

**Sweep retry (job 27950470)**: initial sweep (27918706) had 3 failures — step 20k TIMEOUT on a slow node, steps 550k/650k hit transient `CUDA device busy` on gpu-q-43. Resubmitted on fresh nodes, all 3 completed successfully. 20k finished in 10 min on retry, confirming the original timeout was node-specific rather than workload-specific.

### Key takeaways from the convergence sweep

- **L0 is the strongest REPA variant by a wide margin**: PDB FID plateaus at ~430 after 550k steps, beating both L4 (plateaus ~690) and the baseline's best (488 at 450k).
- **L4 plateaus early**: little improvement from 450k onward (689 → 635).
- **L9 is the weakest** of the three REPA variants on FID, but has the best fJSD_C (0.042 at 350k) — aligning high-level semantics trades off against geometric fidelity.
- All three REPA runs avoid the mid-training collapse seen in the baseline between 450k–650k (FID spikes 488 → 922).

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

| Parameter | Runs above (pre-audit) | Current project default | Reference paper |
|---|---|---|---|
| Projector layers | 2 | 3 | 3 |
| Averaging | per_residue (global) | **per_residue** (project choice) | per-patch mean_flat (≡ per_sample at fixed patch count; no guidance for variable length) |
| Similarity | cosine | cosine | cosine (normalize+dot) |
| Combination | additive | additive | additive |
| Lambda | 0.5 | 0.5 | 0.5 |

Note: the project default was flipped to `per_residue` on 2026-04-17 to match the averaging used to train the existing v2 checkpoints. `per_sample` (each structure weighted equally) remains available via explicit `averaging: per_sample` in the config. Neither option is "the paper default": the paper's `mean_flat` is per-patch, which coincides with per_sample only when every image has the same number of patches — in variable-length domains the distinction is a project choice.

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
| `src/proteina/configs/experiment_config/training_baseline.yaml` | Baseline (no REPA) — 512 max len |
| `src/proteina/configs/experiment_config/training/512/gearnet/training_repa.yaml` | REPA L4 GearNet (512 max len) |
| `src/proteina/configs/experiment_config/training/512/gearnet/training_repa_layer0.yaml` | REPA L0 GearNet (512 max len) |
| `src/proteina/configs/experiment_config/training/512/gearnet/training_repa_layer9.yaml` | REPA L9 GearNet (512 max len) |
| `src/proteina/configs/experiment_config/training/256/gearnet/per_residue/training_repa_l{0,4,9}_256_per_residue.yaml` | REPA L{0,4,9} GearNet (256 max len), per_residue averaging |
| `src/proteina/configs/experiment_config/training/256/gearnet/per_sample/training_repa_l{0,4,9}_256_per_sample.yaml` | REPA L{0,4,9} GearNet (256 max len), per_sample averaging |
| `src/proteina/configs/experiment_config/training/256/esm2/per_residue/training_repa_l9_256_per_residue.yaml` | REPA L9 ESM-2 (256 max len), per_residue |
| `src/proteina/configs/experiment_config/training/128/gearnet/per_residue/training_repa_l{0,4,9}_128_per_residue.yaml` | REPA L{0,4,9} GearNet (128 max len), per_residue |
| `src/proteina/configs/experiment_config/training/{128,256}/training_baseline_{128,256}.yaml` | Baselines at 128 / 256 max len |
| Legacy flat paths (`training_repa.yaml`, `training_repa_l{0,4,9}_256.yaml`, `training_repa_layer{0,9}.yaml`) | Symlinks kept for backwards-compat; resolve into the tree above |

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
| `hpc-scripts/proteina/training/train_baseline.sh` | Train baseline model |
| `hpc-scripts/proteina/training/train_repa.sh` | Train REPA models (configurable layer) |
| `hpc-scripts/proteina/evaluation/eval_fid.sh` | Full FID evaluation (6,125 samples) |
| `hpc-scripts/proteina/evaluation/eval_fid_lite_sweep.sh` | Convergence curves across checkpoints |

## Training Performance (2026-04-17)

Benchmarks run via `hpc-scripts/proteina/bench/` on A100 80GB (Wilkes3, `gpu-q-22`). Each variant in an isolated subprocess; steady-state `steps/sec` after dropping 30 warmup steps.

### torch.compile: 2.6× speedup, also saves GPU memory

[evaluation/proteina/bench/results/compile.csv](../../evaluation/proteina/bench/results/compile.csv) — baseline (no REPA), bf16-mixed:

| seq_len | compile | batch | steps/s | peak GB | speedup |
|---:|---|---:|---:|---:|---:|
| 256 | off     | 8 | 4.14  | 21.8 | 1.00× |
| 256 | default | 8 | **10.81** | **16.7** | **2.61×** |
| 512 | off     | 6 | 2.34  | 59.1 | 1.00× |
| 512 | default | 6 | **6.04**  | **44.7** | **2.58×** |

Compile saves **5 GB at seq=256 / 14 GB at seq=512** — activation memory drops as inductor fuses ops and elides intermediates. That memory headroom is why BS=6 at seq=512 stays safe with compile on (BS=8 still OOMs, the config comment in `pdb_lmdb.yaml` stands). `mode=reduce-overhead` (CUDA graphs) errors with "graph recording observed an input tensor deallocate during graph recording" on both seq lengths — known incompatibility with the proteina forward; not worth pursuing without a refactor.

### SDPA: calling the frontend, getting the slow kernel

[evaluation/proteina/bench/results/sdpa.csv](../../evaluation/proteina/bench/results/sdpa.csv). Forced each backend via `torch.nn.attention.sdpa_kernel([...])`:

| backend | compile=off steps/s | compile=default steps/s | peak GB | status |
|---|---:|---:|---:|---|
| `default` (auto-dispatch) | 2.34 | 6.05 | 59.1 | OK |
| `math` | 2.34 | 6.02 | 59.1 | OK (identical to default) |
| `efficient` | — | — | — | unsupported |
| `flash` | — | — | — | unsupported |

Proteina's forward calls `F.scaled_dot_product_attention` (`use_sdpa=True` in config; [pair_bias_attn.py:126](../../src/proteina/proteinfoundation/nn/pair_bias_attn/pair_bias_attn.py#L126)), **but the auto-dispatcher picks `MATH`, not a fused kernel**. FLASH_ATTENTION and EFFICIENT_ATTENTION both reject the input — likely because of the `(B, H, N, N)` additive float bias. FlashAttention-2 only supports per-head ALiBi-style bias; torch's EFFICIENT *should* accept arbitrary bias but something about the proteina bias (non-contiguous strided view from `rearrange`, `requires_grad=True`, or dtype under autocast) makes it fail. A targeted diagnostic (`hpc-scripts/proteina/bench/diagnose_sdpa.py`) is pending to pin down the exact cause.

**Net:** `use_sdpa=True` currently gives no kernel speedup over the manual `_attn` einsum path; the 2.6× win comes entirely from torch.compile. Unlocking EFFICIENT_ATTENTION (if the bias-path issue turns out to be cosmetic — e.g., needs `.contiguous()`) would stack on top.

### Lustre vs local NVMe (pending)

Job 27960354 was re-submitted with a Lustre fallback after the first run hit `/tmp` full on `gpu-q-22` (12 GB free vs 50 GB needed). Results TBD.

**Takeaway from the `/tmp` failure itself:** the LMDB-to-NVMe copy logic in [train_baseline.sh](../../hpc-scripts/proteina/training/train_baseline.sh) is load-bearing — nodes routinely lack ~50 GB of `/tmp` headroom. Its Lustre-fallback branch catches this; without it, training would have died at startup. The benchmark scripts' initial version didn't have that fallback — fixed in [benchmark_io.py](../../hpc-scripts/proteina/bench/benchmark_io.py) / [benchmark_e2e.py](../../hpc-scripts/proteina/bench/benchmark_e2e.py).

### Other findings from GPU monitor

The 2026-04-01 production runs sustained **93-96% GPU utilization** over 36 h. With compile on plus the I/O stack we have today, there's not much headroom left at the occupancy layer — future gains will come from throughput (larger effective BS via grad accumulation, unlocking a fused attention kernel) rather than "fill the GPU more".

## Related Analysis

| Location | Description |
|---|---|
| `playground/proteina/baseline_overfitting/` | Baseline overfitting analysis |
| `playground/proteina/gearnet/` | GearNet encoder characterization |
| `playground/proteina/val_loss_breakdown/` | Validation loss decomposition |
| `docs/research/repa-codeflow.md` | REPA pipeline code flow + audit findings |
| `hpc-scripts/proteina/bench/` | Performance benchmarks (compile, I/O, E2E, SDPA) |
