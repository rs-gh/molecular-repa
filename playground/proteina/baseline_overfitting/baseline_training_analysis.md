# Baseline Training Performance Analysis

**Date:** 2026-04-14
**Model:** `proteina_60m_baseline_v2` (65M params, CAFlow, no REPA)
**Dataset:** PDB LMDB (579k CIF files, batch_size=6, max_length=512)
**Hardware:** Wilkes3, A100 80GB, torch.compile enabled

## Key Finding: Baseline Overfits After ~1-2 Epochs

The baseline model's generation quality **peaks early and then degrades** with
continued training. FID increased by ~25% between epochs 7 and 10, indicating
the model is overfitting to the training distribution at the expense of
generation diversity and quality.

## Training Loss Progression (from WandB)

| Step | Epoch | train/loss_step | Notes |
|------|-------|-----------------|-------|
| 7,579 | 0 | 1.4777 | Early training |
| 125,118 | 1 | 0.2754 | Rapid improvement |
| 475,172 | 6 | 0.3103 | Loss *increased* vs epoch 1 |

The training loss drops sharply in the first epoch, then **rises slightly**
from 0.275 to 0.310 over epochs 1-6. This is a classic sign of overfitting:
the model memorises training examples rather than learning generalisable
structure.

## FID Evaluation at Three Checkpoints

Full evaluation: 125 samples/length, 49 length bins (60-500, step 10), ~6,125
proteins per run.

### Quick Pass (~1,000 samples, step 266k, epoch ~3)

| Metric | Value |
|--------|-------|
| PDB FID | 547.9 |
| AFDB FID | 550.2 |
| PDB fJSD_C | 0.201 |
| PDB fJSD_A | 0.939 |
| PDB fJSD_T | 3.020 |
| fS_C | 2.50 |
| fS_A | 4.81 |
| fS_T | 22.33 |

*Note: Quick pass used 25 samples/len across 40 lengths (60-255), so these
numbers have higher variance.*

### Full Eval (~6,125 samples, step 535.5k, epoch 7)

| Metric | Value |
|--------|-------|
| PDB FID | 518.4 |
| AFDB FID | 532.9 |
| PDB fJSD_C | 0.218 |
| PDB fJSD_A | 0.830 |
| PDB fJSD_T | 2.660 |
| fS_C | 2.21 |
| fS_A | 4.72 |
| fS_T | 22.20 |

### Full Eval (~6,125 samples, step 742k, epoch 10)

| Metric | Value |
|--------|-------|
| PDB FID | 648.0 |
| AFDB FID | 622.0 |
| PDB fJSD_C | 1.288 |
| PDB fJSD_A | 1.567 |
| PDB fJSD_T | 3.794 |
| fS_C | 2.44 |
| fS_A | 4.23 |
| fS_T | 19.36 |

### Across All Three Checkpoints

| Metric | Epoch ~3 (266k) | Epoch 7 (535.5k) | Epoch 10 (742k) | Direction |
|--------|-----------------|------------------|-----------------|-----------|
| PDB FID | 547.9 | **518.4** | 648.0 | lower = better |
| AFDB FID | 550.2 | **532.9** | 622.0 | lower = better |
| PDB fJSD_C | **0.201** | 0.218 | 1.288 | lower = better |
| PDB fJSD_A | 0.939 | **0.830** | 1.567 | lower = better |
| PDB fJSD_T | **3.020** | 2.660 | 3.794 | lower = better |
| AFDB fJSD_C | **0.354** | 0.528 | 0.655 | lower = better |
| AFDB fJSD_A | **0.780** | 1.011 | 0.982 | lower = better |
| AFDB fJSD_T | **2.995** | 2.911 | 3.717 | lower = better |
| fS_C | **2.50** | 2.21 | 2.44 | higher = better |
| fS_A | **4.81** | 4.72 | 4.23 | higher = better |
| fS_T | 22.33 | **22.20** | 19.36 | higher = better |

*Bold = best across the three checkpoints. Epoch ~3 numbers are from the quick
pass (~1,000 samples) so have higher variance than the full eval numbers.*

### Epoch 7 vs Epoch 10 Comparison

| Metric | Epoch 7 | Epoch 10 | Delta | Direction |
|--------|---------|----------|-------|-----------|
| PDB FID | 518.4 | 648.0 | +129.6 | **worse** |
| AFDB FID | 532.9 | 622.0 | +89.1 | **worse** |
| PDB fJSD_C | 0.218 | 1.288 | +1.070 | **worse** |
| PDB fJSD_A | 0.830 | 1.567 | +0.737 | **worse** |
| PDB fJSD_T | 2.660 | 3.794 | +1.134 | **worse** |
| AFDB fJSD_C | 0.528 | 0.655 | +0.127 | **worse** |
| AFDB fJSD_A | 1.011 | 0.982 | -0.029 | marginal |
| AFDB fJSD_T | 2.911 | 3.717 | +0.806 | **worse** |
| fS_C | 2.21 | 2.44 | +0.23 | better |
| fS_A | 4.72 | 4.23 | -0.49 | **worse** |
| fS_T | 22.20 | 19.36 | -2.84 | **worse** |

Nearly every metric degrades between epoch 7 and 10. The only exception is
fS_C (fold score at class level), which marginally improves — but fold class
is the coarsest level (max 5 classes), so this is not meaningful.

## Interpretation

1. **Peak performance is around epoch 3-7.** The best FID (518) was at epoch 7,
   but the quick-pass result at epoch ~3 (548) was already close, suggesting
   the model reaches near-optimal generation quality very early.

2. **Continued training hurts generation quality.** The 25% FID increase from
   epoch 7 to 10 is substantial. The fJSD degradation is even more dramatic:
   PDB fJSD_C went from 0.218 to 1.288 (6x worse), meaning the model produces
   a much less diverse set of fold classes.

3. **Mode collapse is likely.** The combination of worsening FID and fJSD with
   stable/improving fS_C suggests the model is generating higher-quality
   samples from fewer fold classes — a hallmark of mode collapse.

4. **Training loss is a poor proxy for generation quality.** The loss only
   increased from 0.275 to 0.310 (12%) while FID degraded by 25% and fold
   diversity collapsed. Monitoring generation metrics during training would
   catch this earlier.

## Implications for REPA Experiments

- **REPA models were evaluated at epoch 3 (420k steps).** Given the baseline
  peaks around epoch 3-7, the REPA comparison at 420k steps may be the most
  fair comparison point. The baseline's superiority at 535.5k steps may partly
  reflect being closer to the optimal early-stopping point.

- **Does REPA regularise against overfitting?** The REPA models are currently
  training past epoch 3. If they maintain or improve FID where the baseline
  degraded, this would suggest the alignment loss acts as a regulariser — a
  valuable property independent of absolute FID.

- **Early stopping is critical.** Any future training runs should include
  periodic FID evaluation (e.g., every 50-100k steps) to identify the optimal
  checkpoint rather than training to convergence.

## Checkpoint Locations

| Checkpoint | Steps | Epoch | Path |
|-----------|-------|-------|------|
| Epoch 7 (best) | 535,500 | 7 | Overwritten by epoch 10 (only `last-EMA.ckpt` retained) |
| Epoch 10 (latest) | 742,000 | 10 | `/rds/user/sr2173/hpc-work/proteina/store/proteina_60m_baseline_v2/checkpoints/last-EMA.ckpt` |

**Note:** The epoch 7 checkpoint was not separately saved — only `last-EMA.ckpt`
is retained, which is now at epoch 10. The `checkpoint_every_n_steps: 10000`
config creates periodic checkpoints, but only the most recent `ignore.ckpt` and
`last.ckpt` are kept. If the epoch 7 model is needed, it would require
retraining with early stopping.

## Raw CSV Paths

- Epoch 7 results (backed up): `eval_output/inference_fid_60m_baseline/results_inference_fid_60m_baseline_fid_535k.csv.bak`
- Epoch 10 results: `eval_output/inference_fid_60m_baseline/results_inference_fid_60m_baseline_fid.csv`
