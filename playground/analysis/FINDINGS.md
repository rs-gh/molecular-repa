# Projector Analysis: CheMeleon, MACE, GearNet

**Date**: 2026-04-02
**Script**: `playground/analysis/projector_analysis.py`
**Figures**: `playground/analysis/figures/`

## Question

When REPA training reports high cosine similarity between projected transformer hidden states and encoder targets, how much of that alignment is due to genuine structural learning vs. the projector simply learning atom/residue type prototypes?

## Method

Three tests per encoder, all with 80/20 stratified train/test splits:

1. **Mean-direction floor**: Cosine similarity between the mean embedding and all targets. This is the floor — any model that outputs a constant vector achieves this regardless of input. Computed from 200 molecules/proteins; bootstrap analysis confirms the estimate is stable (std < 0.005 even at 500 samples).
2. **Identity input** (test set): One-hot atom/residue type → 2-layer MLP (256 hidden) → encoder targets. Measures how much identity alone predicts the embedding.
3. **Random input** (test set): Random 128-d vectors → same MLP → targets. Sanity check — should approximate the mean direction since random inputs carry no generalizable information.

## Results

| | Floor | Rand (train) | Rand (test) | ID (train) | ID (test) | REPA val |
|--|---------|-------------|------------|-----------|----------|----------|
| **CheMeleon** | 0.388 | 0.727 | 0.200 | 0.459 | **0.455** | 0.66-0.68 |
| **MACE** | 0.755 | 0.881 | 0.666 | 0.863 | **0.861** | 0.56 |
| **GearNet** | 0.419 | 0.425 | 0.416 | 0.426 | **0.426** | 0.80 |

REPA val cos_sim sources: CheMeleon and MACE from final GEOM production runs (`docs/tabasco_training_runs.md`). GearNet from `proteina_60m_repa_layer4_v2` epoch metric.

## Key numbers

| | Identity (test) | REPA val | Gap |
|--|----------------|----------|-----|
| **GearNet** | 0.426 | **0.80** | **+0.374** |
| **CheMeleon** | 0.455 | **0.66-0.68** | **+0.21** |
| **MACE** | 0.861 | **0.56** | **-0.30** |

## Observations

**GearNet** has the largest positive gap (0.374). Identity barely predicts GearNet embeddings (0.426 ≈ floor 0.419), yet REPA training reaches 0.80. The transformer learns substantial structural information through REPA.

**CheMeleon** shows a moderate gap (0.21). Identity one-hot reaches 0.455, REPA val reaches 0.66-0.68. The transformer learns something beyond atom type, though identity accounts for roughly two-thirds of the alignment.

**MACE** shows a negative gap. REPA val (0.56) is lower than what identity one-hot achieves on a test set (0.861). The REPA alignment overfits on training molecules (train cos_sim = 0.90) but doesn't generalize. MACE's floor is already 0.755 — most embeddings point in a similar direction, and identity explains nearly all the remaining variation.

**Random input overfits on small datasets.** The random train-test gap is large for CheMeleon (0.73→0.20) and moderate for MACE (0.88→0.67), confirming that train-set-only evaluation inflates apparent alignment. GearNet shows no overfitting (0.43→0.42) due to its much larger sample count (45k vs 1.8k-5k).

## Questions

1. Why does MACE REPA val (0.56) fall below identity (0.86)? The projector achieves 0.90 on training data, so it's not a capacity issue — it's a generalization issue. What about the MACE target space makes per-molecule alignment hard to generalize?

2. CheMeleon's gap (0.21) is meaningful but modest. Would a simpler atom-type classification auxiliary loss achieve a similar benefit to generation quality, without the overhead of a frozen encoder?

3. GearNet's 0.374 gap is the largest. What structural features drive this — local geometry, contact patterns, secondary structure? Can we decompose this further?

4. The identity test uses the same MLP architecture across all encoders (256 hidden). Is this fair given the different output dimensions (2048 vs 192 vs 512)?

5. ~~The floor estimates are based on 200 molecules/proteins. Would these change substantially on the full training set?~~ Verified: bootstrap shows floor is stable to within +/-0.005 even at 500 samples. The 200-sample estimate is reliable.
