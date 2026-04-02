# Projector Analysis: CheMeleon, MACE, GearNet

**Date**: 2026-04-02 (corrected 2026-04-02)
**Canonical script**: `playground/projector/encoder_analysis.py`
**Legacy script**: `playground/projector/projector_analysis.py` (hidden_dim=256, superseded)
**Figures**: `playground/projector/figures/`

## Question

When REPA training reports high cosine similarity between projected transformer hidden states and encoder targets, how much of that alignment is due to genuine structural learning vs. the projector simply learning atom/residue type prototypes?

## Method

Three tests per encoder, all with 80/20 stratified train/test splits:

1. **Mean-direction floor**: Cosine similarity between the mean embedding and all targets. This is the floor — any model that outputs a constant vector achieves this regardless of input. Computed from 200 molecules/proteins; bootstrap analysis confirms the estimate is stable (std < 0.005 even at 500 samples).
2. **Identity input** (test set): One-hot atom/residue type → 2-layer MLP → encoder targets. Measures how much identity alone predicts the embedding.
3. **Random input** (test set): Random 128-d vectors → same MLP → targets. Sanity check — should approximate the mean direction since random inputs carry no generalizable information.

**Important**: The MLP hidden_dim now matches the actual projector used in training:
- CheMeleon/MACE: `hidden_dim=128` (from `model.net.hidden_dim` in tabasco configs)
- GearNet: `hidden_dim=512` (from `model.nn.token_dim` in proteina configs)

The previous analysis (`projector_analysis.py`) used `hidden_dim=256` for all encoders, which overestimated CheMeleon/MACE baselines and underestimated GearNet baselines.

## Results (hidden_dim=256, LEGACY — to be updated)

These results are from the OLD analysis with wrong hidden_dim. Run `encoder_analysis.py --phase analysis` to get corrected numbers.

| | Floor | Rand (train) | Rand (test) | ID (train) | ID (test) | REPA val | h_dim |
|--|---------|-------------|------------|-----------|----------|----------|-------|
| **CheMeleon** | 0.388 | 0.727 | 0.200 | 0.459 | **0.455** | 0.66-0.68 | 256 (wrong, actual=128) |
| **MACE** | 0.755 | 0.881 | 0.666 | 0.863 | **0.861** | 0.56 | 256 (wrong, actual=128) |
| **GearNet** | 0.419 | 0.425 | 0.416 | 0.426 | **0.426** | 0.80 | 256 (wrong, actual=512) |

REPA val cos_sim sources: CheMeleon and MACE from final GEOM production runs (`docs/tabasco_training_runs.md`). GearNet from `proteina_60m_repa_layer4_v2` epoch metric.

## Effective Rank (unified definitions)

Previous analyses used inconsistent definitions:
- `chemeleon/investigate.py`: threshold-based (SV > 1% max) → reported 500
- `mace/generate_figures.py`: entropy on normalized SVs → reported 40.6 (MACE), 138.1 (CheMeleon)
- `gearnet/explore_gearnet.py`: entropy on normalized *squared* SVs (variance) → reported 82.6

These are different metrics and should not be compared directly. The unified script (`encoder_analysis.py`) computes all four definitions for every encoder:

| Definition | CheMeleon | MACE | GearNet |
|-----------|-----------|------|---------|
| Threshold (SV > 1% max) | 500 | — | — |
| Entropy (norm SVs) | 138.1 | 40.6 | — |
| Entropy (norm variance) | — | — | 82.6 |
| PCA 90% variance | — | — | — |

*Table to be filled after running `encoder_analysis.py`. The dashes indicate values not yet computed under that definition for that encoder.*

## Key numbers

| | Identity (test) | REPA val | Gap |
|--|----------------|----------|-----|
| **GearNet** | 0.426 | **0.80** | **+0.374** |
| **CheMeleon** | 0.455 | **0.66-0.68** | **+0.21** |
| **MACE** | 0.861 | **0.56** | **-0.30** |

*Identity test values will change when rerun with correct hidden_dim.*

## Observations

**GearNet** has the largest positive gap (0.374). Identity barely predicts GearNet embeddings (0.426 ≈ floor 0.419), yet REPA training reaches 0.80. The transformer learns substantial structural information through REPA.

**CheMeleon** shows a moderate gap (0.21). Identity one-hot reaches 0.455, REPA val reaches 0.66-0.68. The transformer learns something beyond atom type, though identity accounts for roughly two-thirds of the alignment.

**MACE** shows a negative gap — but this comparison is confounded (see caveat below).

**Random input overfits on small datasets.** The random train-test gap is large for CheMeleon (0.73→0.20) and moderate for MACE (0.88→0.67), confirming that train-set-only evaluation inflates apparent alignment. GearNet shows no overfitting (0.43→0.42) due to its much larger sample count (45k vs 1.8k-5k).

## Caveat: MACE negative gap is confounded

The MACE "negative gap" (-0.30) compares two different things:
- **Identity baseline**: a standalone MLP trained on one-hot atom types → MACE embeddings of **clean molecules**, evaluated on held-out clean molecules
- **REPA val**: cosine similarity between projected **noisy** denoiser hidden states and clean molecule MACE embeddings, averaged across **all timesteps**

These are not equivalent measurements. At t≈0 the denoiser receives near-pure noise, so its hidden states carry almost no atom identity. Averaging across all t dilutes the signal. The identity baseline gets clean one-hot inputs every time.

The negative gap partly reflects the difficulty of the REPA task, not necessarily a failure of the approach. Evidence: MACE REPA achieves perfect validity (1.000 vs 0.980 baseline) and slightly better bond metrics in generation, despite the low cos_sim number.

To resolve this: see experiment 4A (timestep-stratified eval) which would reveal cos_sim at t≈1 specifically.

## Open questions

1. ~~The identity test uses the same MLP architecture across all encoders (256 hidden). Is this fair given the different output dimensions (2048 vs 192 vs 512)?~~ **Fixed**: `encoder_analysis.py` now uses per-encoder hidden_dim matching actual training configs.

2. CheMeleon's gap (0.21) is meaningful but modest. Would a simpler atom-type classification auxiliary loss achieve a similar benefit to generation quality, without the overhead of a frozen encoder?

3. GearNet's 0.374 gap is the largest. What structural features drive this — local geometry, contact patterns, secondary structure? Can we decompose this further?

4. Why does MACE REPA val (0.56) fall below identity (0.86)? The projector achieves 0.90 on training data, so it's not a capacity issue — it's a generalization issue. See caveat above for why this comparison is confounded.

5. ~~The floor estimates are based on 200 molecules/proteins. Would these change substantially on the full training set?~~ Verified: bootstrap shows floor is stable to within +/-0.005 even at 500 samples. The 200-sample estimate is reliable.
