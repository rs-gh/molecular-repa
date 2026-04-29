# GearNet CA-only Encoder Characterization

**Date**: 2026-04-01
**Encoder**: GearNet CA-only (`NoTrainCAGearNet`, 8 layers, 512-dim)
**Data**: 200 proteins from PDB LMDB (45,503 residues total)
**Context**: REPA alignment target for protein flow matching model (Proteina 60M)

## Summary

GearNet is a strong REPA alignment target: dense representations (0% sparsity), genuinely 3D-aware, and protein-context-specific. The projector saturation test confirms the transformer IS contributing meaningful signal to REPA alignment (0.46 random baseline vs 0.78 in training). However, early training results show no clear validation loss improvement over the baseline — likely needs more training time or hyperparameter tuning.

## 1. Value Distribution & Sparsity

| Metric | GearNet | CheMeleon | MACE |
|--------|---------|-----------|------|
| Exact zeros | **0.00%** | 93.8% | 0.0% |
| Near-zero (<1e-6) | 0.00% | ~94% | 0.0% |
| Negative values | 62.5% | 0% (ReLU) | — |
| Mean | -1.07 | — | — |
| Std | 3.48 | — | — |

LeakyReLU (slope 0.1) produces a fully dense representation with ~62.5% negative values. No dead neurons, no gradient sparsity issues. This is a decisive advantage over CheMeleon's 93.8% zeros.

**Figure**: `fig_01_value_distribution.png`

## 2. Dimensionality & Singular Values

| Metric | GearNet | CheMeleon | MACE |
|--------|---------|-----------|------|
| Output dim | 512 | 2048 | 192 |
| Effective rank | **82.6** | 500 | 40.6 |
| Participation ratio | 40.8 | 58.5 | — |
| Dims for 90% variance | 82 | ~100 | 7 |
| Dims for 95% variance | 125 | — | — |
| Dims for 99% variance | 283 | ~300 | ~30 |

> **Definition note**: "Effective rank" (82.6) uses entropy on normalized squared singular values (variance): exp(-sum(p log p)) where p = S_i^2 / sum(S^2). This differs slightly from the MACE analysis which normalizes raw SVs. See `playground/projector/encoder_analysis.py` for all four definitions computed consistently.

Effective rank 82.6 out of 512 — the representation uses ~16% of its capacity. The projector (512 -> 512) has plenty of room to learn this structure. No bottleneck concern.

**Figure**: `fig_02_singular_values.png`

## 3. 3D Sensitivity

| Perturbation | Cosine Similarity |
|-------------|-------------------|
| 0.0A (original) | 1.000 |
| 0.1A Gaussian noise | **0.927** +/- 0.021 |
| 0.5A Gaussian noise | **0.362** +/- 0.051 |
| 1.0A Gaussian noise | 0.273 +/- 0.052 |
| 2.0A Gaussian noise | 0.181 +/- 0.031 |
| 5.0A Gaussian noise | -0.006 +/- 0.016 |
| Random rotation | **0.9997** +/- 0.0002 |

GearNet is **highly 3D-sensitive**: a 0.5A perturbation (sub-Angstrom, comparable to thermal fluctuations) drops cosine similarity to 0.36. At 5A, embeddings are essentially random. This is excellent for REPA — the encoder provides a rich 3D gradient signal.

Rotation invariance is near-perfect (0.9997), confirming the distance+angle edge features are rotationally invariant as designed.

Compare: CheMeleon had cos_sim = 1.000 for all conformers (zero 3D sensitivity).

**Figure**: `fig_03_3d_sensitivity.png`

## 4. Residue-Type Discrimination

| Metric | Value |
|--------|-------|
| Linear probe accuracy | **15.4%** (chance ~5%) |
| Mean cos sim between AA types | 0.975 |
| Within-type cos sim | 0.189 +/- 0.139 |
| Between-type cos sim | 0.176 +/- 0.133 |
| Delta (within - between) | 0.013 |

**GearNet embeddings are NOT dominated by amino acid identity.** The linear probe barely beats chance (15.4% vs 5%), and mean cosine similarity between different AA type centroids is 0.975 — nearly identical embeddings when averaged by type. Within-type and between-type pairwise similarities are almost identical (0.189 vs 0.176).

This means GearNet encodes primarily **structural context** (local geometry, neighbor distances) rather than sequence identity. For REPA, this is interesting: the alignment signal pushes the transformer to learn geometric features rather than just amino acid type.

Compare: Both CheMeleon and MACE achieve 100% linear probe accuracy for atom types, with MACE showing a larger within-type vs between-type gap (0.260 vs 0.093).

**Figure**: `fig_04_residue_discrimination.png`

## 5. Structural Context Sensitivity

| AA | Within-context cos sim | Between-context cos sim | Delta |
|----|----------------------|------------------------|-------|
| ALA | 0.292 | 0.180 | **0.113** |
| GLY | 0.179 | 0.173 | 0.006 |
| LEU | 0.235 | 0.185 | 0.049 |
| VAL | 0.197 | 0.173 | 0.024 |

Using CA-CA-CA bond angles as a secondary structure proxy (helix ~91, sheet ~120, loop = other):
- ALA shows the largest context effect (delta 0.113) — its small side chain means the backbone context dominates the embedding
- GLY shows almost no context effect (delta 0.006) — likely because GLY's flexibility places it in all contexts equally
- Overall, structural context has a **moderate** effect on embeddings

**Figure**: `fig_05_structural_context.png`

## 6. Embedding Norms & Conditioning

| Metric | Value |
|--------|-------|
| Mean L2 norm | 76.9 |
| Std L2 norm | 29.4 |
| Dead dimensions | **0 / 512** |
| Dimension std range | [2.53, 4.60] |

Well-conditioned representation:
- No dead dimensions (all 512 active)
- Narrow dimension std range (4.6/2.5 = 1.8x ratio) — no dominant or negligible dimensions
- Norms vary by AA type (CYS=88.7, VAL=87.0 highest; ASP=69.8 lowest) — disulfide and branched sidechains have larger norms

**Figure**: `fig_06_norms.png`

## 7. Projector Saturation Test (KEY RESULT)

**NOTE**: Original results (below) evaluated on training set only. Corrected results with proper train/test split are in `playground/analysis/FINDINGS.md`.

Original (train-set only, unreliable):

| Input Condition | Train Cosine Similarity |
|----------------|------------------------|
| Random 128-d | 0.461 |
| AA one-hot 21-d | 0.430 |
| One-hot + position 22-d | 0.428 |

Corrected (80/20 stratified split — see `playground/analysis/projector_analysis.py`):

| Input Condition | Train | Test |
|----------------|-------|------|
| Mean direction (no MLP) | — | **0.419** |
| Random 128-d | 0.425 | **0.416** |
| AA one-hot 20-d | 0.426 | **0.426** |
| **REPA training (transformer)** | — | **0.78** |

The corrected results show that both random and identity inputs converge to the mean-direction baseline (~0.42) on the test set — no overfitting, but also no information beyond the mean. The 0.354 gap between identity (0.426) and REPA training (0.78) represents genuine structural information learned by the transformer.

Notably, identity (one-hot residue type) adds almost nothing over the mean direction (0.426 vs 0.419), confirming GearNet embeddings are not dominated by residue identity.

**Figure**: `fig_07_projector_saturation.png` (original), `playground/analysis/figures/` (corrected)

## 8. Within-Protein vs Between-Protein Similarity

| Metric | Value |
|--------|-------|
| Within-protein cos sim | **0.420** +/- 0.271 |
| Between-protein cos sim | 0.176 +/- 0.128 |
| Delta | **0.244** |

GearNet embeddings are **protein-specific**: residues within the same protein are much more similar (0.42) than residues across different proteins (0.18). This suggests the encoder captures global protein context (overall fold, radius of gyration, packing density) in addition to local geometry.

**Figure**: `fig_08_protein_similarity.png`

## 9. GearNet Layer-wise Analysis

| Layer | Effective Rank | Mean Norm | Sparsity | Inter-layer Cos Sim |
|-------|---------------|-----------|----------|-------------------|
| 0 | 71.7 | 19.6 | 0.000 | 0.91 |
| 1 | 66.6 | 21.6 | 0.000 | 0.88 |
| 2 | 64.5 | 25.7 | 0.000 | 0.87 |
| 3 | 71.2 | 31.0 | 0.000 | 0.88 |
| 4 | 79.8 | 37.3 | 0.000 | 0.91 |
| 5 | 87.7 | 44.7 | 0.000 | 0.92 |
| 6 | 89.7 | 55.4 | 0.000 | 0.91 |
| 7 | 76.4 | 78.3 | 0.000 | 0.86 |

- **Effective rank peaks at layers 5-6** (~90), then drops at the final layer (76.4). The output layer compresses information.
- **Norms grow monotonically** (20 -> 78) — residual connections accumulate magnitude. This is fine for cosine similarity (normalized).
- **Inter-layer cosine similarity is 0.86-0.92** — each layer makes moderate but consistent changes. No layer is redundant.
- **Zero sparsity throughout** — LeakyReLU keeps all values alive at every layer.

**Figure**: `fig_09_layerwise.png`

## Head-to-Head Comparison: GearNet vs MACE vs CheMeleon

| Property | GearNet (proteins) | MACE (molecules) | CheMeleon (molecules) | Best for REPA |
|----------|-------------------|-------------------|----------------------|---------------|
| **Domain** | Proteins (CA-only) | Small molecules (all-atom) | Small molecules (2D) | — |
| **Output dim** | 512 | 192 | 2048 | MACE (compact) |
| **Exact zero %** | **0.00%** | **0.0%** | 93.8% | GearNet/MACE |
| **Effective rank** | 82.6 | 40.6 | 500 | MACE (most compact) |
| **Dims for 90% var** | 82 | 7 | ~100 | MACE |
| **3D-aware** | **Yes** (strong) | **Yes** (weak, cos=0.998) | No (cos=1.000) | GearNet |
| **3D sensitivity (0.5A)** | **cos=0.36** | cos=0.998 | cos=1.000 | GearNet (by far) |
| **Rotation invariant** | **Yes** (0.9997) | Yes | N/A | Both |
| **Identity discrimination** | 15.4% (AA type) | 100% (atom type) | 100% (atom type) | MACE/CheMeleon |
| **Structural discrimination** | Moderate (delta 0.01-0.11) | Strong (0.747 C envs) | N/A | MACE |
| **Projector saturation** | **No** (identity test 0.43 vs REPA 0.78) | Identity test 0.86 (dominates) | Identity test 0.46 ≈ REPA 0.47 | GearNet |
| **Within vs between sim** | 0.42 vs 0.18 (delta 0.24) | — | 0.25 vs 0.15 (delta 0.10) | GearNet |
| **Dead dimensions** | 0 / 512 | 0 / 192 | 133 / 2048 | GearNet/MACE |
| **Activation** | LeakyReLU (0.1) | None (invariant features) | ReLU | GearNet/MACE |

### Key Differences

**GearNet vs MACE**: GearNet is dramatically more 3D-sensitive (0.36 vs 0.998 at 0.5A perturbation) but less discriminative of residue/atom identity (15% vs 100%). GearNet encodes *where you are* in the protein; MACE encodes *what atom you are*. For REPA on a generative model that must learn both, this suggests MACE may complement GearNet.

**GearNet vs CheMeleon**: GearNet is superior in every dimension relevant to REPA: dense gradients, 3D-aware, no projector saturation, protein-specific embeddings. CheMeleon's only advantage was 100% atom-type discrimination, but at the cost of being completely 2D-blind.

**Projector saturation (corrected)**: With proper train/test splits (`playground/analysis/FINDINGS.md`), the picture is clearer. For CheMeleon, identity one-hot on the test set reaches 0.455 vs REPA training ~0.47 — a gap of only 0.015, meaning REPA mostly teaches atom types. For MACE, identity test reaches 0.861 — atom type dominates the embedding. For GearNet, identity test is only 0.426 vs REPA 0.78 — a 0.354 gap of genuine structural learning. GearNet is the only encoder where REPA provides substantial signal beyond identity.

## Implications for REPA Training

### Why cosine similarity plateaus at 0.78

The projector saturation test shows 0.46 is achievable with random inputs. The transformer adds ~0.32 on top. The plateau at 0.78 likely reflects GearNet's own representation structure — with 62.5% negative values, effective rank 82, and within-protein similarity of 0.42, perfect alignment (1.0) is not a meaningful target. 0.78 may be near the practical ceiling.

### Why all layers (L0, L4, L9) reach similar cosine similarity

Given that the projector contributes 0.46 baseline, and GearNet primarily encodes structural context (not AA identity), any transformer layer that has learned some geometric awareness can match the remaining ~0.32 gap. Even layer 0 (which has seen the noisy coordinates and pair distances) contains enough geometric signal for the projector to exploit.

### Why validation loss doesn't clearly improve over baseline

Several hypotheses:
1. **REPA regularization vs optimization**: The alignment signal may improve representation quality but compete with the flow matching objective for gradient bandwidth. With `lambda_repa=0.5`, half the gradient signal is alignment rather than generation.
2. **GearNet's limited discrimination**: 15% linear probe accuracy means the alignment target doesn't strongly encode residue identity — the transformer may not gain useful chemical information from alignment.
3. **Training duration**: REPA benefits typically emerge over longer training (the original paper used 400k+ steps). We're at ~20k steps.
4. **Batch size mismatch**: Baseline runs at B=6, REPA at B=4 due to encoder memory overhead. Different effective learning rates.

### Recommendations

1. **Continue training** — 20k steps is very early for a 60M parameter model on 580k proteins
2. **Try `lambda_repa=0.25`** — reduce REPA weight to give flow matching more gradient bandwidth
3. **Try multi-layer alignment** (`layers: [2, 4, 6]`) — spread the alignment signal across the network
4. **Add atom-type auxiliary loss** — since GearNet doesn't encode AA identity, a simple cross-entropy loss on predicting residue type from hidden states could complement REPA
5. **Profile step times** — check if GearNet forward pass is a bottleneck (B=4 vs B=6 means ~33% fewer samples per step)
