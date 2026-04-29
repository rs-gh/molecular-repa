# GearNet CA-only Encoder Characterization

**Date**: 2026-04-29
**Encoder**: CA-GearNet (`GearNetPerResidueEncoder`, 8 layers, 512-dim)
**Data**: 200 PDB train proteins (42,034 residues)
**SLURM**: 28596016 (A100, full sweep). Latest results: [results/20260429_135327/results.json](results/20260429_135327/results.json), [layerwise.csv](results/20260429_135327/layerwise.csv).
**Cross-encoder context**: [../FINDINGS.md](../FINDINGS.md) — for the three-question framing (Q1 information, Q2 saturation, Q3 conditioning) used throughout.

## Summary

CA-GearNet is the strongest *3D-aware* REPA target in our shortlist. The representation is dense (Q3.1 OK), genuinely sensitive to sub-Angstrom geometry (Q1.2 strong), and protein-specific (Q1.4 Δ = 0.222). Eff rank 77.5/512 vs the random-init floor of 3.3 confirms pretraining adds substantial structure (Q3.2). The catch is at Q2: `mean_direction_cos = 0.425` and the best (`onehot+pos`) projector input only reaches 0.432 — a gap of just **+0.006**, much smaller than ESM2's +0.053. CA-GearNet is "usable" but the empirical REPA headroom may be narrow, and any wins should be expected in geometric realism (within-protein structure) rather than identity-conditioned generation.

## Q1. What information does the encoder encode?

### 1.1 Residue identity

| Metric | Value |
|--------|-------|
| Linear probe accuracy | 13.7% (chance ~5%) |
| Mean cos between AA centroids | 0.976 |
| Within-type cos | 0.190 ± 0.136 |
| Between-type cos | 0.184 ± 0.136 |
| Δ (within − between) | **0.006** |
| Residue-shuffle cos (coords fixed, labels permuted) | N/A — encoder does not consume residue type |

CA-GearNet does **not** encode amino-acid identity — by design (input is CA-only, no residue features). Probe accuracy 13.7% is barely above the random-init floor (12.8%), and centroid cos 0.976 means the per-AA-type means are nearly indistinguishable. Implication: REPA against CA-GearNet is teaching the transformer about *geometry*, not *chemistry*.

### 1.2 3D geometric sensitivity

| Perturbation | Cosine similarity |
|--------------|-------------------|
| 0.1 Å Gaussian noise | 0.933 ± 0.016 |
| 0.5 Å Gaussian noise | **0.367 ± 0.058** |
| 1.0 Å Gaussian noise | 0.269 ± 0.046 |
| 2.0 Å Gaussian noise | 0.187 ± 0.030 |
| 5.0 Å Gaussian noise | −0.002 ± 0.018 |
| Random rotation | **0.99968 ± 0.0002** |

CA-GearNet is **highly 3D-sensitive**: a 0.5 Å perturbation (sub-Angstrom, comparable to thermal fluctuations) drops cos to 0.37. At 5 Å, embeddings are essentially decorrelated. Rotation invariance is near-perfect, confirming distance/angle edge features are SO(3)-invariant by construction.

### 1.3 Structural context (helix / sheet / loop)

CA-CA-CA bond angle as a coarse SS proxy (helix ~91°, sheet ~120°, loop = otherwise):

| AA  | within-SS cos | between-SS cos | Δ      | (helix / sheet / loop) |
|-----|--------------:|---------------:|-------:|------------------------|
| ALA | 0.237         | 0.201          | 0.036  | 2197 / 781 / 480       |
| GLY | 0.184         | 0.180          | 0.003  | 1496 / 740 / 832       |
| LEU | 0.230         | 0.201          | 0.029  | 2264 / 1166 / 490      |
| VAL | 0.244         | 0.199          | 0.045  | 1122 / 1335 / 540      |

Modest per-AA SS dependence. GLY ≈ 0 as expected (it's the conformationally promiscuous residue); VAL/ALA show the largest deltas. Embeddings carry SS information but it's secondary to overall protein context (§ 1.4).

### 1.4 Protein-level identity (within-protein vs between-protein)

| Metric                 | Value          |
|------------------------|----------------|
| Within-protein cos     | 0.416 ± 0.280  |
| Between-protein cos    | 0.195 ± 0.134  |
| **Δ**                  | **0.222**      |

Strong protein specificity: residues in the same protein are 2.1× more similar than residues across proteins. The random-init floor here is **0.035 ± 0.001** — pretraining lifts protein-specificity by ~6×. This is the metric where pretraining most clearly contributes; geometric/protein-context information is what REPA actually has to teach.

## Q2. How much is reachable from cheap inputs?

3-layer MLP from each input condition to the encoder embedding, 80/20 train/test split, trained 300 epochs.

| Input condition          | Train cos | Test cos |
|--------------------------|----------:|---------:|
| Mean direction (no MLP)  |        —  | **0.425** |
| Random 128-d             |    0.488  |    0.369  |
| AA one-hot (21-d)        |    0.433  |    0.431  |
| AA one-hot + position    |    0.433  | **0.432** |

**Saturation gap = best − mean-dir = +0.006.** This is the single most important number for REPA viability and is much smaller than ESM2's +0.053 (see [../FINDINGS.md](../FINDINGS.md)). Interpretation: identity + position alone reach almost everything the embedding offers; the encoder's *coordinate-derived* signal contributes little additional cosine alignment. Random init's gap is +0.003, so ≈ half of the trained encoder's "extra signal" is still being absorbed by the projector.

The random-input row trains to 0.488 but only generalises to 0.369 — the train/test gap (0.12) means the MLP is partly memorising per-residue noise rather than generalising structure, consistent with the encoder's true variance being small.

This does not mean the encoder is useless — it means the *cosine-aligned projector* extracts most of what's available before the transformer ever sees it. The geometric signal that CA-GearNet does carry shows up most clearly in within-vs-between protein similarity (Q1.4) rather than in absolute cosine to individual residues.

## Q3. Is the encoder a tractable optimisation target?

### 3.1 Sparsity & value distribution

| Metric | Value |
|--------|-------|
| Exact zeros | 0.00% |
| Near-zero (<1e-6) | 2.3 × 10⁻⁵% |
| Negative values | 62.4% |
| Mean | −1.10 |
| Std | 3.63 |
| Min / Max | −65.17 / 59.90 |

LeakyReLU (slope 0.1) produces a fully dense representation with ~62% negative values. No dead neurons, no gradient sparsity — every output dim contributes to every cosine-loss step.

### 3.2 Effective dimensionality & singular values

| Metric | Value |
|--------|-------|
| Output dim | 512 |
| Effective rank | **77.5** |
| Participation ratio | 37.7 |
| Dims for 90% variance | 78 |
| Dims for 95% variance | 118 |
| Dims for 99% variance | 273 |
| Top singular value | 4 323 |
| S[0] / S[−1] | 96.9 |

> Effective rank uses entropy on normalised squared singular values (variance): `exp(−Σ p log p)` where `p = S²ᵢ / Σ S²`. Same definition used across all encoder profiles.

77.5 / 512 = ~15% of capacity. The projector (512 → 512) has ample room. Compare to the random-init floor of **3.3 / 512** — pretraining is contributing ~23× more usable directions. This is the conditioning metric that most clearly distinguishes trained vs random and tracks the (small but positive) Q2 gap.

### 3.3 Norms & dead dimensions

| Metric | Value |
|--------|-------|
| Mean L2 norm | 79.6 |
| Std L2 norm | 32.0 |
| Min / Max L2 | 26.3 / 395.5 |
| Dead dimensions | **0 / 512** |
| Dim std range | [2.64, 4.82] |

Well-conditioned: every dimension active, narrow std range (1.8× ratio), per-AA norms vary modestly (CYS 90.5 highest, ASN 71.6 lowest — disulfide and acidic side chains predictably extreme). No norm explosion.

## Layer-wise representation

From [layerwise.csv](results/20260429_135327/layerwise.csv) (30 proteins, 8 layers):

| Layer | Eff rank | Mean norm | Sparsity |
|------:|---------:|----------:|---------:|
|     0 |     62.0 |     19.77 |     0.00 |
|     1 |     61.2 |     21.66 |     0.00 |
|     2 |     62.0 |     25.64 |     0.00 |
|     3 |     69.8 |     30.89 |     0.00 |
|     4 |     78.8 |     37.12 |     0.00 |
|     5 |     86.4 |     44.64 |     0.00 |
|     6 |   **87.3** |     55.48 |     0.00 |
|     7 |     68.2 |     78.85 |     0.00 |

Effective rank rises through depth and **peaks at layer 6** (87.3), then drops at the final layer (68.2) — the readout compresses information into a more rotation-aligned subspace. Norms grow monotonically (residual accumulation, ~4× from L0 to L7); cosine similarity normalises this away. **Implication**: aligning REPA to layer 6 instead of layer 7 may give more discriminative gradients; worth a brief comparison run.

## Implications for REPA training

1. **What REPA can teach with CA-GearNet:** geometric / protein-context information (Q1.4 Δ = 0.222 vs random 0.035). It cannot teach residue identity (Q1.1 — the encoder doesn't encode it).
2. **Tight projector saturation (Q2 gap +0.006):** the alignment loss may saturate quickly; expect modest cosine improvements during training. Consider lowering `lambda_repa` (e.g. 0.25) to avoid stealing gradient bandwidth from the flow-matching objective.
3. **Multi-layer alignment** (`layers: [2, 4, 6]`) plausibly helps spread the signal across the network — Q3.2 eff rank rises monotonically through depth.
4. **Auxiliary AA-type loss** could complement REPA cleanly: CA-GearNet doesn't carry residue identity (Q1.1), so a small CE loss on residue-type prediction from hidden states adds an orthogonal signal.
5. **Try aligning to layer 6 not 7:** rank/discrimination peaks one layer earlier than the standard readout.

For why CA-GearNet is preferred over MC-GearNet-Edge or PW-GearNet, and how it compares head-to-head against ESM2, see [../FINDINGS.md](../FINDINGS.md).
