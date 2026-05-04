# ESM2-650M Per-Residue Encoder Characterization

**Date**: 2026-05-04 (re-run with RankMe metric).
**Encoder**: ESM2 650M (`facebook/esm2_t33_650M_UR50D`), `last_hidden_state` (layer 33), 1280-dim
**Data**: 200 PDB train proteins (42,034 residues)
**SLURM**: 28852949 (full sweep). Latest results: [results/20260504_182607/results.json](results/20260504_182607/results.json), [layerwise.csv](results/20260504_182607/layerwise.csv).
**Cross-encoder context**: [../FINDINGS.md](../FINDINGS.md) — for the three-question framing (Q1 information, Q2 saturation, Q3 conditioning) used throughout.

## Summary

ESM2-650M produces a dense, well-conditioned representation with near-perfect amino-acid-identity signal (Q1.1: 99.8% linear probe). It also has the **largest projector saturation gap** of any encoder we've profiled (Q2: +0.053) — but that gap is misleading. The mean embedding direction alone reaches 0.67 test cos-sim; AA identity adds another 0.05; what's left for the transformer to teach is mostly *sequence context*, not *3D structure* (ESM ignores coordinates by construction, so Q1.2 is N/A). The last layer (33) collapses via a final LayerNorm — norms drop 56× (555 → 9.8), RankMe by ~5% (1023 → 977 at L33; PR drops by similar margin) — but mid-network layers 24–30 carry ~50% larger participation ratios at much higher norms, and are the more informative REPA targets.

## Q1. What information does the encoder encode?

### 1.1 Residue identity

| Metric | Value |
|--------|-------|
| Linear probe accuracy | **99.8%** (chance ~5%) |
| Mean cos between AA centroids | 0.853 |
| Within-type cos | 0.537 ± 0.157 |
| Between-type cos | 0.448 ± 0.186 |
| Δ (within − between) | **0.089** |
| Residue-shuffle cos (coords fixed, labels permuted) | N/A — encoder is sequence-only; identity *is* the input |

ESM's last layer effectively *is* an amino-acid classifier. The within-vs-between Δ (0.089) is real but modest — embeddings of different AAs are linearly separable but not far apart in cosine.

### 1.2 3D geometric sensitivity

**N/A — ESM2 is sequence-only.** ESM has no view of the 3D structure being generated; it ignores coordinates by construction. The perturbation/rotation tests are skipped via `EncoderProbe.is_3d_aware=False`. This is the central limitation of ESM as a REPA target for a 3D generative model: any cosine signal it provides is a function of `(sequence, position)` only — REPA cannot teach the student about *geometry*.

### 1.3 Sequence context (ESM-specific)

Fix the center residue's identity; scramble or randomise the flanks; measure how much the center embedding changes. (Used in place of structural context for sequence-only encoders.)

| Perturbation                          | Cosine similarity |
|---------------------------------------|-------------------|
| Shuffled flanks (same residue multiset) | 0.581 ± 0.189   |
| Random flanks (uniform 20-AA)         | 0.566 ± 0.191     |

Cos ~0.57 between "same AA, different context" means ESM is **not** a lookup table — neighbourhood information contributes a substantial fraction of the embedding. Shuffled and randomised flanks give indistinguishable results: the *fact* that flanks changed matters more than the specific multiset.

### 1.4 Protein-level identity (within-protein vs between-protein)

| Metric                | Value          |
|-----------------------|----------------|
| Within-protein cos    | 0.562 ± 0.180  |
| Between-protein cos   | 0.464 ± 0.179  |
| **Δ**                 | **0.098**      |

Embeddings cluster by protein, but less dramatically than CA-GearNet (Δ 0.222). The high between-protein baseline (0.46) is again the "shared direction" artefact of last-layer compression.

## Q2. How much is reachable from cheap inputs?

3-layer MLP, 80/20 train/test, 300 epochs.

| Input condition          | Train cos | Test cos |
|--------------------------|----------:|---------:|
| Mean direction (no MLP)  |        —  | **0.671** |
| Random 128-d             |    0.681  |    0.661  |
| AA one-hot (21-d)        |    0.724  |    0.724  |
| AA one-hot + position    |    0.724  | **0.724** |

**Saturation gap = +0.053** — the largest in the field (CA-GearNet +0.006, PW-GearNet:torsional +0.009, MC-GearNet-Edge −0.002). This is consistent with ESM2-REPA showing the cleanest val-loss improvement empirically.

But the gap should be read carefully: ~0.05 of "headroom" is what a cosine-loss projector cannot already extract from `(AA-onehot, position)`, and most of that is plausibly *sequence* context (which the student transformer already partially has via its own residue-type input) rather than *3D* context (which is what we'd most want from a REPA target for a 3D generative model). Cross-reference Q1.2 (N/A) and Q1.3 — the gap is *what kind* of information REPA can teach, and for ESM that kind is conformation-invariant.

The random-input row generalises (train 0.681, test 0.661 — essentially no train/test gap), unlike CA-GearNet's random row. ESM's distribution is regular enough that even meaningless input can fit a stable cosine target through the MLP.

## Q3. Is the encoder a tractable optimisation target?

### 3.1 Sparsity & value distribution

| Metric | Value |
|--------|-------|
| Exact zeros | 0.00% |
| Near-zero (<1e-6) | 4.8 × 10⁻⁴% |
| Negative values | 49.0% |
| Mean | −0.00070 |
| Std | 0.274 |
| Min / Max | −9.14 / 4.29 |

Fully dense, near-symmetric sign distribution. Standard Transformer + GELU output — no gradient sparsity.

### 3.2 Effective dimensionality & singular values

| Metric | Value |
|--------|-------|
| Output dim | 1280 |
| **RankMe** | **1030.5** |
| Participation ratio | 43.3 |
| Dims for 90% variance | 794 |
| Dims for 95% variance | 993 |
| Dims for 99% variance | 1 209 |
| Top singular value (centered) | 456.1 |
| S[0] / S[−1] (centered) | 2.1 × 10⁷ |

RankMe 1030 / 1280 = ~80% of capacity. Plenty of headroom; not bottlenecked. The layer-wise table below shows the L33 LayerNorm dips RankMe only ~5% below the L31 peak (1023 → 977) — under the old σ²-weighted metric this looked like a 45% drop, but it was an artefact of σ² weighting; the rank itself is mostly preserved. **The dramatic L33 effect is on norms, not rank.**

### 3.3 Norms & dead dimensions

| Metric | Value |
|--------|-------|
| Mean L2 norm | **9.81** |
| Std L2 norm | 0.30 |
| Min / Max | 9.12 / 10.85 |
| Dead dimensions | 0 / 1280 |
| Dim std range | [0.148, 1.904] |

**Strikingly uniform norms** (CV ~3%). Per-AA means fall in a narrow ~0.2-unit band. This uniformity is what fuels the Q2 projector saturation: cosine is normalisation-invariant, and the embeddings additionally share direction structure, so emitting the mean direction already gets you 0.67.

## Layer-wise analysis (33 transformer layers + token embedding)

From [layerwise.csv](results/20260504_182607/layerwise.csv) (30 proteins, 7k residues):

| Layer | RankMe | Mean norm | AA probe |
|------:|-------:|----------:|---------:|
|     0 |   19.1 |      2.79 |    1.000 |
|     1 |   57.1 |     94.27 |    1.000 |
|     2 |  255.8 |    112.53 |    1.000 |
|     3 |  487.8 |    118.59 |    1.000 |
|     4 |  658.5 |    111.82 |    1.000 |
|     5 |  772.9 |    103.87 |    1.000 |
|     6 |  836.0 |     93.64 |    1.000 |
|     7 |**852.0**|    91.59 |    1.000 |
|    12 |  562.2 |    160.82 |    0.998 |
|    16 |  739.6 |    262.57 |    0.998 |
|    20 |  803.2 |    333.40 |    0.990 |
|    24 |  903.4 |    368.43 |    0.981 |
|    25 |  926.2 |    374.20 |    0.983 |
|    26 |  945.3 |    384.16 |    0.985 |
|    27 |  969.9 |    391.76 |    0.985 |
|    28 |  992.4 |    399.61 |    0.990 |
|    29 | 1008.0 |    414.96 |    0.993 |
|    30 | 1018.0 |    445.77 |    0.991 |
|    31 |**1023.0**|  488.48 |    0.995 |
|    32 | 1022.4 |    554.66 |    0.995 |
| **33** (REPA target) | **976.6** | **9.79** | 0.998 |

Three distinct phases:

**Early (0–7)**: RankMe climbs from 19 (pure token lookup) to 852 at layer 7 as each block mixes in context. Norms saturate near 100.

**Middle (8–16)**: RankMe dips locally to ~562 at L12 and norms grow — a compression/refinement phase.

**Late (17–32)**: RankMe climbs monotonically to **1023 at layer 31**, with norms growing to ~555. These layers carry the richest representations: high-dim, high-magnitude, near-perfect AA identity preserved.

**Final compression (33)**: a LayerNorm-like operation drops norms by 56× (555 → 9.79). Under RankMe the dimensionality drop is modest (~5%, 1023 → 977) — the *direction structure* is largely preserved, but the *norm scale* is squeezed dramatically. This is the layer we currently align REPA against. Aligning against L24–L30 picks up similar RankMe with much richer norm structure.

## Why ESM is a problematic 3D-REPA target (synthesis across Q1/Q2/Q3)

1. **Q1.2 is N/A by construction.** ESM has no view of the 3D structure being generated. Any cosine signal it provides is a function of `(sequence, position)` only — REPA cannot teach the student transformer about *geometry*.
2. **The Q2 gap is mostly sequence-context.** Of the +0.053 projector gap, the headroom over `(AA-onehot, position)` is what we're really chasing. The student already has residue-type input; what's left is contextual sequence information ESM has internalised — useful but conformation-invariant.
3. **The last layer is a compressed projection (Q3 conditioning).** Layer 33's norms are 56× smaller than L32's, although under RankMe the rank itself only drops ~5% from peak. We're still aligning against a specialised output projection rather than the rich middle representations — the case is just that the compression is in *norm scale*, not in *direction count*.

## Recommendations

1. **Switch `repa.encoder.layer` from `null` to layer 24, 28, or 30.** Comparable RankMe (903–1018 vs L33's 977), AA identity preserved, much larger norms (370–446 vs L33's 9.8) — richer cosine signal per dimension. Cheapest experiment with a high prior on improvement.
2. **Multi-layer alignment** (`layers: [6, 24, 30]`): different depths capture different scales. ESM has high inter-layer redundancy, so the additional loss noise should be small.
3. **Sanity check the running val-cos.** If our ESM-REPA wandb runs plateau near 0.72, the saturation hypothesis is confirmed and REPA is doing nothing beyond AA recall. If they reach 0.78+, the transformer is finding usable additional signal.
4. **Compare empirically against CA-GearNet REPA.** Despite ESM's larger projector gap, CA-GearNet may train better because the headroom there is *geometric*, not sequence-redundant — exactly what a 3D generative model needs.

## Caveats

- 200 proteins, randomised seed 0. Single sample; PR-quality but not finely averaged.
- Layer-wise analysis sub-samples 30 proteins / ~7k residues for memory — per-layer eff-rank is a noisy estimator at that count, though the qualitative pattern (monotonic rise to L30, collapse at L33) is robust.
- Sequence-context test (Q1.3) fixes only the center residue; it conflates "neighbours provide context" with "neighbours break the local MLM prediction". Cleaner test: mutate *far* neighbours only.
