# ESM2-650M Per-Residue Encoder Characterization

**Date**: 2026-04-29
**Encoder**: ESM2 650M (`facebook/esm2_t33_650M_UR50D`), `last_hidden_state` (layer 33), 1280-dim
**Data**: 200 PDB train proteins (42,034 residues)
**SLURM**: 28596016 (full sweep). Latest results: [results/20260429_135732/results.json](results/20260429_135732/results.json), [layerwise.csv](results/20260429_135732/layerwise.csv).
**Cross-encoder context**: [../FINDINGS.md](../FINDINGS.md).

## Summary

ESM2-650M produces a dense, well-conditioned representation with near-perfect amino-acid-identity signal (99.8% linear probe). It also has the **largest projector saturation gap** of any encoder we've profiled (+0.053) — but that gap is misleading. The mean embedding direction alone reaches 0.67 test cos-sim; AA identity adds another 0.05; what's left for the transformer to teach is mostly *sequence context*, not *3D structure* (ESM ignores coordinates by construction). The last layer (33) collapses via a final LayerNorm — norms drop 56× (555 → 9.8), eff rank halves (583 → 322) — so we are aligning the student against ESM's most compressed output. Layers 24–30 carry ~2× the effective rank with preserved identity and are the more informative REPA targets.

## 1. Value distribution & sparsity

| Metric | Value |
|--------|-------|
| Exact zeros | 0.00% |
| Near-zero (<1e-6) | 4.8 × 10⁻⁴% |
| Negative values | 49.0% |
| Mean | −0.00070 |
| Std | 0.274 |
| Min / Max | −9.14 / 4.29 |

Fully dense, near-symmetric sign distribution. Standard Transformer + GELU output — no gradient sparsity.

## 2. Dimensionality & singular values

| Metric | Value |
|--------|-------|
| Output dim | 1280 |
| Effective rank | **360.6** |
| Participation ratio | 43.7 |
| Dims for 90% variance | 794 |
| Dims for 95% variance | 993 |
| Dims for 99% variance | 1 209 |
| Top singular value | 454.6 |
| S[0] / S[−1] | 2.0 × 10⁷ |

361 / 1280 = ~28% of capacity. Plenty of headroom; not bottlenecked.

## 3. Residue-type discrimination

| Metric | Value |
|--------|-------|
| Linear probe accuracy | **99.8%** (chance ~5%) |
| Mean cos between AA centroids | 0.853 |
| Within-type cos | 0.537 ± 0.157 |
| Between-type cos | 0.448 ± 0.186 |
| Δ (within − between) | **0.089** |

ESM's last layer effectively *is* an amino-acid classifier. The within-vs-between Δ (0.089) is real but modest — embeddings of different AAs are linearly separable but not far apart.

## 4. Sequence context sensitivity (ESM-specific)

Fix the center residue's identity; scramble or randomize the flanks; measure how much the center embedding changes.

| Perturbation                          | Cosine similarity |
|---------------------------------------|-------------------|
| Shuffled flanks (same residue multiset) | 0.581 ± 0.189   |
| Random flanks (uniform 20-AA)         | 0.566 ± 0.191     |

Cos ~0.57 between "same AA, different context" means ESM is **not** a lookup table — neighborhood information contributes a substantial fraction of the embedding. Shuffled and randomized flanks give indistinguishable results: the *fact* that flanks changed matters more than the specific multiset.

## 5. Embedding norms & conditioning

| Metric | Value |
|--------|-------|
| Mean L2 norm | **9.81** |
| Std L2 norm | 0.30 |
| Min / Max | 9.12 / 10.85 |
| Dead dimensions | 0 / 1280 |
| Dim std range | [0.148, 1.904] |

**Strikingly uniform norms** (CV ~3%). Per-AA means fall in a narrow ~0.2-unit band. This uniformity is what fuels the projector saturation: cosine is normalization-invariant, and the embeddings additionally share direction structure, so emitting the mean direction already gets you 0.67.

## 6. Projector saturation (key result)

3-layer MLP, 80/20 train/test, 300 epochs.

| Input condition          | Train cos | Test cos |
|--------------------------|----------:|---------:|
| Mean direction (no MLP)  |        —  | **0.671** |
| Random 128-d             |    0.681  |    0.661  |
| AA one-hot (21-d)        |    0.724  |    0.724  |
| AA one-hot + position    |    0.724  | **0.724** |

**Saturation gap = +0.053** — the largest in the field (CA-GearNet +0.006, PW-GearNet:torsional +0.009, MC-GearNet-Edge −0.002). This is consistent with ESM2-REPA showing the cleanest val-loss improvement empirically. But the gap should be read carefully: ~0.05 of "headroom" is what a cosine-loss projector cannot already extract from `(AA-onehot, position)`, and most of that is plausibly *sequence* context (which the student transformer already partially has via its own residue-type input) rather than *3D* context (which is what we'd most want).

## 7. Within-protein vs between-protein similarity

| Metric                | Value          |
|-----------------------|----------------|
| Within-protein cos    | 0.562 ± 0.180  |
| Between-protein cos   | 0.464 ± 0.179  |
| **Δ**                 | **0.098**      |

Embeddings cluster by protein, but less dramatically than CA-GearNet (Δ 0.222). The high between-protein baseline (0.46) is again the "shared direction" artefact.

## 8. Layer-wise analysis (33 transformer layers + token embedding)

From [layerwise.csv](results/20260429_135732/layerwise.csv) (30 proteins, 7k residues):

| Layer | Eff rank | Mean norm | AA probe |
|------:|---------:|----------:|---------:|
|     0 |     16.0 |      2.79 |    1.000 |
|     1 |     45.1 |     94.27 |    1.000 |
|     2 |    151.7 |    112.53 |    1.000 |
|     3 |    289.1 |    118.59 |    1.000 |
|     4 |    407.0 |    111.82 |    1.000 |
|     5 |    502.6 |    103.87 |    1.000 |
|     6 | **520.8** |     93.64 |    1.000 |
|     7 |    457.6 |     91.59 |    1.000 |
|    12 |    212.3 |    160.82 |    0.998 |
|    16 |    332.8 |    262.57 |    0.998 |
|    20 |    349.2 |    333.40 |    0.990 |
|    24 |    465.9 |    368.43 |    0.981 |
|    25 |    505.7 |    374.20 |    0.983 |
|    26 |    545.6 |    384.16 |    0.985 |
|    27 |    592.6 |    391.76 |    0.985 |
|    28 |    637.9 |    399.61 |    0.990 |
|    29 |    659.6 |    414.96 |    0.993 |
|    30 |  **661.9** |    445.77 |    0.991 |
|    31 |    651.9 |    488.48 |    0.995 |
|    32 |    582.7 |    554.66 |    0.995 |
| **33** (REPA target) | **321.7** | **9.79** | 0.998 |

Three distinct phases:

**Early (0–6)**: rank climbs from 16 (pure token lookup) to 521 at layer 6 as each block mixes in context. Norms saturate near 100. **Layer 6 is a local peak**.

**Middle (7–14)**: rank dips and norms grow — a compression/refinement phase.

**Late (15–32)**: rank recovers and climbs to **662 at layer 30**. Norms grow with depth. These are the richest representations: high-dim, high-magnitude, near-perfect AA identity preserved.

**Final collapse (33)**: a LayerNorm-like operation drops norms by 56× (555 → 9.79) and effective rank by ~45% (583 → 322). This is the layer we currently align REPA against.

## 9. Why ESM is a problematic 3D-REPA target

1. **Coordinates are ignored.** ESM has no view of the 3D structure being generated. Any cosine signal it provides is a function of `(sequence, position)` only — REPA cannot teach the student transformer about *geometry*.
2. **The gap is mostly sequence-context.** Of the +0.053 projector gap, the headroom over `(AA-onehot, position)` is what we're really chasing. The student already has residue-type input; what's left is contextual sequence information ESM has internalised — useful but conformation-invariant.
3. **The last layer is the worst layer.** Layer 33 has roughly half the effective rank of layers 24–30 with norms compressed to ~10. We're aligning against a specialised output projection rather than the rich middle representations.

## Recommendations

1. **Switch `repa.encoder.layer` from `null` to layer 24, 28, or 30.** ~2× effective rank, identity preserved, richer representation. Cheapest experiment with a high prior on improvement.
2. **Multi-layer alignment** (`layers: [6, 24, 30]`): different depths capture different scales. ESM has high inter-layer redundancy, so the additional loss noise should be small.
3. **Sanity check the running val-cos.** If our ESM-REPA wandb runs plateau near 0.72, the saturation hypothesis is confirmed and REPA is doing nothing beyond AA recall. If they reach 0.78+, the transformer is finding usable additional signal.
4. **Compare empirically against CA-GearNet REPA.** Despite ESM's larger projector gap, CA-GearNet may train better because the headroom there is *geometric*, not sequence-redundant — exactly what a 3D generative model needs.

## Caveats

- 200 proteins, randomized seed 0. Single sample; PR-quality but not finely averaged.
- Layer-wise analysis sub-samples 30 proteins / ~7k residues for memory — per-layer eff-rank is a noisy estimator at that count, though the qualitative pattern (monotonic rise to L30, collapse at L33) is robust.
- Sequence-context test fixes only the center residue; it conflates "neighbors provide context" with "neighbors break the local MLM prediction". Cleaner test: mutate *far* neighbors only.
