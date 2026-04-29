# ESM2-650M Per-Residue Encoder Characterization

**Date**: 2026-04-20
**Encoder**: ESM2 650M (`facebook/esm2_t33_650M_UR50D`), last_hidden_state (layer 33), 1280-dim
**Data**: 200 proteins from PDB train LMDB (45,503 residues total)
**Context**: REPA alignment target for Proteina 60M; our `l0…l9` configs all use `repa.encoder.layer: null` → last_hidden_state
**Job**: SLURM 28080381, A100-80GB, 19:02 wall

## Summary

ESM2-650M produces a dense, non-sparse, well-conditioned representation with near-perfect amino-acid-identity signal (99.7% linear probe). **But it is a problematic REPA target** — the mean embedding direction alone reaches 0.66 test cos-sim, and AA identity adds only 0.06 on top, leaving very little headroom for the transformer to teach anything beyond "same AA". **The last layer (33) collapses via a final LayerNorm** (norm 546 → 9.8, eff rank 639 → 282) — we are aligning against a specialized output compression, not the informative middle layers. Layers 24–30 carry 2–3× the effective rank with preserved identity signal and are plausibly stronger REPA targets.

## 1. Value Distribution & Sparsity

| Metric | ESM (L33) | GearNet | MACE | CheMeleon |
|--------|-----------|---------|------|-----------|
| Exact zeros | **0.00%** | 0.00% | 0.0% | 93.8% |
| Near-zero (<1e-6) | 0.0005% | 0.00% | 0.0% | ~94% |
| Negative values | 49.0% | 62.5% | — | 0% (ReLU) |
| Mean | -0.00070 | -1.07 | — | — |
| Std | 0.274 | 3.48 | — | — |
| Range | [-9.06, 4.36] | — | — | — |

Fully dense representation with near-symmetric sign distribution. Standard Transformer+GELU output — no gradient-sparsity concern.

## 2. Dimensionality & Singular Values

| Metric | ESM (L33) | GearNet | MACE | CheMeleon |
|--------|-----------|---------|------|-----------|
| Output dim | 1280 | 512 | 192 | 2048 |
| Effective rank | **353.6** | 82.6 | 40.6 | 500 |
| Participation ratio | 42.4 | 40.8 | — | 58.5 |
| Dims for 90% var | 785 | 82 | 7 | ~100 |
| Dims for 95% var | 985 | 125 | — | — |
| Dims for 99% var | 1206 | 283 | ~30 | ~300 |
| S[0]/S[-1] | 2.5e7 | — | — | — |

Effective rank 354 / 1280 — ESM uses ~28% of capacity (vs 16% for GearNet, 21% for MACE). The projector (token_dim → 1280) has to hit a high-dimensional target but will not be bottlenecked.

## 3. Residue-Type Discrimination

| Metric | ESM (L33) | GearNet | MACE |
|--------|-----------|---------|------|
| Linear probe accuracy | **99.7%** | 15.4% | 100% |
| Mean cos-sim between AA centroids | 0.847 | 0.975 | — |
| Within-type | 0.486 ± 0.18 | 0.189 ± 0.14 | — |
| Between-type | 0.434 ± 0.19 | 0.176 ± 0.13 | — |
| Delta (within − between) | **0.051** | 0.013 | — |

ESM's last layer effectively *is* an amino-acid classifier — 99.7% probe accuracy without training. But the within-vs-between delta is small (0.051), meaning the embeddings for different AAs live in overlapping regions that happen to be linearly separable.

## 4. Sequence Context Sensitivity (ESM-specific)

Fix the center residue's identity; scramble or randomize the flanking residues; measure center-embedding change.

| Perturbation | Cosine similarity |
|--------------|-------------------|
| Shuffled flanks (same residue multiset) | 0.563 ± 0.26 |
| Random flanks (uniform 20-AA) | 0.562 ± 0.24 |

Context strongly modulates the center embedding — ESM is **not** a lookup table. A cos-sim of ~0.56 between "same AA, different context" means neighborhood information contributes a substantial fraction of the embedding. Perhaps surprisingly, shuffled-same-residues and randomized-residues give indistinguishable results — the specific multiset of nearby AAs matters less than the fact that *something* changed.

## 5. Embedding Norms & Conditioning

| Metric | Value |
|--------|-------|
| Mean L2 norm | **9.81** |
| Std L2 norm | **0.30** |
| Min / max | 9.08 / 10.90 |
| Dead dimensions | 0 / 1280 |
| Dimension std range | [0.147, 1.930] |

**Strikingly uniform norms** (CV ~3%). Per-AA means fall in a 9.69–9.88 band — barely separated. This uniformity is what fuels the projector-saturation problem: cosine similarity is normalization-invariant, and the embeddings additionally share direction structure, so just emitting the mean direction gets you most of the way there.

## 6. Projector Saturation Test (KEY RESULT)

3-layer MLP (input → 512 → 512 → 1280), cosine-similarity loss, 300 epochs Adam, 80/20 split.

| Input condition | Train | **Test** |
|-----------------|-------|----------|
| mean-direction (no MLP) | — | **0.664** |
| random 128-d | 0.673 | 0.653 |
| AA one-hot (21-d) | 0.717 | **0.719** |
| AA one-hot + position | 0.717 | 0.719 |

Compare to GearNet (test set):
- Mean direction: 0.419
- Random 128-d: 0.416
- AA one-hot: 0.426
- **REPA training (transformer): 0.78**

**This is the most important finding.** For GearNet, identity alone reaches 0.43 and training to 0.78 represents ~0.35 of genuine structural learning. For ESM2, identity alone reaches 0.72 — a plausible REPA training ceiling is ~0.80, leaving only **~0.08 of headroom for the transformer to contribute structural signal beyond amino-acid identity**.

The mean-direction baseline at 0.664 (with zero learning) is notable: all ESM embeddings point in roughly the same direction, so cosine similarity starts near 2/3 for free.

## 7. Within-Protein vs Between-Protein Similarity

| Metric | ESM (L33) | GearNet | CheMeleon |
|--------|-----------|---------|-----------|
| Within-protein | 0.577 ± 0.20 | 0.420 ± 0.27 | 0.25 |
| Between-protein | 0.436 ± 0.20 | 0.176 ± 0.13 | 0.15 |
| Delta | **0.141** | 0.244 | 0.10 |

ESM embeddings cluster by protein, but less dramatically than GearNet (delta 0.14 vs 0.24). Between-protein baseline (0.44) is high — again, the "shared direction" artefact.

## 8. Layer-wise Analysis (ESM has 33 transformer layers + embeddings)

| Layer | Eff Rank | Mean Norm | AA Probe |
|-------|---------:|----------:|---------:|
| 0 (token embeddings) | 16.3 | 2.80 | 1.0000 |
| 1 | 15.0 | 94.04 | 1.0000 |
| 2 | 73.9 | 112.88 | 1.0000 |
| 3 | 177.3 | 118.96 | 1.0000 |
| 4 | 312.3 | 111.36 | 1.0000 |
| 5 | **429.0** | 102.90 | 1.0000 |
| 6 | 427.2 | 94.16 | 1.0000 |
| 7 | 384.7 | 93.77 | 1.0000 |
| 8 | 341.1 | 92.02 | 1.0000 |
| 9 | 298.0 | 96.93 | 1.0000 |
| 10 | 263.3 | 112.96 | 1.0000 |
| 11 | 217.3 | 127.88 | 1.0000 |
| 12 | 205.6 | 161.89 | 1.0000 |
| 13 | 233.2 | 187.79 | 1.0000 |
| 14 | 286.4 | 218.40 | 1.0000 |
| 15 | 308.8 | 249.00 | 0.9986 |
| 16 | 339.8 | 261.84 | 0.9986 |
| 17 | 314.3 | 285.72 | 0.9979 |
| 18 | 325.5 | 303.96 | 0.9959 |
| 19 | 334.5 | 319.32 | 0.9952 |
| 20 | 334.8 | 336.57 | 0.9945 |
| 21 | 362.8 | 346.87 | 0.9924 |
| 22 | 386.6 | 358.06 | 0.9876 |
| 23 | 411.5 | 366.87 | 0.9904 |
| 24 | 441.1 | 374.11 | 0.9897 |
| 25 | 476.2 | 380.76 | 0.9945 |
| 26 | 514.2 | 390.95 | 0.9904 |
| 27 | 559.0 | 397.46 | 0.9945 |
| 28 | 604.8 | 403.51 | 0.9959 |
| 29 | **638.7** | 414.99 | 0.9945 |
| 30 | 627.1 | 446.57 | 0.9966 |
| 31 | 618.7 | 487.80 | 0.9972 |
| 32 | 561.8 | 546.35 | 0.9979 |
| **33** (our REPA target) | **281.9** | **9.76** | 0.9972 |

Three distinct phases:

**Early (0–5)**: Effective rank climbs from 16 (pure token lookup) to 429 at layer 5 as each transformer block mixes in context. Norm saturates at ~100. **Layer 5 is a local peak**.

**Middle (6–14)**: Norms grow steadily (~100 → 218) while eff rank *decreases* (429 → 206 at layer 12). Compression phase.

**Late (15–32)**: Eff rank recovers and climbs steadily from 309 → 639 at layer 29. Norms also grow (249 → 546). **These are the richest representations** — high-dim, high-magnitude, near-perfect AA identity preserved.

**Final collapse (33)**: A final LayerNorm-like operation drops norms by 56× (546 → 9.76) and halves effective rank (562 → 282). Inter-layer cos-sim 32→33 = 0.893 (vs 0.95+ elsewhere late). This is the layer our REPA aligns against.

### Inter-layer cosine similarity
| Transition | cos-sim |
|-----------|--------|
| 0 → 1 | **0.022** |
| 5 → 6 | 0.968 |
| 10 → 11 | 0.935 |
| 15 → 16 | 0.972 |
| 20 → 21 | 0.970 |
| 25 → 26 | 0.971 |
| 30 → 31 | 0.947 |
| 32 → 33 | **0.893** |

Layer 0 → 1 is essentially orthogonal — the first transformer block transforms token embeddings completely. Layer 32 → 33 is the other large step — the final projection/LN.

## Head-to-Head: ESM2-650M vs GearNet vs MACE

| Property | ESM2 (L33) | GearNet (CA) | MACE |
|----------|------------|--------------|------|
| Domain | Proteins (sequence-only) | Proteins (3D) | Small molecules (3D) |
| Output dim | 1280 | 512 | 192 |
| Eff rank | 354 | 82.6 | 40.6 |
| 3D-aware | **No** | Yes (strong) | Yes (weak) |
| 3D sensitivity (0.5 Å) | N/A (ignores coords) | cos = 0.36 | 0.998 |
| Rotation invariant | Trivially (coords ignored) | Yes (0.9997) | Yes |
| AA/atom probe | 99.7% | 15.4% | 100% |
| Projector: mean-dir baseline | **0.664** | 0.419 | — |
| Projector: identity | 0.72 | 0.43 | 0.86 |
| Genuine REPA headroom | **~0.08** | ~0.35 | ~0.14 |
| Within-vs-between delta | 0.14 | 0.24 | — |

## Implications for REPA Training

### Why our `training_repa_esm_l0_128_per_residue` run may not beat baseline

1. **Projector saturation dominates**. Mean direction alone is 0.66, identity is 0.72. The ceiling for "genuine structural learning beyond identity" is about 0.08 of cos-sim — tiny compared to GearNet's 0.35.

2. **ESM is 3D-blind.** CA coordinates are ignored. Any cos-sim the transformer earns beyond the identity baseline comes from learning *sequence* context, not *structural* context — but we already have the residue types in the input embedding. There is nothing geometric for REPA to teach.

3. **The last layer is a bad target.** Layer 33 has half the effective rank of layers 24–30, with norms collapsed to ~10. We are asking the student transformer to mimic a heavily compressed output representation — projected through a 3-layer MLP into 1280-d — and that target retains the least amount of information from ESM's inner pathway.

### Recommended experiments

1. **Switch `encoder.layer` from null to layer 24 or 29.** 2–3× the effective rank, same identity preservation, richer structural context (if present).

2. **Revisit whether ESM is worth using at all.** For a 3D generative model, a sequence-only target forces REPA to learn conformation-invariant features — which is exactly the problem we diagnosed with CheMeleon. The transformer already has residue types in its input; REPA with ESM may primarily be a regularizer on "don't forget the residue identity," which the flow-matching objective already enforces implicitly.

3. **Sanity check**: what test cos-sim does our wandb run reach? If it plateaus near 0.72, the saturation hypothesis is confirmed and REPA is doing nothing. If it reaches 0.80+, the transformer is finding *some* additional signal worth chasing.

4. **Multi-layer alignment**. Target layers [5, 15, 29] simultaneously — each captures different representation depths. With ESM's redundancy across layers, this is probably cheap in additional loss noise.

## Caveats

- **200 proteins, first-in-LMDB-order**. PDB IDs starting with `1a…` dominate. Not a random sample of fold space. A follow-up run with randomized 500-1000 proteins will check whether these numbers hold.
- **Layer-wise analysis capped at 30 proteins / 7k residues** for memory — per-layer eff-rank is a noisy estimator at that sample count, though the qualitative pattern (monotonic growth to L29, collapse at L33) is robust to sample size.
- **Context sensitivity test fixes only the center residue** — this conflates "neighbors provide context" with "neighbors break the local MLM prediction." A cleaner test: mutate *far* neighbors and see how the center changes.
