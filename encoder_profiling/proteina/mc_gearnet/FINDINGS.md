# MC-GearNet-Edge Per-Residue Encoder Characterization

**Date**: 2026-04-29
**Encoder**: MC-GearNet-Edge (`MCGearNetEdgePerResidueEncoder`, 6 layers, 3072-dim concat output)
**Checkpoint**: `mc_gearnet_edge.pth` (Zenodo 7593637)
**Data**: 200 PDB train proteins (42,034 residues)
**SLURM**: 28596016 (full sweep). Latest results: [results/20260429_135919/results.json](results/20260429_135919/results.json), [layerwise.csv](results/20260429_135919/layerwise.csv).
**Cross-encoder context**: [../FINDINGS.md](../FINDINGS.md).

## Summary

**MC-GearNet-Edge is unusable as a REPA target in its current form.** The `concat_hidden=True` output concatenates 6 BatchNorm + residual layers whose mean L2 norms grow ~17,500× from layer 0 (82) to layer 5 (1.45 × 10⁶). Layer 5 dominates the concatenated 3072-d output; the whole representation collapses onto a 1-D subspace (effective rank **1.1 / 3072**, top direction carries ≥95% of variance). The mean-direction baseline reaches **0.855** test cos-sim, and no projector input (random, AA one-hot, +position) beats it by more than 0.001 — the saturation gap is **negative** (−0.002). The residue-shuffle test (coords fixed, labels permuted) yields cos = 0.983: residue identity barely modulates the embedding despite being fed in as node features. Geometry is preserved (0.5 Å noise drops cos only to 0.81), but the gradient signal sits in the 1-D exploding subspace and carries almost no information to align against.

## 1. Value distribution & sparsity

| Metric | Value |
|--------|-------|
| Exact zeros | 0.00% |
| Negative values | 85.1% |
| Mean | 601.7 |
| Std | 52 290 |
| Min / Max | −41 487 / **52 008 680** |

Values are enormous — max 5.2 × 10⁷. Fully dense, but "dense" is meaningless when one axis dwarfs the rest.

## 2. Dimensionality & singular values

| Metric | Value |
|--------|-------|
| Output dim | 3072 |
| Effective rank | **1.1** |
| Participation ratio | 1.0 |
| Dims for 90% / 95% / 99% variance | **1 / 1 / 2** |
| Top singular value | 4.3 × 10⁸ |
| S[0] / S[−1] | **4.3 × 10²⁰** |

**Catastrophic rank collapse.** Eff rank 1.1 / 3072 means the representation occupies essentially a single direction. S[0]/S[−1] = 10²⁰ is numerical-precision territory — layer 5 (see § 9) accounts for nearly all variance because its norm is ~60× the sum of layers 0–4.

## 3. 3D sensitivity

| Perturbation | Cosine similarity |
|--------------|------------------:|
| 0.1 Å Gaussian | 0.883 ± 0.032 |
| 0.5 Å Gaussian | 0.809 ± 0.063 |
| 1.0 Å Gaussian | 0.783 ± 0.073 |
| 2.0 Å Gaussian | 0.740 ± 0.081 |
| 5.0 Å Gaussian | **0.696 ± 0.110** |
| Random rotation | 0.99974 ± 0.0011 |
| **Residue-type shuffle** | **0.983 ± 0.005** |

CA-GearNet drops to 0.37 at 0.5 Å. MC-GearNet stays above 0.70 even at 5 Å. The embedding is dominated by a direction that depends only weakly on coordinates — consistent with an exploding-BN attractor that all inputs converge toward.

**Rotation invariant** (0.99974) — geometry is used only via invariant features, as designed.

**Residue-shuffle cos = 0.983** is the most damning. MC-GearNet-Edge takes residue identity as its node features (one-hot → 21 dims), and yet permuting the labels within a protein barely moves the embedding. Whatever information the network transmits is neither from *which* residues nor from *where* they are.

## 4. Residue-type discrimination

| Metric | Value |
|--------|-------|
| Linear probe accuracy | 15.0% |
| Mean cos between AA centroids | **0.9999** |
| Within-type cos | 0.770 ± 0.365 |
| Between-type cos | 0.729 ± 0.394 |
| Δ (within − between) | **+0.041** |

Per-AA centroids have pairwise cos 0.9999 — they all point in the same direction. The within/between Δ of +0.04 swims inside the ±0.4 standard deviations: it is not a meaningful discrimination signal. Probe accuracy 15.0% is comparable to CA-GearNet's 13.7% — but CA-GearNet doesn't see residue types at all.

## 5. Structural context sensitivity

| AA  | within-SS cos | between-SS cos | Δ      |
|-----|--------------:|---------------:|-------:|
| ALA | 0.837         | 0.846          | −0.009 |
| GLY | 0.698         | 0.723          | −0.025 |
| LEU | 0.773         | 0.762          | +0.011 |
| VAL | 0.746         | 0.737          | +0.009 |

Two of four AAs show *negative* delta (between > within), again consistent with noise. CA-GearNet had clear positive deltas for most AAs. MC-GearNet has no meaningful within- vs between-context structure.

## 6. Embedding norms & conditioning

| Metric | Value |
|--------|-------|
| Mean L2 norm | **1 476 209** |
| Std L2 norm | 2 494 352 |
| Min / Max | 35.3 / 5.2 × 10⁷ |
| Dead dimensions | **507 / 3072** |
| Dim std range | [0.0, 2.48 × 10⁶] |

Mean norm 1.48 million — ~18 500× CA-GearNet. **507 dead dimensions** — close to one full 512-dim layer slab (any of the early layers; see § 9), confirming one of the 6 concatenated slabs contributes nothing. Norm ratio max/min ≈ 1.5 × 10⁶ per residue — no consistent magnitude scale.

## 7. Projector saturation (key result)

3-layer MLP, 80/20 train/test, 300 epochs.

| Input condition          | Train cos | Test cos |
|--------------------------|----------:|---------:|
| Mean direction (no MLP)  |        —  | **0.855** |
| Random 128-d             |    0.860  |    0.852  |
| AA one-hot (21-d)        |    0.860  |    0.853  |
| AA one-hot + position    |    0.860  | **0.853** |

**Saturation gap = best − mean-dir = −0.002.** The projector cannot beat the zero-parameter mean-direction baseline. Three radically different inputs (random noise, AA one-hot, AA one-hot + position) all land within 0.001 of each other and below the constant baseline. A 3-layer MLP with 512 hidden has hundreds of thousands of parameters and 300 epochs to beat a constant — and cannot. The target is a ray in 3072-d; any prediction that points down that ray wins, regardless of input.

## 8. Within-protein vs between-protein similarity

| Metric                | Value          |
|-----------------------|----------------|
| Within-protein cos    | 0.757 ± 0.376  |
| Between-protein cos   | 0.714 ± 0.402  |
| **Δ**                 | **0.043**      |

Δ 0.043 — residues in the same protein are barely more similar than residues in different proteins, with overlapping ±0.4 std bands. By contrast CA-GearNet has Δ 0.222, ESM2 (sequence-only) has Δ 0.098.

## 9. Layer-wise representation

The final output is the **concatenation** of all 6 hidden layers (`concat_hidden=True`), each 512-dim → 3072-d. From [layerwise.csv](results/20260429_135919/layerwise.csv):

| Layer | Dim | Eff rank | Mean norm |
|------:|----:|---------:|----------:|
|     0 | 512 |      5.6 |     **82.0** |
|     1 | 512 |      9.9 |        36.0 |
|     2 | 512 |      7.1 |       264.7 |
|     3 | 512 |      5.0 |      1 380.3 |
|     4 | 512 |      5.2 |     **24 379.8** |
|     5 | 512 |    **1.1** | **1 451 946.8** |

**Layer-norm timeline**: 82 → 36 → 265 → 1 380 → 24 380 → 1 451 947. Layer 5 / layer 0 = ~17 700×; layer 5 / Σ(layers 0–4) ≈ 56×. When concatenated, layer 5's values — collapsed to ~1-D — dwarf everything.

**Effective rank** collapses from 9.9 (layer 1) to 1.1 (layer 5). Each successive layer projects onto fewer directions with larger magnitude. This is not healthy representation evolution; it is the BN+residual chain amplifying and rotating a collapsing attractor.

In isolation, **layer 0 or layer 1** (norms 82 and 36, eff rank 5–10, 0% sparsity, rotation-invariant) would be a usable — if low-rank — REPA target. The problem is the concat: it exposes only the worst slab.

## Implications for REPA training

### Do not use MC-GearNet-Edge as a REPA target in its current form

The representation is 1-dimensional with exploding norms. No projector can fit it above the mean-direction baseline, and no gradient signal can flow back to the student transformer. Any REPA run using MC-GearNet will plateau at cos-sim ≈ 0.85 on the first few steps and contribute nothing afterward — the loss curve will look *better* than CA-GearNet (higher cos-sim!) while teaching the model nothing.

### Root cause

`concat_hidden=True` over 6 BatchNorm + residual layers without a final LayerNorm — norms grow geometrically. Layer 5 has mean L2 ≈ 1.5 million; all earlier layers together contribute < 2% of the output magnitude. The concatenation is effectively layer 5 padded with noise.

### Possible fixes (not tested here)

1. **Use a single intermediate layer, not the concat.** Layer 0 or 1 has eff rank 5–10 and norms O(100). Usable, low-rank REPA target.
2. **LayerNorm the per-layer slabs before concat.** Standard protocol for `concat_hidden` encoders; absent here.
3. **z-score the encoder output** inside the REPA loss (running mean/std). Addresses the symptom but not the collapse — eff rank 1.1 means 1 informative dim out of 3072 even after standardisation.
4. **Drop MC-GearNet-Edge.** CA-GearNet gives more usable REPA signal; MC's extra features (edge types, angle bins, residue identity) do not translate into representation. The simplest conclusion is that this checkpoint's output isn't designed to be consumed raw.

### Implication for `gearnet_mc_edge` config variants in the tree

`src/proteina/configs/experiment_config/training/{128,256,512}/gearnet_mc_edge/per_residue/training_repa_l{0,4,9}_*_mc_edge.yaml` all target the concatenated 3072-d output via a 3-layer projector. If any of these have been run, expect their REPA loss curves to be near-flat (or confusingly high cos-sim) and their downstream metrics to match the non-REPA baseline within noise.

## Caveats

- 200 proteins, randomized seed 0. Same sample as the other encoders so comparisons are apples-to-apples.
- SVD/probe subsampled to 30k residues for memory. Qualitative result (eff rank ≈ 1) is robust to sample size; the collapse is structural.
- Linear probe did not converge (`lbfgs` hit max_iter=1000). Scaling features first would likely help, but the probe is near chance anyway.
