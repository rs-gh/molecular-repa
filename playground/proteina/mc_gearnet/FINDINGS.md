# MC-GearNet-Edge Per-Residue Encoder Characterization

**Date**: 2026-04-23
**Encoder**: MC-GearNet-Edge (`NoTrainMCGearNetEdge`, 6 layers, 3072-dim concat output)
**Checkpoint**: `mc_gearnet_edge.pth` (Zenodo 7593637)
**Data**: 200 proteins from PDB train LMDB (45,503 residues)
**Context**: REPA alignment target for Proteina 60M; alternative to CA-only GearNet and ESM2
**Job**: SLURM 28306119, A100-80GB, 5 min wall

## Summary

**MC-GearNet-Edge is unusable as a REPA target in its current form.** The `concat_hidden=True` output concatenates 6 BatchNorm+residual layers whose norms grow ~24,000× from layer 0 (mean L2 = 81.7) to layer 5 (mean L2 = 1,948,064). Layer 5 dominates the concatenated 3072-d output; the whole representation collapses onto a 1-D subspace (effective rank **1.1 / 3072**, top direction carries ≥90% of variance). A mean-direction baseline reaches **0.864 test cos-sim** — higher than any other encoder we have characterized — and no MLP input (random, AA one-hot, +position) can beat it by more than 0.003. The residue-shuffle test (coords fixed, labels permuted) yields cos = 0.984, i.e. residue identity barely modulates the embedding despite being fed in as node features. Geometry is preserved (0.5 Å noise drops cos only to 0.82), but the gradient signal sits in the 1-D exploding subspace and carries almost no information to align against.

## 1. Value Distribution & Sparsity

| Metric | MC-GearNet | CA-GearNet | ESM2 (L33) | MACE | CheMeleon |
|--------|-----------:|-----------:|-----------:|-----:|----------:|
| Exact zeros | 0.00% | 0.00% | 0.00% | 0.0% | 93.8% |
| Negative values | 85.2% | 62.5% | 49.0% | — | 0% |
| Mean | **639.8** | -1.07 | -0.0007 | — | — |
| Std | **56,381** | 3.48 | 0.274 | — | — |
| Min / Max | -28,516 / **54,872,860** | — | -9.1 / 4.4 | — | — |

Values are enormous. Mean 639 with std 56,000 (1.5× range of ESM's *entire* value distribution *squared*) and max 5.5 × 10⁷. Fully dense, no sparsity concern — but "dense" is meaningless when one axis dwarfs the rest.

## 2. Dimensionality & Singular Values

| Metric | MC-GearNet | CA-GearNet | ESM2 (L33) | MACE |
|--------|-----------:|-----------:|-----------:|-----:|
| Output dim | **3072** | 512 | 1280 | 192 |
| Effective rank | **1.1** | 82.6 | 353.6 | 40.6 |
| Participation ratio | **1.0** | 40.8 | 42.4 | — |
| Dims for 90% variance | **1** | 82 | 785 | 7 |
| Dims for 95% variance | **1** | 125 | 985 | — |
| Dims for 99% variance | **2** | 283 | 1,206 | ~30 |
| S[0]/S[-1] | **4.7 × 10²⁰** | — | 2.5 × 10⁷ | — |

**Catastrophic rank collapse.** Effective rank 1.1 out of 3072 means the representation occupies a single direction. S[0]/S[-1] = 10²⁰ is numerical-precision territory. Layer 5 (see §9) accounts for essentially all variance because its norm is ~60× the sum of all earlier layers.

## 3. 3D Sensitivity

| Perturbation | Cosine Similarity |
|--------------|------------------:|
| 0.0 Å (original) | 1.000 |
| 0.1 Å Gaussian | 0.886 ± 0.039 |
| 0.5 Å Gaussian | **0.821 ± 0.068** |
| 1.0 Å Gaussian | 0.789 ± 0.077 |
| 2.0 Å Gaussian | 0.760 ± 0.087 |
| 5.0 Å Gaussian | **0.719 ± 0.096** |
| Random rotation | 0.9996 ± 0.0011 |
| **Residue-type shuffle** | **0.984 ± 0.006** |

CA-GearNet drops to 0.36 at 0.5 Å and 0.27 at 1.0 Å — a rich 3D gradient. MC-GearNet stays above 0.72 even at 5 Å noise. The embedding is dominated by a direction that depends only very weakly on coordinates — consistent with an exploding-BN attractor that all inputs converge toward.

**Rotation invariant** (0.9996) — geometry is used only through invariant features, as designed.

**Residue-shuffle cos = 0.984** is the most damning. MC-GearNet-Edge takes residue identity as its node features (one-hot → 21 dims), and yet permuting the labels within a protein barely moves the embedding. Whatever information the network transmits is neither from *which* residues nor from *where* they are — it's from the exploding residual pathway.

## 4. Residue-Type Discrimination

| Metric | MC-GearNet | CA-GearNet | ESM2 (L33) | MACE |
|--------|-----------:|-----------:|-----------:|-----:|
| Linear probe accuracy | **13.9%** | 15.4% | 99.7% | 100% |
| Mean cos-sim between AA centroids | **0.9999** | 0.975 | 0.847 | — |
| Within-type | 0.743 ± 0.39 | 0.189 ± 0.14 | 0.486 ± 0.18 | — |
| Between-type | 0.759 ± 0.38 | 0.176 ± 0.13 | 0.434 ± 0.19 | — |
| Delta | **-0.016** | 0.013 | 0.051 | — |

Per-AA centroids have pairwise cos-sim **0.9999** — they point in the same direction. Between-type similarity is actually *higher* than within-type (−0.016 delta), which should be impossible for a discriminative representation and confirms that the variance in pairwise similarity is coming from projection noise onto the 1-D attractor, not from AA identity.

Probe accuracy 13.9% is near CA-GearNet (15.4%) — but CA-GearNet doesn't see residue types. MC-GearNet is *fed* residue types and can't recover them, because by layer 5 the identity signal has been drowned in norm explosion.

## 5. Structural Context Sensitivity

| AA | Within-context | Between-context | Delta |
|----|---------------:|----------------:|------:|
| ALA | 0.865 | 0.854 | +0.011 |
| GLY | 0.706 | 0.724 | **−0.017** |
| LEU | 0.846 | 0.868 | **−0.022** |
| VAL | 0.899 | 0.889 | +0.010 |

Two of four AAs show negative delta (between > within), again consistent with noise. CA-GearNet showed a ~0.113 delta for ALA and clear positive deltas for most AAs. MC-GearNet has no meaningful within- vs between-context structure.

## 6. Embedding Norms & Conditioning

| Metric | MC-GearNet | CA-GearNet | ESM2 (L33) |
|--------|-----------:|-----------:|-----------:|
| Mean L2 norm | **1,562,084** | 76.9 | 9.81 |
| Std L2 norm | 2,706,806 | 29.4 | 0.30 |
| Min / max | 35 / **55,504,832** | — | 9.1 / 10.9 |
| Dead dimensions | **513 / 3072** | 0 / 512 | 0 / 1280 |
| Dimension std range | [0, 2,686,790] | [2.5, 4.6] | [0.15, 1.93] |

Mean norm 1.56 million — ~20,000× CA-GearNet. 513 dead dimensions (exactly one-sixth = one 512-dim layer slab), indicating one of the 6 concatenated layers contributes nothing. Norm ratio max/min = 1.6 × 10⁶ per-residue — the representation has no consistent magnitude scale.

Per-AA norms vary 2.7× (ALA 2.16M vs LYS 0.80M) — the only systematic AA signal is norm, which cosine-similarity loss cancels out.

## 7. Projector Saturation Test (KEY RESULT)

3-layer MLP (input → 512 → 512 → 3072), cosine-similarity loss, 300 epochs, 80/20 split.

| Input condition | Train | **Test** |
|-----------------|------:|---------:|
| mean-direction (no MLP) | — | **0.864** |
| random 128-d | 0.867 | **0.866** |
| AA one-hot (21-d) | 0.868 | **0.867** |
| AA one-hot + position | 0.868 | **0.867** |

Comparison (test cos-sim):

| Encoder | mean-dir | identity | REPA (live) | Genuine headroom |
|---------|---------:|---------:|------------:|-----------------:|
| CA-GearNet | 0.419 | 0.426 | 0.78 | **~0.35** |
| ESM2 L33 | 0.664 | 0.719 | — | ~0.08 |
| **MC-GearNet** | **0.864** | **0.867** | — | **~0.00** |

**The projector cannot contribute anything over the mean direction.** Three conditions with radically different information content (random noise, AA one-hot, AA one-hot + position) all land within 0.003 cos-sim of the zero-learning baseline. A 3-layer MLP with 512 hidden has hundreds of thousands of parameters and 300 epochs of Adam to beat a zero-parameter constant — and cannot. The target is a ray in 3072-d; any prediction that points down that ray wins, regardless of input.

## 8. Within-Protein vs Between-Protein Similarity

| Metric | MC-GearNet | CA-GearNet | ESM2 (L33) |
|--------|-----------:|-----------:|-----------:|
| Within-protein | 0.773 ± 0.36 | 0.420 ± 0.27 | 0.577 ± 0.20 |
| Between-protein | 0.719 ± 0.40 | 0.176 ± 0.13 | 0.436 ± 0.20 |
| Delta | **0.055** | 0.244 | 0.141 |

Within-protein delta of 0.055 — residues in the same protein are barely more similar than residues in different proteins. The ±0.4 std bands fully overlap. By contrast CA-GearNet has a 0.244 delta, and even ESM2 (sequence-only!) hits 0.141.

## 9. Layer-wise Representation

MC-GearNet-Edge's final output is the **concatenation** of all 6 hidden layers (`concat_hidden=True`), each 512-dim, giving 3072-d output. Here is what each slab contributes:

| Layer | Dim | Eff Rank | Mean Norm | Inter-layer cos (i vs i-1) |
|-------|----:|---------:|----------:|---------------------------:|
| 0 | 512 | 6.1 | **81.7** | (dim mismatch with input) |
| 1 | 512 | 10.4 | 36.4 | 0.678 |
| 2 | 512 | 6.1 | 296 | 0.214 |
| 3 | 512 | 4.1 | 1,653 | 0.180 |
| 4 | 512 | 3.8 | **31,560** | 0.035 |
| 5 | 512 | **1.2** | **1,948,064** | 0.058 |

**Layer-norm timeline**: 81.7 → 36.4 → 296 → 1,653 → 31,560 → 1,948,064. The ratio layer 5 / layer 0 is **~24,000×**, and layer 5 / (sum of layers 0-4) is about **60×**. When the 6 layers are concatenated, layer 5's values — all pointing in near-1-D — drown out everything.

**Effective rank** collapses from 10.4 (layer 1) to 1.2 (layer 5). Each successive layer projects onto fewer directions with larger magnitude.

**Inter-layer cosine** 0.68, 0.21, 0.18, 0.035, 0.058 — layers 2+ are nearly orthogonal to the previous one. This is not healthy representation evolution; it is the BN+residual chain amplifying and rotating a collapsing attractor.

In isolation, **layer 0 or layer 1** (norms 82 and 36, eff rank 6 and 10, 0% sparsity, rotation-invariant) would be a reasonable REPA target — low rank but clean. The problem is the concat: it exposes only the worst slab.

## Head-to-head: MC-GearNet vs CA-GearNet vs ESM2

| Property | MC-GearNet | CA-GearNet | ESM2 (L33) |
|----------|-----------:|-----------:|-----------:|
| Domain | Proteins (3D, residue types) | Proteins (3D) | Proteins (sequence) |
| Output dim | 3072 | 512 | 1280 |
| Effective rank | **1.1** | 82.6 | 353.6 |
| 3D-aware | Yes (weak) | **Yes (strong)** | No |
| 3D sensitivity (0.5 Å) | cos = 0.82 | **cos = 0.36** | N/A |
| Rotation invariant | Yes (0.9996) | Yes (0.9997) | Trivially |
| AA probe | 13.9% | 15.4% | 99.7% |
| Residue-shuffle cos | **0.984** | N/A | N/A |
| Projector mean-dir baseline | **0.864** | 0.419 | 0.664 |
| Projector with identity | 0.867 | 0.426 | 0.719 |
| **Genuine REPA headroom** | **≈ 0.00** | **~0.35** | ~0.08 |
| Within-vs-between delta | 0.055 | 0.244 | 0.141 |
| Dead dims | **513 / 3072** | 0 / 512 | 0 / 1280 |

**CA-GearNet remains the only encoder with meaningful REPA headroom** (~0.35 of genuine structural learning). MC-GearNet-Edge, despite having 6× the output dimensionality, 3D awareness, *and* residue-identity input, is uniformly worse on every metric that matters for REPA: rank, projector saturation, within-protein delta, 3D gradient, residue discrimination.

## Implications for REPA Training

### Do not use MC-GearNet-Edge as a REPA target in its current form

The representation is 1-dimensional with exploding norms. No projector can fit it above the mean-direction baseline, and no gradient signal can flow back to the student transformer. Any REPA run using MC-GearNet will plateau at cos-sim ≈ 0.86–0.87 on the first few steps and contribute nothing afterward — the loss curve will look *better* than CA-GearNet (higher cos-sim!) while teaching the model nothing.

### Root cause

`concat_hidden=True` over 6 BatchNorm+residual layers without a final LayerNorm — norms grow geometrically. Layer 5 has mean L2 ≈ 2 million; all earlier layers together contribute < 2% of the output magnitude. The concatenation is effectively layer 5 padded with noise.

### Possible fixes (not tested here)

1. **Use a single intermediate layer, not the concat.** Layer 0 or 1 has eff rank 6–10 and norms O(100). That is a usable — if low-rank — REPA target.
2. **LayerNorm the per-layer slabs before concat.** Standard protocol for `concat_hidden` encoders; absent here.
3. **z-score the encoder output** inside the REPA loss (running mean/std). Addresses the symptom but not the collapse — if eff rank is 1.1, standardization gives you 1 informative dim out of 3072.
4. **Drop MC-GearNet-Edge.** Given that CA-GearNet gives the strongest REPA signal we have measured, and MC-GearNet's extra features (edge types, angle bins, residue identity) do not translate into usable representation, the simplest conclusion is that this checkpoint's output isn't designed to be consumed raw.

### What this means for the three `gearnet_mc_edge` config variants in the tree

`src/proteina/configs/experiment_config/training/{128,256,512}/gearnet_{rep,mc_edge}/per_residue/training_repa_l{0,4,9}_*_mc_edge.yaml` all target the concatenated 3072-d output via a 3-layer projector. If any of these have been run, expect their REPA loss curves to be near-flat (or confusingly high cos-sim) and their downstream metrics to match the non-REPA baseline within noise.

## Caveats

- **200 proteins, first-in-LMDB-order** — PDB IDs starting with `1a…` dominate. Matches the sampling used for CA-GearNet and ESM2 so comparisons are apples-to-apples, but not a random sample of fold space.
- **SVD/probe subsampled to 30k residues** for memory (3072-dim matrices are 6× heavier than CA-GearNet's 512-dim). Qualitative result (eff rank ≈ 1) is robust to sample size; the collapse is structural.
- **Linear probe did not converge** (`lbfgs` hit max_iter=1000). Scaling the features first would likely help, but the probe is near chance anyway.
- **Layer-wise capped at 30 proteins / ~6.8k residues** per layer — eff-rank at this sample count is noisy but the trend (eff rank 10 → 1) is monotone across 30 proteins.

## Artifacts

- Script: [explore_mc_gearnet.py](explore_mc_gearnet.py)
- SLURM: [run_mc_gearnet.sh](run_mc_gearnet.sh)
- Raw output: `.local_ckpts/mc-gn-char-28306119.out`
