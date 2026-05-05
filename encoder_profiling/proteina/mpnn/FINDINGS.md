# ProteinMPNN CA-only Encoder Characterization

**Date**: 2026-05-05
**Encoder**: ProteinMPNN CA-only, weights `v_48_020.pt` (`ProteinMPNNPerResidueEncoder`, 3 encoder layers, 128-dim, k=48, augment_eps=0)
**Pretraining**: Supervised autoregressive inverse folding (predict residue identity given backbone), Dauparas et al. 2022.
**Data**: 200 PDB train proteins (42,034 residues)
**Probe run**: [results/20260505_190822/results.json](results/20260505_190822/results.json), [layerwise.csv](results/20260505_190822/layerwise.csv).
**Cross-encoder context**: [../FINDINGS.md](../FINDINGS.md) — for the three-question framing (Q1 information, Q2 saturation, Q3 conditioning) used throughout.

## Summary

ProteinMPNN-CA is a **conditioning-clean but headroom-narrow** REPA target. Q3 is healthy across the board (0% sparsity, 0 dead dims, RankMe 84.9 / 128 = 66% utilisation, tight norms σ=0.20). Q1 is dominated by what its inverse-folding objective demands: it carries **moderate residue-identity information** (probe 0.327, ~3× CA-GearNet's 0.137) without seeing residue-type input — a leakage from the inverse-folding pretraining task. But the encoder is **less geometrically discriminative than CA-GearNet** at every length scale (1 Å noise → cos 0.726 vs CA-GN 0.269), and **per-protein specificity is weak** (Δ within−between = 0.022 vs CA-GN 0.222). The Q2 saturation gap is **+0.014** — slightly above CA-GearNet's +0.009 but on a much higher absolute floor (mean-dir 0.835 vs 0.425), so the embedding space is a much tighter "ball" than CA-GearNet's.

What this means for REPA: the optimisation conditioning is fine, but the geometric and protein-context signal that REPA can teach is weaker than CA-GearNet's. The clearest contrast is layer-wise — ProteinMPNN's representation **norm collapses through depth** (7.63 → 4.38 → 3.13) while CA-GearNet's grows monotonically (19.8 → 78.9). The inverse-folding objective compresses; the CATH-classification objective spreads.

## Q1. What information does the encoder encode?

### 1.1 Residue identity

| Metric | Value |
|--------|-------|
| Linear probe accuracy | **32.7%** (chance ~5%; CA-GN trained 13.7%) |
| Mean cos between AA centroids | 0.966 |
| Within-type cos | 0.722 ± 0.085 |
| Between-type cos | 0.697 ± 0.090 |
| Δ (within − between) | **0.025** |
| Residue-shuffle cos (coords fixed, labels permuted) | N/A — encoder does not consume residue type |

ProteinMPNN-CA carries **non-trivial residue-identity information without ever seeing a sequence label** at probe time — its linear-probe accuracy (32.7%) is more than 2× CA-GearNet's (13.7%). This is a direct fingerprint of the inverse-folding pretraining objective: the encoder was optimised to make `h_V` linearly predictive of residue identity (its decoder is a linear-ish head from `h_V` over 21 amino acids). So features that disambiguate AA classes are baked into the pretrained representation, even though they're not in the input. Per-AA centroids are still nearly co-linear (cos 0.966), so this is not a dramatic separation in cosine space — but it's there.

### 1.2 3D geometric sensitivity

| Perturbation | Cosine similarity | (CA-GearNet for comparison) |
|--------------|-------------------|---------------------------:|
| 0.1 Å Gaussian noise | 0.996 ± 0.0006 | 0.933 |
| 0.5 Å Gaussian noise | **0.904 ± 0.014** | 0.367 |
| 1.0 Å Gaussian noise | 0.726 ± 0.026 | 0.269 |
| 2.0 Å Gaussian noise | 0.639 ± 0.024 | 0.187 |
| 5.0 Å Gaussian noise | 0.601 ± 0.020 | −0.002 |
| Random rotation | **0.99999 ± 3 × 10⁻⁶** | 0.99968 |

ProteinMPNN-CA is **substantially less coordinate-sensitive than CA-GearNet** at every perturbation scale. At sub-Angstrom noise (0.5 Å, comparable to thermal fluctuations) the cosine drop is from 1.0 → 0.90, vs CA-GearNet's 1.0 → 0.37. At 5 Å noise, CA-GearNet is fully decorrelated (cos ≈ 0); ProteinMPNN still sits at 0.60 — its embedding floor is the global mean direction (Q2: mean-dir cos = 0.835, similar magnitude). Rotation invariance is essentially perfect (better than CA-GearNet's). Implication: the encoder produces a *smoother* representation of geometry — a lot of coordinate variation rounds to the same `h_V`, because the inverse-folding objective requires the encoder to be invariant to geometric details that don't change residue identity.

### 1.3 Structural context (helix / sheet / loop)

CA-CA-CA bond angle as a coarse SS proxy:

| AA  | within-SS cos | between-SS cos | Δ      | (helix / sheet / loop) |
|-----|--------------:|---------------:|-------:|------------------------|
| ALA | 0.723 | 0.699 | 0.024 | 2197 / 781 / 480 |
| GLY | 0.665 | 0.632 | 0.033 | 1496 / 740 / 832 |
| LEU | 0.753 | 0.728 | 0.025 | 2264 / 1166 / 490 |
| VAL | 0.757 | 0.726 | 0.032 | 1122 / 1335 / 540 |

Per-AA SS sensitivity is comparable in magnitude to CA-GearNet (Δ ~ 0.02–0.04). GLY and VAL show slightly larger deltas than ALA/LEU. SS information is present but not the dominant axis.

### 1.4 Protein-level identity (within-protein vs between-protein)

| Metric | Value | (CA-GearNet for comparison) |
|--------|------:|---------------------------:|
| Within-protein cos | 0.721 ± 0.098 | 0.416 |
| Between-protein cos | 0.698 ± 0.088 | 0.195 |
| **Δ** | **0.022** | **0.222** |

This is the most striking divergence from CA-GearNet. Residues from the same protein are barely more similar than residues from different proteins (Δ = 0.022). CA-GearNet has **10× the protein-specificity signal** (Δ = 0.222). Interpretation: ProteinMPNN's per-residue features describe a *local* environment (what AA is plausible at this site given the surrounding backbone) — it is not concerned with making residues belonging to the same protein "look like each other." That's the inverse-folding bias: locality wins, global fingerprint loses.

## Q2. How much is reachable from cheap inputs?

3-layer MLP from each input condition to the encoder embedding, 80/20 train/test split, 300 epochs.

| Input condition | Train cos | Test cos |
|-----------------|----------:|---------:|
| Mean direction (no MLP) | — | **0.835** |
| Random 128-d | 0.839 | 0.833 |
| AA one-hot (21-d) | 0.850 | 0.850 |
| AA one-hot + sin/cos position | 0.850 | **0.850** |

**Saturation gap = best − mean-dir = +0.014**. Slightly larger than CA-GearNet's +0.009 but interpreted differently — it sits on a much higher *absolute* floor (mean-dir 0.835 vs CA-GN 0.425), reflecting how tightly clustered the embeddings are. The cheap-input projector basically nails it; richer inputs can lift cosine by ~0.014. AA one-hot alone explains the entire gap; adding position contributes nothing — confirming the encoder produces representations dominated by per-residue identity-relevant features, not positional or contextual ones.

## Q3. Is the encoder a tractable optimisation target?

### 3.1 Sparsity & value distribution

| Metric | Value |
|--------|-------|
| Exact zeros | 0.00% |
| Near-zero (<1e-6) | 5.9 × 10⁻⁶% |
| Negative values | 50.4% |
| Mean | 0.0165 |
| Std | 0.277 |
| Min / Max | −1.58 / 2.69 |

Fully dense, ~50% negative — symmetric around zero. **No gradient sparsity, no dead path** for the cosine loss. By far the cleanest distribution profile of any structural encoder we've probed (CheMeleon was 93% sparse on ReLU; MC-GearNet-Edge had norm explosion; CA-GearNet has 62% negative on LeakyReLU but std 3.6).

### 3.2 Effective dimensionality & singular values

| Metric | Value |
|--------|-------|
| Output dim | 128 |
| **RankMe** | **84.9** |
| Participation ratio | 33.3 |
| Dims for 90% variance | 72 |
| Dims for 95% variance | 92 |
| Dims for 99% variance | 117 |
| Top singular value (centered) | 99.2 |
| S[0] / S[−1] (centered) | 5.5 × 10⁶ |

RankMe 84.9 / 128 = 66% utilisation. By absolute fraction this is *higher* than CA-GearNet's 256 / 512 = 50% — the smaller embedding is more uniformly used. PR 33 is comparable to CA-GearNet's 38 in absolute terms (despite the smaller output dim). 95% of variance fits in 92 dims, well within the 128-d projector — **no bottleneck risk**.

### 3.3 Norms & dead dimensions

| Metric | Value | (CA-GearNet) |
|--------|------:|------------:|
| Mean L2 norm | **3.13** | 79.6 |
| Std L2 norm | **0.20** | 32.0 |
| Min / Max L2 | 2.87 / 6.53 | 26.3 / 395.5 |
| Dead dimensions | **0 / 128** | 0 / 512 |
| Dim std range | [0.075, 0.369] | [2.64, 4.82] |

Tight, well-conditioned, and far smaller in magnitude than CA-GearNet's. Std/mean ratio is 0.06 (CA-GN: 0.40) — extremely concentrated norm distribution. Per-AA mean norms span ALA 3.12 to GLY 3.39 — Gly slightly higher (more conformational flexibility translating to a slightly larger feature norm). No norm explosion.

## Layer-wise representation

From [layerwise.csv](results/20260505_190822/layerwise.csv) (30 proteins, 3 encoder layers):

| Layer | RankMe | Mean norm | Sparsity |
|------:|-------:|----------:|---------:|
|     0 |   89.0 |      7.63 |     0.00 |
|     1 |   76.0 |      4.38 |     0.00 |
|     2 |**84.7**|      3.13 |     0.00 |

Counter-intuitive shape: norm **decreases monotonically** through depth (7.63 → 4.38 → 3.13), with a small RankMe dip at layer 1 then partial recovery at layer 2. Compared to CA-GearNet's norm-grows + RankMe-peaks-at-L5 shape, ProteinMPNN is **compressing through depth**. This is consistent with a contractive map toward the inverse-folding decision manifold: the late representation has less variance because the network is funnelling backbone variation into AA-identity-relevant features. **Implication**: aligning REPA to layer 0 might give a slightly more spread, less-saturated target than the standard layer-2 readout — worth a short ablation if first-layer alignment is being explored later.

## Implications for REPA training

1. **What REPA can teach with ProteinMPNN-CA:** identity-relevant local features. The encoder behaves like a soft, structure-derived AA classifier — its embeddings cluster around 21-ish modal directions, and the headroom is "what can a richer input add to this AA-conditional view." This is meaningfully different from CA-GearNet, which teaches geometry + protein-specific identity.

2. **Headroom is narrow but not collapsed (Q2 +0.014).** Slightly above CA-GearNet's +0.009 — but on a much tighter base. The student can plausibly lift `cos_sim_layer_4` from ~0.85 to ~0.86–0.87. A run that plateaus at the floor will tell us REPA contributed nothing the projector wouldn't already.

3. **Conditioning is the strongest of any 3D-aware encoder we've probed.** No sparsity, no dead dims, tight bounded norms. Optimisation should be smooth.

4. **Low protein-specificity (Q1.4 Δ = 0.022) is a real concern.** The "global fingerprint" signal that helps generation be self-consistent at the whole-protein level is weak here — a property CA-GearNet has and ProteinMPNN doesn't. Empirical question whether this matters for the n=128 / n=256 scales we run.

5. **Compare against CA-GearNet at matched checkpoint range.** Same data, same projector depth (3), same layer (4). Runs in flight: `proteina_60m_repa_mpnn_l4_128_per_residue_bs80` (job 28897711), `proteina_60m_repa_mpnn_l4_256_per_residue` (job 28898082).

For how this fits among the rest of the encoder shortlist see [../FINDINGS.md](../FINDINGS.md).
