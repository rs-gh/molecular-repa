# Does REPA Transfer Beyond Vision? An Empirical Analysis

**Authors:** Shreyas Ravishankar
**Date:** April 2026
**Status:** Working document

## 1. Motivation

Representation Alignment (REPA) (Yu et al., 2024; arXiv:2410.06940) demonstrated
that aligning a diffusion transformer's hidden states with frozen DINOv2
representations dramatically improves image generation quality and training
efficiency on ImageNet. The core mechanism is simple: a trainable MLP projects the
denoiser's intermediate hidden states into the encoder's embedding space, and a
cosine similarity loss encourages alignment. The encoder is frozen; only the
projector and the denoiser receive gradients from the alignment loss.

This project investigated whether REPA transfers to two scientific domains:
unconditional 3D small-molecule generation (Tabasco, a flow matching model on
GEOM-drugs) and protein backbone generation (Proteina, a flow matching model on
PDB). The short answer is: **no clear improvement**, and this document presents
the evidence for why, together with a structural argument about what made REPA
work in vision and what is absent in our domains.

## 2. Experimental Setup

### 2.1 Tabasco (Small Molecules)

- **Generative model:** Tabasco flow matching transformer on GEOM-drugs
  (1,142,099 molecules, batch size 256, 4,461 steps/epoch)
- **Encoders tested:**
  - **CheMeleon** (ChemProp foundation model): 2D message-passing network
    pretrained on 1M PubChem molecules; output dim 2048
  - **MACE-OFF small**: 3D equivariant encoder pretrained on molecular
    energies/forces; output dim 192
- **Projector:** 2-layer MLP (hidden_dim=128, matching `model.net.hidden_dim`)
- **REPA weight:** lambda=0.5, both `additive` (FM + 0.5*REPA) and `tradeoff`
  ((1-0.5)*FM + 0.5*REPA) modes
- **Alignment layer:** Final transformer hidden state
- **Evaluation:** 1,000 molecules, 100 Euler steps; metrics include validity,
  PoseBusters bond geometry, FCD, diversity, novelty

### 2.2 Proteina (Proteins)

- **Generative model:** Proteina 60M-parameter CAFlow transformer on PDB
  (579k structures, batch size 4-6, max length 512 residues)
- **Encoder:** GearNet CA-only (8-layer GNN, 512-dim, frozen, pretrained on
  protein structures)
- **Projector:** 2-layer MLP (hidden_dim=512)
- **REPA weight:** lambda=0.5, additive mode
- **Alignment layers tested:** Layer 0 (first), layer 4 (middle), layer 9 (last),
  and all layers simultaneously
- **Evaluation:** FID and feature Jensen-Shannon divergence (fJSD) on PDB/AFDB
  reference distributions; 6,125 generated proteins per evaluation

### 2.3 Key Difference from Original Paper

The original REPA paper uses DINOv2-Large as the frozen encoder and evaluates on
ImageNet FID. Both the encoder and the evaluation are grounded in the same
well-understood distribution (natural images), and DINOv2 achieves >80% linear
probing accuracy on ImageNet classification. No analogous encoder-evaluation
pairing exists for molecules or proteins.

## 3. Results

### 3.1 Tabasco: Small Molecule Generation (GEOM-drugs)

All REPA models trained ~15 epochs (~73k steps); baseline trained 33 epochs
(~152k steps). All models ran for approximately 16 wall-clock hours on A100.

| Model | Validity | Bond Lengths | Bond Angles | Steric Clash | FCD (lower=better) | Novelty | Diversity |
|-------|----------|-------------|-------------|--------------|-----|---------|-----------|
| **Baseline** | 0.980 | 0.974 | 0.961 | 0.933 | **5.61** | 0.966 | **0.886** |
| CheMeleon additive | 0.972 | 0.959 | 0.952 | 0.920 | 7.43 | 0.961 | 0.882 |
| CheMeleon tradeoff | 0.976 | 0.958 | 0.954 | 0.916 | 6.49 | 0.964 | 0.883 |
| MACE additive | **1.000** | 0.977 | **0.973** | 0.921 | 6.81 | 0.977 | 0.884 |
| MACE tradeoff | **1.000** | **0.982** | 0.972 | **0.941** | 6.32 | **0.982** | 0.883 |

**Interpretation.** MACE-REPA achieves perfect validity (1.000 vs 0.980) and
marginally better local geometry (bond lengths, angles). However, the
distributional fidelity metric FCD worsens by 13-21% (5.61 to 6.32-6.81), and
diversity decreases slightly. CheMeleon-REPA is worse than baseline on nearly
every metric. The MACE result represents a trade-off — better local structure at
the cost of worse distributional fit — not a clear improvement.

**Training speed.** REPA models converge in fewer epochs but are ~2x slower per
step (0.78 s/step vs 0.38 s/step for baseline), resulting in roughly equal
wall-clock time for similar-quality outputs.

| Model | s/step | steps/hr | GPU util |
|-------|--------|----------|----------|
| Baseline | 0.376 | 9,567 | 59.3% |
| CheMeleon | 0.737 | 4,884 | 56.7% |
| MACE cached | 0.780 | 4,618 | 30.1% |

### 3.2 Proteina: Protein Backbone Generation (PDB)

A critical complication: the baseline overfits. FID peaks at epoch 7 (518.4) and
degrades to 648.0 by epoch 10. REPA models were evaluated at epoch 3 (420k
steps). We present results at both the baseline's final checkpoint (epoch 10) and
its peak (epoch 7).

| Model | Steps | Epoch | PDB FID | fJSD_C | fJSD_A | fJSD_T |
|-------|-------|-------|---------|--------|--------|--------|
| **Baseline (peak)** | 535,500 | 7 | **518.4** | **0.218** | **0.830** | **2.660** |
| Baseline (final) | 742,000 | 10 | 648.0 | 1.288 | 1.567 | 3.794 |
| REPA layer 0 | 420,000 | 3 | 599.1 | 0.842 | 1.394 | 3.149 |
| REPA layer 4 | 420,000 | 3 | 657.0 | 1.075 | 1.570 | 3.445 |
| REPA layer 9 | 420,000 | 3 | 879.2 | 0.094 | 1.152 | 3.951 |

**Interpretation.** Against the overfitted baseline (epoch 10, FID 648), REPA
layer 0 looks promising (FID 599). But against the properly early-stopped
baseline (epoch 7, FID 518), no REPA variant is competitive. Layer 9 (last layer)
alignment is catastrophic (FID 879). The fJSD_C value of 0.094 for layer 9
suggests near-total collapse of fold-class diversity.

**Caveats on fairness.** REPA models were evaluated at epoch 3 only; they may
improve with further training. The batch size is also smaller (B=4 vs B=6) due to
GearNet memory overhead (~10 GB), meaning REPA models see 33% fewer samples per
step. These are real confounds, but the overall picture does not suggest a strong
positive signal.

## 4. Encoder Characterisation

A central question is whether the available encoders are good enough to serve as
REPA alignment targets. We systematically characterised all three encoders along
dimensions relevant to REPA.

### 4.1 Representation Quality

| Property | GearNet (proteins) | MACE (molecules) | CheMeleon (molecules) | DINOv2 (vision) |
|----------|-------------------|-------------------|----------------------|-----------------|
| Output dim | 512 | 192 | 2048 | 1024 |
| Exact zero fraction | 0.0% | 0.0% | **93.8%** | ~0% |
| Effective rank | 82.6 | 40.6 | 138 | — |
| 3D sensitivity (0.5A) | **cos=0.36** | cos=0.998 | cos=1.000 (2D-blind) | N/A |
| Linear probe (identity) | 15.4% | 100% | 100% | >80% (ImageNet) |
| Within-entity similarity | 0.420 | 0.706 | 0.245 | — |
| Between-entity similarity | 0.176 | 0.447 | 0.152 | — |

**Key observations:**

- **CheMeleon is 2D-only.** Identical embeddings for all 3D conformers of the same
  molecule (L2 distance = 0.000 across 484 conformer pairs with RMSD 0.03-4.19A).
  It fundamentally cannot guide 3D geometry learning.
- **CheMeleon is extremely sparse.** 93.8% exact zeros, only 4-7 dimensions active
  >50% of the time. This creates noisy cosine similarity gradients.
- **MACE is weakly 3D-sensitive.** Cosine similarity of 0.998 between conformers
  with 1.4-3.7A RMSD. The 3D signal exists but is subtle.
- **GearNet is strongly 3D-sensitive.** Cosine similarity drops to 0.36 at just
  0.5A perturbation. It encodes genuine structural context.
- **GearNet has weak chemical discrimination.** Only 15.4% linear probe accuracy
  for amino acid type (chance ~5%). It encodes *where* a residue is, not *what* it
  is — the opposite of MACE/CheMeleon, which encode atom types perfectly.

### 4.2 Projector Saturation: Is the Transformer Contributing?

The projector saturation test asks: how much of the REPA cosine similarity is due
to the transformer learning meaningful representations, versus the projector
simply memorising type-level prototypes? We train a standalone MLP on simple
inputs (random vectors, one-hot identity) and compare to REPA training values.

| Encoder | Mean floor | Identity (test) | REPA training | Gap (Identity → REPA) |
|---------|-----------|----------------|---------------|----------------------|
| **GearNet** | 0.419 | 0.426 | **0.78** | **+0.354** |
| **CheMeleon** | 0.388 | 0.455 | 0.66-0.68 | +0.21 |
| **MACE** | 0.755 | 0.861 | 0.56 | -0.30 (confounded; see below) |

*Identity = one-hot atom/residue type. Mean floor = cosine similarity to the mean
embedding (achievable with a constant output). All test-set values; 80/20
stratified split.*

**GearNet** has the largest positive gap: identity barely predicts its embeddings
(0.426 ~ floor 0.419), yet REPA training reaches 0.78. The transformer learns
0.354 of genuine structural alignment beyond type identity.

**CheMeleon** shows a moderate gap (0.21). Roughly two-thirds of the alignment is
attributable to atom type alone.

**MACE** shows a negative gap, but this comparison is confounded: the identity
baseline is evaluated on clean molecules, while REPA training averages cosine
similarity across all noise levels (including t~0 where the denoiser receives
near-pure noise). MACE-REPA does achieve perfect validity, suggesting the
alignment provides *some* signal despite the low aggregate cosine similarity.

### 4.3 Fast-Path vs Slow-Path Divergence (CheMeleon-specific)

CheMeleon's SMILES-based fast path (used during cached training) and the
bond-inference slow path produce fundamentally different molecular graphs:

| Comparison | Value |
|-----------|-------|
| Mean cosine similarity (fast vs slow path) | 0.23 |
| Molecules with cosine sim < 0.99 | 100/100 (100%) |
| Aromatic molecules | 0.28 |
| Non-aromatic molecules | 0.22 |

This divergence affects all molecules, not just edge cases. The two code paths
produce qualitatively different encodings due to different bond perception
algorithms (RDKit `MolFromSmiles` vs `DetermineConnectivity`).

## 5. Why REPA Worked in Vision

The original REPA paper's success rests on three pillars, all of which are absent
or degraded in our domains.

### 5.1 An Oracle-Quality Encoder

DINOv2 is trained on 142M images with self-supervised objectives and achieves
>80% linear probing accuracy on ImageNet. Its representations are essentially
solved for natural-image semantics. When a diffusion transformer aligns to DINOv2,
it aligns to a representation that provably captures the target distribution's
semantic structure.

**In our domains:** No encoder approaches this quality level. GearNet achieves
15.4% amino acid classification (3x chance); MACE achieves 100% atom type
classification but with weak 3D sensitivity. CheMeleon is 2D-blind. None of these
encoders can be considered oracle-quality representations of the data they encode.

### 5.2 Alignment Between Encoder Strength and Evaluation Metric

In vision, FID on ImageNet directly measures distributional fidelity to a dataset
where DINOv2's representations are strongest. The encoder and the evaluation
metric are aligned by construction: improving alignment with DINOv2 directly
improves the features that FID measures.

**In our domains:** There is no canonical "representation quality" benchmark.
For molecules, FCD measures distributional similarity using a separate neural
network; for proteins, FID uses GearNet-derived features. Neither metric has a
direct relationship to the REPA encoder's embedding space. Improving alignment
with MACE may not improve FCD; improving alignment with GearNet may not improve
protein FID (despite GearNet being used in the FID computation — the FID operates
on globally-pooled features, while REPA operates per-residue).

### 5.3 A Clear Representation Gap to Close

The REPA paper demonstrated (Figure 3) that diffusion model hidden states have
weak but non-zero alignment with DINOv2 that improves with training — a
measurable gap between what the model learns naturally and what the encoder
provides. With REPA, this gap closes faster: the semantic gap starts small in
early layers and grows in later layers, freeing them to focus on generation.

**In our domains:** The projector saturation tests (Section 4.2) suggest that for
MACE, the gap may not exist in a meaningful sense: atom type identity alone
explains most of the alignment (0.861 vs 0.755 floor). For GearNet, a genuine
gap exists (0.354), but the encoder's weak chemical discrimination (15.4% linear
probe) means the alignment signal is primarily geometric — useful in principle,
but perhaps too indirect to help a generative model that must learn both geometry
and chemistry simultaneously.

## 6. Additional Complicating Factors

### 6.1 Conformer Invariance Problem

REPA aligns denoiser hidden states (at noisy intermediate x_t) with encoder
representations of the clean target (x_1). For 3D generation, the encoder must
be sensitive to 3D conformation. CheMeleon is completely conformer-blind
(cos_sim = 1.000 between conformers). MACE is nearly so (cos_sim = 0.998).
Only GearNet has strong conformer sensitivity (cos_sim = 0.362 at 0.5A
perturbation). REPA fundamentally requires a 3D-sensitive encoder to guide 3D
generation — and two of our three encoders fail this requirement.

### 6.2 Baseline Metric Saturation (Tabasco)

The Tabasco baseline already achieves 98.0% validity, 97.4% bond lengths, and
96.1% bond angles on GEOM-drugs. There is limited room for improvement, making
it difficult to detect a positive effect even if one exists. The FCD metric,
where more room exists, worsens with REPA.

### 6.3 Baseline Overfitting (Proteina)

The Proteina baseline overfits after epoch 7: PDB FID degrades from 518.4 to
648.0 (+25%), and fold-class diversity collapses (fJSD_C from 0.218 to 1.288,
a 6x degradation). This complicates interpretation of REPA results evaluated at
epoch 3. A fair comparison requires either (a) evaluating REPA at its own peak,
or (b) comparing to the baseline at the same step count. Neither comparison
currently shows a clear REPA advantage.

### 6.4 Computational Overhead

REPA models are ~2x slower per step due to encoder forward passes (even with
caching). For Proteina, the encoder memory overhead forces a batch size reduction
from 6 to 4 (33% fewer samples per gradient step). This means REPA must provide
a substantial per-sample benefit to compensate for the throughput penalty.

## 7. Summary

| Dimension | Vision (original REPA) | Molecules (Tabasco) | Proteins (Proteina) |
|-----------|----------------------|---------------------|---------------------|
| Encoder quality | DINOv2 (oracle, >80% linear probe) | MACE (100% atom type, weak 3D) / CheMeleon (2D-blind, 94% sparse) | GearNet (15% AA type, strong 3D) |
| Evaluation-encoder alignment | Direct (FID measures what DINOv2 captures) | Indirect (FCD uses separate network) | Partial (FID uses GearNet, but globally-pooled vs per-residue REPA) |
| Conformer sensitivity | N/A (2D images) | Weak (MACE 0.998) / None (CheMeleon 1.000) | Strong (GearNet 0.362 at 0.5A) |
| Projector signal | Large gap (encoder >> identity) | Small or negative gap | Large gap (+0.354) |
| Generation improvement | Yes (state-of-the-art FID) | No (FCD worsens 13-21%) | Inconclusive (confounded by training duration, batch size) |
| Training efficiency gain | Yes (fewer iterations to converge) | No (2x slower/step negates faster convergence) | Unknown (insufficient evaluation points) |

## 8. Conclusions and Future Directions

The evidence suggests that REPA's success in vision relies on a confluence of
factors — an oracle-quality encoder, alignment between encoder strength and
evaluation metric, and a clear representation gap — that do not currently exist
in molecular or protein generation.

This is not simply a matter of needing a better encoder. Even GearNet, which has
strong 3D sensitivity and a genuine projector signal gap, does not produce clear
generation improvements. The problem may be more fundamental: scientific domains
lack the scale of self-supervised pretraining data and the canonical
classification benchmarks that make DINOv2 an effective REPA target.

**Possible next steps, if pursuing this direction:**

1. **Longer Proteina training with periodic FID evaluation.** GearNet has the
   strongest theoretical case for REPA (large projector gap, strong 3D
   sensitivity). The current 3-epoch evaluation is insufficient to draw
   conclusions. Training to epoch 10+ with FID every 50k steps would clarify
   whether REPA helps, hurts, or acts as a regulariser against overfitting.

2. **Simpler auxiliary losses.** If the goal is to inject structural knowledge,
   atom/residue-type classification from hidden states is a simpler and cheaper
   alternative that avoids the encoder quality problem entirely.

3. **PCA-reduced alignment space.** MACE has effective rank 40.6 in 192
   dimensions; CheMeleon has effective rank 138 in 2048. Projecting to the
   principal subspace before computing cosine similarity would address the
   projector saturation issue.

4. **3D-aware molecular encoder.** For small molecules, the fundamental problem
   is that no available encoder combines strong 3D sensitivity with dense
   gradients. A 3D GNN pretrained on conformational energies (e.g., SchNet or
   TorchMD-NET) might be a better REPA target than MACE or CheMeleon.

5. **Writing up the negative result.** "REPA works because DINOv2 is an oracle;
   no such oracle exists for molecular generation" is a finding that saves the
   community time and identifies a concrete prerequisite for REPA-style
   approaches in new domains.

## Appendix: Data Sources

All data referenced in this document is available in the repository:

| Data | Location |
|------|----------|
| Tabasco evaluation metrics | `evaluation/tabasco/generation/results/geom/evaluation/evaluation_summary.csv` |
| Tabasco training performance | `evaluation/tabasco/generation/results/geom/training_performance/training_performance.csv` |
| Tabasco training run log | `docs/tabasco_training_runs.md` |
| Proteina FID results (baseline) | `evaluation/proteina/generation/results/pdb/fid/inference_fid_60m_baseline.csv` |
| Proteina FID results (REPA L0) | `evaluation/proteina/generation/results/pdb/fid/inference_fid_60m_repa_layer0.csv` |
| Proteina FID results (REPA L4) | `evaluation/proteina/generation/results/pdb/fid/inference_fid_60m_repa.csv` |
| Proteina FID results (REPA L9) | `evaluation/proteina/generation/results/pdb/fid/inference_fid_60m_repa_layer9.csv` |
| Proteina baseline overfitting analysis | `playground/proteina/baseline_overfitting/baseline_training_analysis.md` |
| CheMeleon encoder characterisation | `encoder_profiling/tabasco/chemeleon/FINDINGS.md` |
| MACE encoder characterisation | `encoder_profiling/tabasco/mace/FINDINGS.md` |
| GearNet encoder characterisation | `encoder_profiling/proteina/gearnet/FINDINGS.md` |
| Projector saturation analysis | `playground/projector/FINDINGS.md` |
| REPA paper notes | `docs/notes.md` |

## References

- Yu et al. (2024). Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think. arXiv:2410.06940.
- Tabasco: Flow matching model for 3D molecular generation.
- Proteina: Yim et al. (2025). Scalable Protein Structure Generation. arXiv:2503.00710.
- DINOv2: Oquab et al. (2024). DINOv2: Learning Robust Visual Features without Supervision.
- CheMeleon: ChemProp foundation model. arXiv:2506.15792.
- MACE-OFF: Batatia et al. (2024). A foundation model for atomistic simulation.
- GearNet: Zhang et al. (2023). Protein representation learning by geometric structure pretraining.
