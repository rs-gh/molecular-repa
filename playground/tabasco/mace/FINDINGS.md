# MACE Encoder Investigation: Findings

**Date**: 2026-03-19
**Scripts**: `playground/tabasco/mace/explore_mace.py`, `playground/tabasco/mace/generate_figures.py`
**Figures**: `playground/tabasco/mace/figures/`
**Related**: `playground/tabasco/chemeleon/FINDINGS.md` (CheMeleon investigation)

## Background

Following the CheMeleon investigation (2026-03-18) which identified fundamental limitations of a 2D encoder for REPA alignment in 3D molecular generation, we evaluated MACE-OFF as a 3D-aware alternative encoder. MACE is an equivariant neural network pretrained on molecular energies and forces.

### Motivation (from CheMeleon findings)
1. CheMeleon produces **identical** embeddings for all conformers (L2 = 0.0)
2. 93.8% sparsity from ReLU — noisy cosine similarity gradients
3. Projector bottleneck: 128-dim input for 500 effective rank target
4. 2D encoder cannot provide 3D geometric guidance

### Model used
- **MACE-OFF23 small** (`mace_off("small")`)
- 2 interaction layers, l_max=0 (all features are rotation-invariant scalars)
- 96 invariant features per layer, **192 total** output dimensions
- Pretrained on molecular energies/forces (organic molecules)

---

## Key Results

### 1. Zero sparsity — dense gradients (fig_01)

|                        | MACE   | CheMeleon |
|------------------------|--------|-----------|
| Exact zero fraction    | 0.0%   | 93.2%     |
| Near-zero (<1e-6)      | 0.0%   | 93.2%     |

MACE uses no ReLU in its output, producing dense feature vectors with a roughly Gaussian value distribution centered near 0. This eliminates the sparse gradient problem that plagued CheMeleon alignment.

### 2. Compact representation with low effective rank (fig_02, fig_05)

|                          | MACE   | CheMeleon |
|--------------------------|--------|-----------|
| Output dimension         | 192    | 2048      |
| Effective rank           | 40.6   | 138.1     |
| Dims for 90% variance   | 7      | ~100      |
| Dims for 99% variance   | ~30    | ~300      |

MACE's representation is much more compact: 94% of variance in just 10 dimensions. While the effective rank (40.6) is lower than CheMeleon's (138.1), this is actually advantageous:
- The projector (192-dim target) easily covers the full feature space
- No bottleneck problem: projector dim > effective rank
- Denser information per dimension = stronger gradient signal

### 3. Conformer sensitivity — geometry-aware but subtle (fig_03)

| Metric                              | MACE          | CheMeleon |
|-------------------------------------|---------------|-----------|
| Mean cosine sim between conformers  | 0.998         | 1.000     |
| L2 distance between conformers      | 0.08 - 0.29   | 0.000     |
| Coordinate RMSD range               | 1.4 - 3.7 A   | N/A       |

MACE does distinguish conformers (cos < 1.0), confirming it encodes 3D geometry. However, the sensitivity is subtle — conformer pairs with RMSD up to 3.7A still have cos > 0.995. This is expected: MACE-OFF was trained on energies, and local chemical environments (which dominate the energy) are similar across conformers. The global conformation (torsion angles, extended structure) has less impact.

**Implication**: MACE provides a **weak but nonzero** 3D signal for REPA alignment. It can guide the model toward correct local geometry (bond lengths, angles) but may not strongly distinguish global conformations.

### 4. Hydrogen atoms are invisible to heavy-atom embeddings (fig_06)

| Metric                                      | Value   |
|----------------------------------------------|---------|
| Cosine sim (all-atom vs heavy-only embedding) | 1.0000  |
| Min cosine sim across all atoms tested        | 1.0000  |

Removing hydrogen atoms has **zero effect** on heavy atom MACE embeddings for GEOM molecules. This is a critical finding:
- Our model uses heavy atoms only (9 types: C, N, O, F, S, Cl, Br, I, *)
- MACE can be run on heavy atoms only with no information loss
- No need to handle H-atom padding or filtering in the encoder

**Why**: GEOM molecules are pre-processed with `RemoveAllHs`, so the LMDB stores no H atoms. MACE's neighbor graph only includes the atoms present, so heavy-atom-only input produces the same embeddings as if we had removed H from an all-atom structure.

### 5. Strong atom-type discrimination (fig_04)

| Same-element comparison | Mean cosine sim | Range |
|------------------------|-----------------|-------|
| C-C (different envs)   | 0.747           | 0.327 - 1.000 |
| N-N (different envs)   | 0.709           | 0.528 - 1.000 |
| O-O (different envs)   | 1.000           | (only 2 O, same env) |

MACE clearly distinguishes:
- Different elements (Br vs C: cos = 0.19)
- Same element in different chemical environments (aromatic C vs aliphatic C: cos = 0.63-0.91)
- Symmetry-equivalent atoms (benzene ring carbons: cos ~ 0.98)

### 6. Representation comparison summary (fig_05)

| Property               | MACE (small) | CheMeleon | Winner for REPA |
|------------------------|-------------|-----------|-----------------|
| Dimension              | 192         | 2048      | MACE (smaller projector) |
| Sparsity               | 0.0%        | 93.2%     | MACE (dense gradients) |
| Effective rank         | 40.6        | 138.1     | MACE (fits in projector) |
| 3D sensitivity         | Yes (weak)  | None      | MACE |
| Atom discrimination    | Strong      | Strong    | Tie |
| Projector feasibility  | Excellent   | Poor      | MACE |

---

## Diagnosis

MACE-OFF addresses all four structural problems identified in the CheMeleon investigation:

### Problem 1: Sparsity kills cosine similarity -> SOLVED
MACE has 0% sparsity. Every dimension contributes to cosine similarity, providing clean, dense gradient signals.

### Problem 2: Projection bottleneck -> SOLVED
MACE's 192-dim output with effective rank 40.6 is easily captured by even a small projector. No information is lost in the projection step.

### Problem 3: 2D-only = no geometry guidance -> PARTIALLY SOLVED
MACE does encode 3D geometry, but the conformer sensitivity is subtle (cos ~ 0.998 between conformers). Local geometry (bond lengths/angles) is captured well; global conformation is captured weakly.

### Problem 4: Fast/slow path divergence -> NOT APPLICABLE
MACE takes coordinates directly — no SMILES-dependent code path. The same input always produces the same output.

---

## Caveats

### Performance concern
The current MACEEncoder creates a MACE DataLoader in every `forward()` call. This involves:
1. Converting coords/atomics to ASE Atoms (CPU, per-molecule loop)
2. Building MACE's graph structure (neighbor lists, edge features)
3. Running the MACE model (2 interaction layers)

This is slower than ChemPropEncoder's SMILES-cached path. Profiling needed to quantify the cost per training step.

### Weak conformer signal
cos = 0.998 between conformers means the alignment gradient for 3D structure is very small compared to the atom-identity signal. REPA may still primarily teach atom types rather than geometry, similar to CheMeleon but with cleaner gradients.

### MACE sets global dtype to float64
`mace_off()` changes `torch.default_dtype` to float64 as a side effect. The MACEEncoder handles this by explicitly casting output to float32, but care is needed if combining with other components in the same process.

---

## Recommendations

### 1. Try MACE for REPA training
The MACEEncoder is implemented and tested ([encoders.py](../../../src/tabasco/src/tabasco/models/components/encoders.py), config: [mace.yaml](../../../src/tabasco/configs/experiment/qm9/mace.yaml)). Run a training comparison:
- MACE-REPA vs CheMeleon-REPA vs no-REPA baseline
- Monitor cosine similarity convergence speed (expect faster with dense MACE gradients)

### 2. Profile encoder speed
Measure wall-clock time per training step with MACE vs CheMeleon encoder. If MACE is too slow, consider:
- Pre-computing MACE embeddings (like `CachedChemPropEncoder`)
- Caching MACE's graph construction (neighbor lists don't change for fixed coords)

### 3. Consider MACE medium model
The small model (192-dim) has very concentrated variance (94% in 10 dims). The medium model has more interaction layers and may provide richer representations with better conformer sensitivity. Worth testing if the small model's conformer signal is too weak.

### 4. Auxiliary geometry loss
Given the weak conformer signal, consider supplementing MACE-REPA with an explicit geometric loss (e.g., pairwise distance matching) for stronger 3D guidance.

### 5. Atom-type classification baseline
As recommended in the CheMeleon findings, compare MACE-REPA against a simple atom-type classification auxiliary loss. If REPA is primarily teaching atom identity (which both MACE and CheMeleon do well), the simpler approach may achieve similar benefits.

---

## Additional Analyses (2026-04-02)

**Script**: `playground/tabasco/mace/probe_and_saturation.py`

### Linear Probe: Atom-Type Discrimination

| Metric | MACE | CheMeleon | GearNet (proteins) |
|--------|------|-----------|-------------------|
| Linear probe accuracy | **1.000** | 1.000 | 0.154 |
| Within-type cosine sim | 0.706 +/- 0.202 | 0.245 | 0.189 +/- 0.139 |
| Between-type cosine sim | 0.447 +/- 0.148 | 0.152 | 0.176 +/- 0.133 |
| Delta (within - between) | **0.260** | 0.093 | 0.013 |

MACE achieves 100% linear probe accuracy — atom type is perfectly recoverable from the embedding. The within-type vs between-type gap (0.260) is larger than CheMeleon's (0.093), meaning MACE discriminates atom types more strongly while also encoding chemical environment variation (within-type std of 0.202 shows atoms of the same element in different contexts get distinct embeddings).

### Projector Saturation Test

| Input condition | MACE | CheMeleon | GearNet |
|----------------|------|-----------|---------|
| Random (128-d) | **0.858** | ~0.43 | 0.461 |
| Identity (one-hot) | **0.863** | — | 0.430 |

**MACE has severe projector saturation.** A random MLP reaches 0.86 cosine similarity with MACE targets — even higher than CheMeleon's saturation (0.43). This is because MACE's 192-dim output with effective rank ~40 is extremely low-rank; a simple MLP can approximate the mean embedding structure with very little information.

The atom-type one-hot input (0.863) performs nearly identically to random (0.858), confirming that the saturation is due to the low-rank target space, not the input information.

**Implication for REPA**: Any MACE-REPA training that reports high cosine similarity should be interpreted with extreme caution. A cos_sim of 0.86 is achievable without any meaningful input — the transformer would need to exceed ~0.90 to demonstrate genuine structural alignment beyond what the projector provides for free.

### Comparison with GearNet

GearNet's projector saturation baseline is much lower (0.46 random) yet REPA training reaches 0.78 — a 0.32 gap that represents genuine structural learning. MACE's gap would be at most ~0.14 (from 0.86 to theoretical max ~1.0), leaving much less room for the transformer to contribute meaningful signal. This analysis retroactively validates the choice of GearNet over MACE for the protein REPA pipeline.
