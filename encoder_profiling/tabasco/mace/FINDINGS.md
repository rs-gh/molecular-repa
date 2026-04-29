# MACE-OFF Encoder Characterization

**Date**: 2026-03-19 (initial), 2026-04-02 (additional probes)
**Encoder**: MACE-OFF23 small (`mace_off("small")`, frozen), 2 interaction layers, l_max=0 (all features rotation-invariant scalars), 192-dim output (96 invariant features × 2 layers concatenated)
**Data**: GEOM train molecules; per-section sample sizes vary
**Scripts**: [explore_mace.py](explore_mace.py), [probe_and_saturation.py](probe_and_saturation.py), [generate_figures.py](generate_figures.py), [precompute_embeddings.py](precompute_embeddings.py), [extract_descriptors.py](extract_descriptors.py)
**Figures**: [figures/](figures/)
**Cross-encoder context**: [../FINDINGS.md](../FINDINGS.md) — for the three-question framing (Q1 information, Q2 saturation, Q3 conditioning) used throughout.

## Summary

MACE-OFF small is the cleanest *3D-aware* REPA target available off-the-shelf in our shortlist, and it fixes every Q3 conditioning problem CheMeleon has: **0% sparsity** (vs CheMeleon 93.8%), eff rank 40.6 fits the projector with room to spare, no dead dims, no norm explosion. Q1.1 and Q1.3 are strong (probe 1.000, atom-type Δ = 0.260 — 3× CheMeleon).

The catch is at Q2. The mean-direction floor is ~0.86 — *random inputs alone reach this number* — and the projector gap above the floor is only ~+0.005. The Q1.2 conformer signal is real but small (cos 0.998 between conformers): MACE-OFF was trained on energies, and local chemistry dominates the energy, so global conformation has weak influence on embeddings. Net: **borderline; saturated**. REPA against MACE has very little operating budget — any cosine ≥ 0.86 reported during training is likely the projector floor, not learning.

## Q1. What information does the encoder encode?

### 1.1 Atom identity

Linear probe (logistic regression) from MACE embedding to atom type (`probe_and_saturation.py`):

| Metric | MACE | CheMeleon | GearNet (proteins, for context) |
|--------|------|-----------|----------------------------------|
| Linear probe accuracy | **1.000** | 1.000 | 0.154 |
| Within-type cos | 0.706 ± 0.202 | 0.245 | 0.189 ± 0.139 |
| Between-type cos | 0.447 ± 0.148 | 0.152 | 0.176 ± 0.133 |
| **Δ (within − between)** | **0.260** | 0.093 | 0.013 |

Atom type is perfectly recoverable. The within-type std (0.202) shows MACE also encodes *environment* variation within a single element — same-element atoms in different chemical contexts get different embeddings, not just different types. This is the strongest atom-type discrimination of any encoder we've profiled.

### 1.2 3D geometric sensitivity — weak but nonzero

Conformer pair comparison on GEOM:

| Metric | MACE | CheMeleon |
|--------|------|-----------|
| Mean cos between conformers | **0.998** | 1.000 |
| L2 between conformers | 0.08 – 0.29 | 0.000 |
| Coordinate RMSD range | 1.4 – 3.7 Å | (same) |

MACE *does* distinguish conformers (cos < 1.0), confirming 3D awareness — but the signal is small. Conformer pairs with up to 3.7 Å RMSD still have cos > 0.995. **Why**: MACE-OFF was trained on energies and forces, where local chemical environment (bond lengths, angles, immediate neighbours) dominates. Global conformational degrees of freedom (torsion angles, extended structure) have weaker influence on the energy and therefore weaker influence on embeddings.

**Implication**: MACE provides genuine 3D headroom for REPA, but it's local (bond geometry) rather than global (conformation). That's the *right kind* of headroom for a 3D generative model in principle, but the magnitude is limited.

Rotation invariance is exact by construction (`l_max=0`, all output features are SO(3)-invariant scalars).

### 1.3 Chemical / topological context

Same-element-different-environment cosine (`probe_and_saturation.py`):

| Same-element comparison | Mean cos sim | Range |
|------------------------|--------------|-------|
| C-C (different envs) | 0.747 | 0.327 – 1.000 |
| N-N (different envs) | 0.709 | 0.528 – 1.000 |
| Symmetry-equiv (benzene ring C) | ~0.98 | (tight) |
| Different elements (Br vs C) | 0.19 | (clean separation) |

MACE cleanly separates: different elements; same element in different environments (aromatic vs aliphatic C: cos 0.63–0.91); symmetry-equivalent atoms (benzene ring carbons: cos ~ 0.98). This is what enables the strong Q1.1 atom-type Δ — the within-type structure isn't noise; it's chemistry.

### 1.4 Hydrogen-atom invariance

Heavy-atom-only vs all-atom embedding comparison (relevant because the tabasco data pipeline pre-removes hydrogens):

| Metric | Value |
|--------|-------|
| Cos sim (all-atom vs heavy-only embedding) | **1.0000** |
| Min cos sim across atoms tested | 1.0000 |

**Removing hydrogens has zero effect on heavy-atom embeddings.** GEOM molecules are pre-processed with `RemoveAllHs`, so the LMDB stores no H atoms; MACE's neighbour graph only includes the atoms present, and heavy-atom-only input produces the same embeddings as if H atoms had been silently removed. **No padding/filtering logic needed in the encoder for H atoms.**

## Q2. How much is reachable from cheap inputs?

MLP from each input condition to the MACE embedding, cosine loss (`probe_and_saturation.py`):

| Input condition | MACE | CheMeleon | GearNet (proteins, for context) |
|-----------------|------|-----------|----------------------------------|
| Random 128-d (mean-direction proxy) | **0.858** | ~0.43 | 0.461 |
| Atom-type one-hot (8-d) | **0.863** | ~0.46 | 0.430 |

**Saturation gap = atom-onehot − random ≈ +0.005.** Two facts to read together:

1. **The floor is enormous.** Random inputs reach 0.858 — i.e., the projector emits something close to the mean direction of the embedding distribution and gets free 0.86 cosine. This is consistent with Q3.2: eff rank 40.6 means the embedding manifold is low-dim, the mean direction captures most of it.
2. **Atom identity barely moves the needle.** From 0.858 → 0.863 with atom-type input. The encoder's variance off the mean direction is so tight that even informative input can't extract much more.

**Implication for REPA training**: any reported `cos_sim > 0.86` is achievable without input information. Genuine learning requires `cos > 0.86 + ε`. The theoretical ceiling (1.0) leaves ~0.14 of room; most of that is local geometry from Q1.2; the headroom for the transformer to do useful work is tiny.

For comparison, GearNet for proteins has random-input floor 0.461 and trained REPA reaches 0.78 — a 0.32 gap that genuinely represents structural learning. MACE's gap is bounded above by ~0.14, much less room for the student to contribute.

## Q3. Is the encoder a tractable optimisation target?

### 3.1 Sparsity & value distribution

| Metric | MACE | CheMeleon |
|--------|------|-----------|
| Exact-zero fraction | **0.0%** | 93.2% |
| Near-zero (<1e-6) | 0.0% | 93.2% |

MACE has **no sparsity** — output values follow a roughly Gaussian distribution centered near 0. Every dimension contributes to every cosine computation. **This eliminates the gradient-noise problem that dominated CheMeleon alignment.**

### 3.2 Effective dimensionality & singular values

> **Definition note**: Eff rank below uses entropy on normalised squared singular values (`exp(−Σ p log p)` where `p = Sᵢ² / Σ S²`). Same definition used everywhere in the proteina probes and the [cross-encoder summary](../FINDINGS.md). CheMeleon's "500" eff rank in its own FINDINGS uses the threshold-based variant.

| Metric | MACE | CheMeleon |
|--------|------|-----------|
| Output dim | 192 | 2048 |
| Effective rank (entropy) | **40.6** | ~138 |
| Dims for 90% variance | 7 | ~100 |
| Dims for 99% variance | ~30 | ~300 |

40.6 / 192 ≈ 21% of capacity. **The projector input dim (128 or 256) is larger than the eff rank** — no bottleneck, every direction in the target manifold is recoverable in principle. This is the structural opposite of CheMeleon, where 500 dims for 90% variance vs 128-d input was a hard bottleneck.

The flip side: 94% of variance lives in just 10 dimensions. That's why Q2 saturates so hard — capturing those 10 dims is easy, and there's not much else.

### 3.3 Norms

Per-atom L2 norms vary modestly with atom type and chemical environment, no dead dims, no norm explosion. Well-conditioned across the board.

## Why MACE addresses CheMeleon's structural problems (and where it doesn't)

CheMeleon failure mode → MACE status:

| CheMeleon problem (Q-tag) | MACE status |
|---------------------------|-------------|
| Q3.1 sparsity 93.8% | ✅ solved (0%) |
| Q3.2 projector bottleneck (500 dims for 90%, 128-d input) | ✅ solved (eff rank 40.6, fits) |
| Q1.2 zero 3D sensitivity (L2 = 0 across conformers) | ◐ partially solved (cos 0.998 between conformers — small but nonzero) |
| Q2 fast/slow path divergence (cos 0.23) | ✅ N/A (MACE takes coords directly, no SMILES path) |

MACE does *not* solve the saturation problem at Q2 — it trades the high-dim sparse-target mode for a low-rank dense-target mode. Both modes are bad for the projector, just differently:

- **CheMeleon**: lots of structure (high eff rank) but trapped behind sparse activation patterns and a too-narrow projector input. Floor 0.43, modest +0.04 gap.
- **MACE**: clean dense gradients but the structure itself is so low-dim that the mean direction already explains most of it. Floor 0.86, tiny +0.005 gap.

## Caveats

### Performance / runtime

The naive `MACEEncoder` builds a MACE DataLoader per `forward()` call: ASE Atoms conversion (CPU loop), neighbour-list construction, 2 interaction layers. Slower than `ChemPropEncoder`'s SMILES-cached path. Profiling needed; precomputed embeddings (per-conformer LMDB, see [precompute_embeddings.py](precompute_embeddings.py)) sidestep this entirely since MACE is rotation-invariant and the conformer→embedding map is fixed.

### Global float64 side-effect

`mace_off()` mutates `torch.default_dtype` to float64. `MACEEncoder` casts its output back to float32; care needed if combining with other float32-only components in the same process.

### Weak conformer signal at Q1.2

cos = 0.998 between conformers means even a perfect transformer would only see a small gradient pulling toward correct global conformation. Local geometry (bond lengths/angles) is captured well; torsion angles less so. If the goal is sharp conformational accuracy, MACE-REPA alone won't deliver it — see "auxiliary geometry loss" below.

## Implications for REPA training

1. **Q2 saturation is the dominant concern.** Any MACE-REPA training run reporting `cos > 0.86` should be cross-checked against the random-input floor (0.858) before claiming alignment progress. Aim for `> 0.90` to demonstrate genuine learning.
2. **MACE small may be too compressed.** 94% variance in 10 dims is extreme. MACE *medium* has more interaction layers and a larger feature dim — likely higher eff rank, possibly stronger conformer sensitivity. Worth profiling with the same probes if MACE-REPA results are mixed.
3. **Auxiliary geometry loss.** Given the weak conformer signal, an explicit pairwise-distance / RMSD loss alongside MACE-REPA likely beats either alone for 3D accuracy.
4. **Atom-type CE baseline.** Both MACE and CheMeleon hit Q1.1 probe = 1.000. If REPA's main contribution is teaching atom identity, a direct atom-type CE loss is much cheaper and likely matches the effect.
5. **Cache aggressively.** MACE is rotation-invariant, so per-conformer embeddings precompute once; `CachedMACEEncoder` mirrors `CachedChemPropEncoder` and avoids the per-step encoder cost entirely.

## Why CheMeleon was investigated first

CheMeleon was the original encoder; the 2026-03-18 investigation ([../chemeleon/FINDINGS.md](../chemeleon/FINDINGS.md)) identified Q1.2 (no 3D signal), Q3.1 (sparsity), and Q3.2 (projector bottleneck) as fundamental blockers. MACE-OFF was selected as the natural 3D-aware alternative; the head-to-head verdict and operational guidance live in [../FINDINGS.md](../FINDINGS.md).
