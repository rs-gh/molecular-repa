# CheMeleon Encoder Characterization

**Date**: 2026-03-18 (initial), 2026-04-02 (additional probes)
**Encoder**: CheMeleon (`ChemPropEncoder`, frozen, pretrained on ~1M PubChem molecules), 2048-dim, ReLU output
**Data**: QM9 train (≤9 atoms) and GEOM train molecules; per-section sample sizes vary (20-1000 mols)
**Scripts**: [investigate.py](investigate.py), [verify_pipeline.py](verify_pipeline.py), [precompute_embeddings.py](precompute_embeddings.py), [benchmark_encoder.py](benchmark_encoder.py)
**Figures**: [figures/](figures/)
**Cross-encoder context**: [../FINDINGS.md](../FINDINGS.md) — for the three-question framing (Q1 information, Q2 saturation, Q3 conditioning) used throughout.

## Summary

CheMeleon is **unusable as a REPA target for a 3D molecular generative model**. The fundamental problem is Q1.2: CheMeleon is a 2D bond-graph encoder by construction (`MolFromSmiles → MolGraph`, no coordinates), and produces *bit-identical* embeddings (L2 = 0.000) for every conformer of a given molecule. REPA against CheMeleon therefore cannot transfer any 3D geometric signal — it can only teach conformation-invariant atom identity and topology.

Stacked on top of this fundamental issue: Q3 conditioning is poor — 93.8% sparsity (ReLU), 500 dims for 90% variance against a 128-d projector input — and Q2 saturation is gentle but exists (+0.04 gap). The Q1.1 / Q1.3 atom-type and bond-context discrimination *is* genuinely strong (probe 1.000, clean clustering), so the encoder is "good at what it does" — that ability just isn't 3D guidance.

## Q1. What information does the encoder encode?

### 1.1 Atom identity

Linear probe (logistic regression) from CheMeleon embedding to atom type.

| Metric | QM9 | GEOM |
|--------|-----|------|
| Linear probe accuracy | **1.000** | **1.000** |
| Atom types separated | All 8 (C, N, O, F, S, Cl, Br, I) | All 8 |

Atom type is perfectly recoverable. For QM9 (≤9 atoms with very limited type diversity) this is trivially learnable from the input one-hot — REPA against CheMeleon adds nothing here.

### 1.2 3D geometric sensitivity — **NONE**

CheMeleon takes a 2D `MolGraph` (atoms, bonds, bond orders, aromaticity) — *not* coordinates. Conformer comparison on GEOM:

| Metric | Value |
|--------|-------|
| Conformer pairs analyzed | 484 |
| **CheMeleon L2 distance, all pairs** | **0.000** |
| Coordinate RMSD range | 0.03 – 4.19 Å |
| Coordinate RMSD mean | 2.11 Å |

Conformers with up to 4 Å RMSD produce *bit-identical* embeddings. **REPA therefore forces the flow model to learn conformation-invariant representations** — it can teach atom identity and topology, but it cannot guide 3D geometry. This is the central reason CheMeleon is unsuitable as a REPA target for tabasco.

### 1.3 Chemical / topological context

Within-molecule and atom-type-based context probing (`investigate.py`, 2026-04-02 probes):

| Metric | QM9 | GEOM |
|--------|-----|------|
| Within-molecule atom cos sim | 0.245 | 0.160 |
| Between-molecule atom cos sim | 0.152 | 0.116 |
| Δ (within-mol − between-mol) | 0.093 | 0.044 |

| Atom-type discrimination | Value |
|--------------------------|-------|
| Within-type cos | 0.245 |
| Between-type cos | 0.152 |
| Δ (within − between) | **0.093** |

Embeddings cluster by molecule (within-mol > between-mol) and by atom type. The chemistry signal is real: benzene's six carbons share cos = 1.000 (perfect symmetry awareness), aromatic vs aliphatic carbons cleanly separate, alcohols cluster. None of this requires 3D coordinates — it's all derivable from the bond graph.

### 1.4 Molecule-level identity

Mean cosine sim *between molecules* (taking molecule-level mean embedding):

| Dataset | Mean mol-level cos sim |
|---------|-----------------------|
| QM9     | 0.456 |
| GEOM    | 0.566 |

Wide distribution (good discrimination, no centroid collapse). Higher in GEOM is consistent with longer, more chemically similar drug-like molecules.

## Q2. How much is reachable from cheap inputs?

2-layer MLP from each input condition to the CheMeleon embedding, 200 epochs, cosine loss.

| Input condition | Final cos sim |
|-----------------|--------------|
| Random baseline (no training) | 0.004 |
| Random inputs (trained MLP) | **0.434** |
| Atom-type only | 0.455 |
| Atom-type + within-molecule context | **0.471** |

**Saturation gap = best − random ≈ +0.04.** Interpretation:

- The mean direction of the embedding distribution gets the projector to ~0.43 with no input information at all (random inputs reach this floor).
- Atom identity adds +0.02; atom identity + within-mol context adds another +0.02 on top.
- The student projector reads off a transformer hidden state computed from `(atom-onehot, noisy 3D coords, attention, timestep)` — the *only* information available there that the cheap MLP doesn't have is coords / attention / timestep. Since CheMeleon is conformation-invariant (Q1.2), coords contribute nothing to the target alignment; only attention-derived bond-graph context can extract the remaining +0.02.

The headroom that exists is real but is conformation-invariant by construction. For a 3D generative model this headroom is the wrong kind.

## Q3. Is the encoder a tractable optimisation target?

### 3.1 Sparsity & value distribution

| Metric | QM9 | GEOM |
|--------|-----|------|
| Exact-zero fraction | **0.9376** | **0.9380** |
| Dead dimensions (never active) | 133 | 111 |
| Rarely active (<1%) | 644 | 685 |
| Frequently active (>50%) | 7 | 4 |

CheMeleon uses ReLU at the output, so 93.8% of values are exactly zero. Cosine similarity is computed over the union of active dimensions of the two operands — typically ~130 of 2048. **Gradients are dominated by activation-pattern matching rather than fine-grained vector alignment.** This is the single biggest Q3 failure.

### 3.2 Effective dimensionality & singular values

> **Definition note (stale)**: This page's effective-rank numbers were computed under two definitions: (a) **Threshold-based** = count of singular values exceeding 1% of S₀; (b) **Entropy-based on σ²** = exp(−Σ p log p) where p = Sᵢ² / Σ S². The proteina probes were switched to **RankMe** (Garrido et al. 2023, ICML — entropy on raw σ of the *uncentered* matrix) on 2026-05-04, so neither of the numbers below is directly comparable to the cross-encoder [proteina FINDINGS](../../proteina/FINDINGS.md) anymore. The CheMeleon `investigate.py` was updated 2026-05-04 to also emit RankMe, but **has not yet been re-run**; the numbers below stand under their original definitions until a fresh sweep lands.

| Metric | QM9 | GEOM |
|--------|-----|------|
| Output dim | 2048 | 2048 |
| Effective rank (1% threshold) | 500 | 500 |
| Effective rank (entropy on σ², old) | ~138 | ~138 |
| RankMe (Roy–Vetterli on raw σ) | *pending re-run* | *pending re-run* |
| Participation ratio | 58.5 | 50.0 |
| Dims for 90% variance | ~100 | ~100 |
| Dims for 99% variance | ~300 | ~300 |

The threshold-based metric (500) is the right one to compare to the projector input dim. **The projector input is 128 (or 256 with cross-attention); the target needs 500 dims for 90% variance.** This is a hard bottleneck — even a perfect projector can capture at most a low-rank approximation of the target manifold.

### 3.3 Norms & dead dimensions

133 dead dims (QM9) / 111 (GEOM) — ~5–6% of the output dimension is permanently silent. Per-atom-type norm distributions are distinct (C, N, O, S each have characteristic norm bands), confirming that *active* dims do encode chemistry well. Padded atoms produce all-zero embeddings (verified 100/100 — the encoder doesn't accidentally embed the `*` dummy atom into a junk vector).

## Pipeline integrity

Operational checks (not part of Q1/Q2/Q3 but critical for correct probing):

| Check | Result |
|-------|--------|
| Atom order match (tensor vs SMILES) | 100/100 |
| Padding produces all-zero embeddings | 100/100 |
| Fast vs slow path agreement | **fails (mean cos = 0.23)** |

**Fast/slow path divergence — major caveat for inference.** `MolFromSmiles` (fast, used during training when SMILES are available) and `DetermineConnectivity` (slow, coordinate-based bond inference, used when SMILES aren't) produce substantially different MolGraphs:

| Path comparison | Value |
|-----------------|-------|
| Mean cos sim (fast vs slow) | 0.23 |
| Min cos sim | 0.11 |
| Mols with cos < 0.99 | **100/100** |

The two paths are **not interchangeable**. Training uses SMILES; generation/sampling may not have SMILES — the slow path produces totally different alignment targets. This is an operational concern, not a Q1/Q2/Q3 metric, but it bites hard if the inference pipeline ever falls off the SMILES path.

## Implications for REPA training

1. **Don't use CheMeleon as a 3D-REPA target** (Q1.2 — conformation-invariant by construction). Any cosine signal it provides is orthogonal to 3D structure.
2. **If used anyway** (e.g., to teach atom identity / topology in a multi-objective setup), expect the projector to absorb most of the available signal — the structural gap is only +0.04, and a simple atom-type CE loss likely does most of what CheMeleon-REPA does at much lower complexity.
3. **The sparsity is the structural blocker for fine-grained alignment.** Even if you accept the 2D limitation, 93.8% zeros makes the cosine objective dominated by which dims are active rather than how well the active values align. Pre-ReLU features would help if accessible.
4. **Cache, don't recompute.** Since CheMeleon is conformation-invariant, embeddings can be precomputed per unique SMILES — see [precompute_embeddings.py](precompute_embeddings.py) and `CachedChemPropEncoder`.

## Why MACE was investigated next

MACE-OFF small was evaluated as the natural 3D-aware alternative — see [../mace/FINDINGS.md](../mace/FINDINGS.md). MACE solves Q1.2 (genuinely 3D-aware) and Q3.1 (0% sparsity), but introduces a different problem: severe Q2 saturation (mean-dir floor ≈ 0.86, gap +0.005). The cross-encoder verdict and the head-to-head context live in [../FINDINGS.md](../FINDINGS.md).
