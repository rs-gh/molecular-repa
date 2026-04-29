# CheMeleon Embedding Investigation: Findings

**Date**: 2026-03-18
**Script**: `playground/analysis/investigate_chemeleon.py`
**Figures**: `playground/analysis/figures/`

## Background

We investigated whether the CheMeleon encoder (frozen, pretrained on 1M PubChem molecules) produces useful embeddings for REPA alignment. REPA aligns the flow model's hidden states with CheMeleon's atom-level embeddings via cosine similarity. Training results so far are inconclusive.

### Hypotheses for why REPA isn't working

1. Aligning at the wrong place or time
2. **Aligning with a bad model** (this investigation)
3. Metrics are saturated (separate investigation)
4. Data pipeline broken (partially tested here)
5. 2D<->3D fundamental mismatch (tested here)
6. Sparsity problem (tested here)
7. Dimensionality explosion in projection (tested here)
8. Weak gradient signal at initialization (tested here)

---

## Key Results

### 1. Embeddings ARE chemically meaningful (fig_01, fig_02, fig_03)

- **Cosine similarity range**: 0.15 to 1.0 across 20 known molecules (mean 0.46)
- **Benzene symmetry**: All 6 carbons have cosine sim = 1.0 (perfect symmetry awareness)
- **Chemical clustering**: Similar molecules group together (benzene/pyridine/toluene cluster, alcohols cluster, alkanes cluster)
- **Embedding norms**: Vary by atom type (C, N, O, S have distinct norm distributions)

### 2. Extreme sparsity: 93.8% zeros (fig_04, fig_05, fig_06)

|                             |    QM9 |   GEOM |
|-----------------------------|--------|--------|
| Exact zero fraction         | 0.9376 | 0.9380 |
| Dead dimensions (never active) | 133 | 111    |
| Rarely active (<1%)         |    644 |    685 |
| Frequently active (>50%)    |      7 |      4 |

- CheMeleon uses ReLU, so most dimensions are exactly 0
- Only ~7 dimensions are active for >50% of atoms
- This means cosine similarity operates on very sparse vectors, producing noisy gradients

### 3. High effective rank, low participation ratio

> **Definition note**: "Effective rank (1% SV)" below uses a threshold-based metric: count of singular values exceeding 1% of the maximum. Other analyses use entropy-based definitions which give ~138 for CheMeleon. See `playground/projector/encoder_analysis.py` for all four definitions computed consistently across all encoders.

|                        |  QM9 |  GEOM |
|------------------------|------|-------|
| Effective rank (1% SV) |  500 |   500 |
| Participation ratio    | 58.5 |  50.0 |

- PCA needs >500 components for 90% variance - this exceeds the 128-dim projector input
- But participation ratio is only ~55, meaning variance is concentrated in few directions
- **The projector bottleneck (128-dim) is too narrow for a target with 500 effective dimensions**

### 4. Embeddings are highly discriminative (fig_07, fig_08, fig_09, fig_10)

|                                |  QM9  |  GEOM |
|--------------------------------|-------|-------|
| Mol-level cosine sim (mean)    | 0.456 | 0.566 |
| Within-mol atom sim (mean)     | 0.245 | 0.160 |
| Between-mol atom sim (mean)    | 0.152 | 0.116 |
| Linear probe accuracy          | 1.000 | 1.000 |

- **Perfect linear probe**: A linear classifier achieves 100% accuracy predicting atom type from CheMeleon embedding on both datasets
- Within-molecule similarity > between-molecule (embeddings capture local context)
- Wide distribution of molecule-level similarities (good discrimination, not collapsed)
- Per-class: all atom types perfectly separated (C, N, O, F, S, Cl, Br, I)

### 5. Fast path != slow path — BIG discrepancy (fig_11)

| Path comparison              | Value |
|------------------------------|-------|
| Mean cosine sim (fast vs slow) | 0.23 |
| Min cosine sim               | 0.11  |
| Molecules with sim < 0.99    | 100/100 (100%) |
| Aromatic mean                | 0.28  |
| Non-aromatic mean            | 0.22  |

**This is a major finding.** The SMILES-based fast path and the coordinate-based slow path produce substantially different embeddings for ALL molecules, not just aromatic ones. The discrepancy (mean cosine sim = 0.23) is enormous.

- Fast path: `MolFromSmiles` -> proper SMILES-derived molecular graph with correct aromaticity, bond orders, etc.
- Slow path: `DetermineConnectivity` from 3D coords -> crude bond inference from geometry

This means:
- If training uses SMILES (fast path), the alignment targets are based on canonical chemistry
- If SMILES are ever lost (e.g., during generation/sampling), the slow path produces completely different targets
- The two paths are NOT interchangeable

### 6. Pipeline integrity: atom order and padding are correct

| Check                    | Result |
|--------------------------|--------|
| Atom order match         | 100/100 (100%) |
| Padding all-zero         | 100/100 (100%) |

- Atom ordering between tensor representation and SMILES is consistent
- Padding positions correctly produce zero embeddings

### 7. Projection is learnable but limited (fig_12)

| Input type          | Final cosine sim (200 epochs) |
|---------------------|------------------------------|
| Random inputs       | 0.434                        |
| Atom-type only      | 0.455                        |
| Atom-type + context | 0.471                        |
| Random baseline (no training) | 0.004              |

- A 2-layer MLP CAN learn to project to CheMeleon space (0.43+ cosine sim)
- **But**: random inputs achieve nearly the same as informative inputs (0.43 vs 0.47)
- This suggests the projector mostly learns the **average structure** of CheMeleon embeddings (atom-type templates), not fine-grained molecular context
- The marginal benefit of context beyond atom identity is very small (~0.02)

### 8. 2D vs 3D mismatch confirmed (fig_13)

| Metric                              | Value |
|-------------------------------------|-------|
| Conformer pairs analyzed            | 484   |
| CheMeleon L2 distance (all pairs)   | 0.000 |
| 3D RMSD range                       | 0.03 - 4.19 |
| 3D RMSD mean                        | 2.11  |

- CheMeleon produces **identical** embeddings for all conformers of the same molecule
- Conformers have wildly different 3D geometries (RMSD up to 4.2)
- REPA therefore forces the flow model to learn conformation-invariant representations
- **REPA can help with atom identity/topology but CANNOT provide 3D geometric guidance**

---

## Diagnosis

CheMeleon embeddings are **chemically meaningful and discriminative**, so hypothesis 2 ("bad model") is not entirely correct — the model IS good at what it does. However, several structural problems make it a poor REPA target:

### Problem 1: Sparsity kills cosine similarity
93.8% zeros means cosine similarity is computed over ~130 active dimensions out of 2048. The gradient signal is dominated by matching the sparse activation pattern rather than learning fine-grained alignment.

### Problem 2: Projection bottleneck
The target has ~500 effective dimensions but the projector input is only 128 (or 256 with cross-attention). The projector cannot represent the full target space. It learns atom-type templates and little else.

### Problem 3: 2D-only = no geometry guidance
For a 3D molecular generation task, the alignment target provides zero geometric information. For QM9 (max 9 atoms), atom identity is trivially learnable without REPA. For GEOM, the topology signal could be more useful but still cannot guide 3D structure.

### Problem 4: Fast/slow path divergence
The SMILES and coordinate-based paths produce very different MolGraphs. Training uses SMILES (fast path) but generation/sampling may not have access to ground-truth SMILES.

---

## Recommendations

### Short-term fixes (within current architecture)
1. **Align in PCA-reduced space**: Project CheMeleon targets to top-K PCA components (K ~ 50-100) before computing cosine similarity. Removes dead/noisy dimensions and makes the projection feasible.
2. **Use MSE in PCA space instead of cosine**: After PCA reduction, MSE provides cleaner gradients than cosine on sparse vectors.
3. **Reduce projector output dim**: Match projector to PCA-reduced dim (e.g., 64-100) instead of 2048.

### Medium-term alternatives
4. **Use a 3D-aware encoder**: Replace CheMeleon with a 3D encoder (e.g., MACE, SchNet, DimeNet++) that uses coordinates. This addresses the fundamental 2D/3D mismatch.
5. **Pre-ReLU features**: If accessible, use CheMeleon features before the final ReLU — this eliminates the sparsity problem.
6. **Contrastive loss**: Instead of cosine similarity, use a contrastive loss that only requires relative ordering (similar molecules closer than dissimilar), not exact alignment.

### Experiments to run
7. **Ablation**: Train with REPA but in PCA-reduced space (recommendation 1) and compare to baseline.
8. **Atom-type-only baseline**: If REPA is mostly teaching atom identity (Section 5 result), compare to a simple auxiliary atom-type classification loss — it may achieve the same benefit with much less complexity.
