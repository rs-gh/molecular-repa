# Mechanistic Interpretability: Input Ablation Experiments

**Model**: GEOM-mild baseline (`evaluation_checkpoints/baseline.ckpt`)
**Setup**: 9 conditions x 50 molecules x 100 Euler steps each
**Method**: Zero out specific input embedding components at specific timestep ranges during generation, then evaluate molecular validity and connectivity.

---

## Results

| Condition | Validity (%) | Connectivity (%) | Notes |
|-----------|-------------|-----------------|-------|
| **baseline** | **96.0** | **96.0** | Reference |
| no_coords (full) | 100.0 | 0.0 | Valid fragments, no connected molecules |
| no_coords_late (t>=0.5) | 100.0 | 0.0 | Same as full ablation |
| no_coords_early (t<=0.5) | 98.0 | 96.0 | **Matches baseline** |
| no_atoms (full) | 44.0 | 0.0 | Severe degradation |
| no_atoms_late (t>=0.5) | 58.0 | 0.0 | Substantial degradation |
| no_atoms_early (t<=0.5) | 96.0 | 92.0 | **Near-baseline** |
| no_posenc (full) | 96.0 | 0.0 | Valid but disconnected |
| no_time (full) | 98.0 | 0.0 | Valid but disconnected |

---

## Key Findings

### 1. Coordinate guidance is memoryless — early geometry leaves no trace

This is the headline result and directly mirrors proteina's central finding.

- **no_coords_early (t<=0.5)**: 98% validity, 96% connectivity — **indistinguishable from baseline**
- **no_coords_late (t>=0.5)**: 100% validity, 0% connectivity — **completely broken**

The model does not use coordinate information in the first half of generation. Zeroing out the 3D geometry embedding for t <= 0.5 has essentially zero effect. But removing it for t >= 0.5 is catastrophic for connectivity.

**Interpretation**: During early generation (t=0 to 0.5), the molecular structure is still mostly noise — the coordinate embedding carries no useful geometric signal. The model relies on other inputs (atom types, positional encoding, time) to begin organizing its internal representations. Only in the second half (t=0.5 to 1.0), when the coordinates have become structured enough to carry meaningful geometric information, does the model actually use them.

This is exactly the "memoryless" property from proteina: geometric guidance is re-computed from the current state at each step, with no accumulated memory from early steps.

### 2. Atom type information follows the same temporal pattern

- **no_atoms_early (t<=0.5)**: 96% validity, 92% connectivity — near-baseline
- **no_atoms_late (t>=0.5)**: 58% validity, 0% connectivity — severe degradation
- **no_atoms (full)**: 44% validity, 0% connectivity — near-total failure

Early atom type information is disposable; late atom type information is essential. The model needs to know what atoms it's working with in the refinement phase, but during the initial organisation from noise, atom types are redundant.

### 3. Atom types matter more than coordinates for validity

- **no_coords (full)**: 100% validity, 0% connectivity — valid individual fragments
- **no_atoms (full)**: 44% validity, 0% connectivity — even fragments are mostly invalid

Without coordinates, the model can still produce chemically valid atom-level structures (correct valences, etc.) — it just can't connect them into a single molecule. Without atom types, the model cannot even determine correct valences, leading to invalid structures.

This hierarchy — **atom types > coordinates for validity** — makes chemical sense: valence rules depend on element identity, while connectivity depends on spatial arrangement.

### 4. Positional encoding and time encoding are required for connectivity

- **no_posenc**: 96% validity, 0% connectivity
- **no_time**: 98% validity, 0% connectivity

Both positional encoding and time encoding are independently necessary for molecular connectivity. Neither affects validity.

**Positional encoding**: The sinusoidal encoding provides an ordering signal that helps the model associate atoms within the same molecule. Without it, the model cannot distinguish which atoms should be connected.

**Time encoding**: The Fourier time encoding tells the model where it is in the generation trajectory. Without it, the model cannot adjust its behaviour from "organise from noise" (early) to "refine structure" (late), resulting in a failure to produce connected molecules.

### 5. Connectivity is fragile; validity is robust

The most striking pattern across all ablations: **validity is remarkably robust** (only no_atoms drops it significantly), while **connectivity is fragile** (destroyed by removing ANY of: coords, posenc, or time).

This suggests that validity (correct atom valences) is a "local" property that the model learns at the atom level, while connectivity (all atoms in one molecule) is a "global" property that requires the model to coordinate across all atoms — and this coordination depends on multiple complementary input signals.

---

## Comparison with Proteina

| Finding | Proteina (pair bias B) | Tabasco (input embeddings) |
|---------|----------------------|---------------------------|
| Early geometry disposable | Yes (B early has no effect) | Yes (no_coords_early matches baseline) |
| Late geometry essential | Yes (B late is critical) | Yes (no_coords_late destroys connectivity) |
| Memoryless property | Yes (gap experiment confirms) | Yes (early ablation ≈ no ablation) |
| Structure at output only | Yes (final 1-2 layers) | Partially (bond precision peaks in mid-layers 4-11) |
| Descriptive ≠ causal | Yes (Rc magnitude misleading) | Yes (coords have high distance correlation but validity doesn't require them) |

The temporal dynamics are remarkably similar despite the architectural differences:
- Proteina uses explicit pair bias B derived from pairwise distances
- Tabasco uses coordinate embeddings summed into the input

Both architectures exhibit the same "memoryless" property where early geometric information is irrelevant and only late-stage geometry matters. This suggests it's a **fundamental property of flow-matching generation**, not an architectural artifact.

---

## Implications for REPA

### 1. Time-weighted REPA loss should focus on t > 0.5

Since the model only uses geometric/chemical information in the second half of generation, REPA alignment during t < 0.5 is likely wasted — the model isn't building meaningful representations that could benefit from alignment. A time-weighted REPA loss that upweights t > 0.5 (or simply ignores t < 0.5) could be more efficient.

### 2. Coordinate and atom type embeddings are complementary targets

Coordinates control connectivity; atom types control validity. REPA alignment with ChemProp (which encodes 2D chemical graphs — i.e. both atom types AND bond connectivity) could improve both aspects simultaneously. The question is which layers to target:
- **Layers 4-11** (where bond precision peaks) for connectivity improvement
- **Final layers** (where atom accuracy is highest) for validity refinement

### 3. The connectivity fragility suggests room for improvement

The fact that connectivity requires ALL of {coords, posenc, time} to function suggests the baseline model's connectivity mechanism is fragile — a small perturbation in any input channel breaks it. REPA alignment could make the model more robust by providing an alternative structural signal through the frozen encoder, potentially making connectivity less dependent on any single input component.

---

## Files

- `ablation_results.npz` — labels, validity, connectivity arrays
- `figures/ablation_summary.png` — bar charts

## Reproduction

```bash
source .venv/bin/activate && export PROJECT_ROOT=$(pwd)/src/tabasco
python playground/mech_interp/run_analysis.py \
    --checkpoint evaluation_checkpoints/baseline.ckpt \
    --num-molecules 50 --num-steps 100 \
    --run-ablations \
    --output-dir playground/mech_interp/results/ablations
```
