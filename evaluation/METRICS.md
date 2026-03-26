# Evaluation Metrics Reference

All metrics used to evaluate the quality of generated molecules from our flow matching models, across validation (during training) and evaluation (post-hoc).

## Summary Table

Two collection contexts:
- **Validation** — computed automatically during training by callbacks, every epoch, on 100 generated samples. Logged to WandB. Noisier but available at every epoch for epoch-matched comparisons. See `scripts/tabasco/geom/compile_wandb_curves.py`.
- **Evaluation** — computed post-hoc from final checkpoints by `scripts/evaluate.py` on 1000 generated samples with optional bootstrap CIs. More reliable but only available for saved checkpoints. See `scripts/compile_results.py`.

| # | Metric | Validation (n=100) | Evaluation (n=1000) | Observed Spread | Saturation |
|---|--------|---------------------|----------------------|----------------|------------|
| 1 | Validity | Every 1000 steps | With bootstrap CI | 4% (0.94–0.98) | HIGH |
| 2 | Connectivity | Every 1000 steps | With bootstrap CI | 0% (~1.00) | HIGH |
| 3 | Uniqueness | Every 1000 steps | With bootstrap CI | 0% (~1.00) | HIGH |
| 4 | Novelty | Every 1000 steps | With bootstrap CI | 4% (0.92–0.96) | MODERATE |
| 5 | Lipinski | Every 1000 steps | With bootstrap CI | 0 (~4.8/5) | HIGH |
| 6 | QED | Every 1000 steps | With bootstrap CI | 7% (0.63–0.70) | LOW-MOD |
| 7 | LogP | Every 1000 steps | With bootstrap CI | — | LOW |
| 8 | Atom Type Dist | Every 1000 steps | With bootstrap CI | 1% (0.96–0.98) | HIGH |
| 9 | PB Intersection | Every 1000 steps | No bootstrap (slow) | 5% (0.84–0.89) | LOW-MOD |
| 10 | PB Bond Angles | Every 1000 steps | No bootstrap (slow) | 7% (0.89–0.96) | MODERATE |
| 11 | PB Bond Lengths | Every 1000 steps | No bootstrap (slow) | 4% (0.93–0.97) | MODERATE |
| 12 | PB Steric Clash | Every 1000 steps | No bootstrap (slow) | 6% (0.87–0.93) | LOW-MOD |
| 13 | Diversity | Not computed | With bootstrap CI | Not yet measured | LOW |
| 14 | FCD | Not computed | Point estimate only | Not yet measured | LOW |
| 15 | Atom Fractions | Every 1000 steps | Not computed | ~0% | HIGH |
| 16 | Val Losses | Every epoch (full set) | Not computed | ~0% | N/A |

## Observed Values (last evaluation epoch, n=100)

From WandB training runs. All 5 GEOM models.

| Metric | Baseline | Add-Fused | Add-Same | Trade-Fused | Trade-Same |
|--------|----------|-----------|----------|-------------|------------|
| Validity | ~0.94 | ~0.94 | ~0.94 | ~0.98 | ~0.98 |
| Connectivity | ~1.00 | ~1.00 | ~1.00 | ~1.00 | ~1.00 |
| Uniqueness | ~1.00 | ~1.00 | ~1.00 | ~1.00 | ~1.00 |
| Novelty | ~0.94 | ~0.94 | ~0.92 | ~0.96 | ~0.96 |
| Lipinski | ~4.8 | ~4.8 | ~4.8 | ~4.8 | ~4.8 |
| QED | ~0.63 | ~0.63 | ~0.70 | ~0.63 | ~0.65 |
| Atom Type Dist | ~0.98 | ~0.98 | ~0.98 | ~0.97 | ~0.97 |
| PB Intersection | ~0.84 | ~0.89 | ~0.86 | ~0.84 | ~0.84 |
| PB Bond Angles | ~0.94 | ~0.89 | ~0.96 | ~0.95 | ~0.95 |
| PB Bond Lengths | ~0.93 | ~0.96 | ~0.96 | ~0.97 | ~0.97 |
| PB Steric Clash | ~0.87 | ~0.93 | ~0.93 | ~0.90 | ~0.90 |
| Diversity | ? | ? | ? | ? | ? |
| FCD | ? | ? | ? | ? | ? |

---

## Per-Metric Details

### 1. Validity

- **Measures:** Fraction of samples producing a valid RDKit Mol after sanitization.
- **Formula:** `num_valid / num_total` — valid means `MoleculeConverter.from_batch()` returns non-None (atom assignment + bond inference succeeds).
- **Code:** `MolecularValidity` at `src/tabasco/src/tabasco/utils/metrics.py:19-40`
- **When:** Validation: every 1000 global steps, on 100 generated samples. evaluate.py: once on 1000 samples with bootstrap CI.
- **Saturation:** HIGH. All models reach 0.94-0.98 early. The 4% spread between models is swallowed by +/-5% epoch-to-epoch noise at n=100. Any model that learns basic atom placement saturates this.

### 2. Connectivity

- **Measures:** Fraction of valid molecules that are a single connected component (no disconnected fragments).
- **Formula:** `largest_component(mol).GetNumAtoms() == mol.GetNumAtoms()` per valid mol.
- **Code:** `MolecularConnectivity` at `metrics.py:43-79`
- **When:** Same as validity (every 1000 steps, n=100 training / n=1000 eval).
- **Saturation:** HIGH. Completely at ceiling for all runs. Zero discriminative power.

### 3. Uniqueness

- **Measures:** Fraction of unique canonical SMILES among valid generated molecules.
- **Formula:** Hash-based dedup of `Chem.MolToSmiles()` outputs; `unique_count / total_valid`.
- **Code:** `MolecularUniqueness` at `metrics.py:82-141`
- **When:** Same schedule (every 1000 steps, n=100 / n=1000).
- **Saturation:** HIGH. Chemical space is vast (~10^60 drug-like molecules); 100-1000 random samples almost never collide. Always ~1.0 regardless of model quality.

### 4. Novelty

- **Measures:** Fraction of generated SMILES absent from the training set.
- **Formula:** `smile not in original_smiles_set` for each valid mol. Requires training SMILES via `set_data_stats()`.
- **Code:** `MolecularNovelty` at `metrics.py:144-167`
- **When:** Same schedule, requires `set_data_stats()`. evaluate.py: n=1000 with bootstrap CI.
- **Saturation:** MODERATE. High overall since exact SMILES matches are rare in huge chemical space, but shows more variance than validity/connectivity. Noise dominates signal at n=100.

### 5. Lipinski Score

- **Measures:** Average drug-likeness score (0-5) based on Lipinski's Rule of Five.
- **Formula:** Sum of 5 binary rules per mol: MW<500 (+1), HBD<=5 (+1), HBA<=10 (+1), -2<=logP<=5 (+1), rotatable bonds<=10 (+1).
- **Code:** `MolecularLipinski` at `metrics.py:247-281`
- **When:** Same schedule (every 1000 steps, n=100 / n=1000).
- **Saturation:** HIGH. Small drug-like molecules from GEOM trivially satisfy all 5 rules. Flat line for all runs. The rules were designed for screening large compound libraries, not small organic molecules.

### 6. QED (Quantitative Estimate of Drug-likeness)

- **Measures:** Average QED score (0-1); continuous composite of MW, logP, HBD, HBA, PSA, rotatable bonds, aromatic rings, structural alerts.
- **Formula:** `Descriptors.qed(mol)` averaged over valid molecules.
- **Code:** `MolecularQEDValue` at `metrics.py:207-225`
- **When:** Same schedule. evaluate.py: n=1000 with bootstrap CI.
- **Saturation:** LOW-MODERATE. Continuous score with real variance — potentially the most informative "simple" metric. But averaging hides distribution shape (a bimodal QED distribution looks the same as a unimodal one in the mean). The 7% spread between models is plausible but unresolvable at n=100 noise levels. Should be informative at n=1000 with CIs.

### 7. LogP

- **Measures:** Average octanol-water partition coefficient (hydrophobicity).
- **Formula:** `Crippen.MolLogP(mol)` averaged over valid molecules.
- **Code:** `MolecularLogP` at `metrics.py:228-244`
- **When:** Same schedule. evaluate.py: n=1000 with bootstrap CI.
- **Saturation:** LOW for the raw values, but the mean alone is uninformative — distribution shape (histogram/KDE) matters more. Two very different molecular distributions can have the same mean logP.

### 8. Atom Type Distribution

- **Measures:** Similarity of atom-type frequency histograms between generated and training molecules.
- **Formula:** Histogram intersection: `sum(min(P_train[atom_type], P_gen[atom_type]))` for all atom types (C,N,O,F,S,Cl,Br,I,*).
- **Code:** `AtomTypeDistribution` at `metrics.py:284-357`
- **When:** Same schedule, requires `set_data_stats()`. evaluate.py: n=1000 with bootstrap CI.
- **Saturation:** HIGH. Near ceiling once model learns basic element ratios. Carbon dominates organic chemistry, so even crude models get the histogram roughly right.

### 9. PoseBusters Intersection (all checks pass)

- **Measures:** Fraction of generated molecules passing ALL PoseBusters geometric quality checks simultaneously.
- **Formula:** Per molecule: 1 if no PB column is False, else 0. Averaged over all generated samples. Uses `posebusters_no_strain.yaml` config (omits slow strain-energy checks).
- **Code:** `PoseBustersValidity` at `metrics.py:390-437`; callback at `callbacks/posebusters.py`
- **When:** Same schedule (every 1000 steps, n=100). evaluate.py: once, NO bootstrap CI (too slow).
- **Saturation:** LOW-MODERATE. Most discriminative single metric available during training — not at ceiling, real spread between models. But the +/-20% noise at n=100 makes epoch-level comparisons meaningless. This is a composite: one bad bond angle fails the entire molecule, so it's stricter than individual checks.

### 10. PoseBusters: Bond Angles OK

- **Measures:** Fraction of molecules with all bond angles within expected CSD-derived ranges.
- **Formula:** PoseBusters `bond_angles` check (pass/fail per mol), averaged.
- **Code:** PoseBusters library via callback; logged as `val/pb_bond_angles`.
- **When:** Same schedule. evaluate.py: included in PB results.
- **Saturation:** MODERATE. Some spread between models but still fairly high for all. The ordering is inconsistent (Add-Fused is worst) suggesting noise.

### 11. PoseBusters: Bond Lengths OK

- **Measures:** Fraction of molecules with all bond lengths within expected CSD-derived ranges.
- **Formula:** PoseBusters `bond_lengths` check (pass/fail per mol), averaged.
- **Code:** Same PoseBusters callback.
- **When:** Same schedule.
- **Saturation:** MODERATE. Slight trend: REPA variants may have better bond lengths than baseline. But 3-4% difference at n=100 is within noise.

### 12. PoseBusters: No Steric Clash

- **Measures:** Fraction of molecules with no severe steric clashes between non-bonded atoms.
- **Formula:** PoseBusters `steric_clash` check (pass/fail per mol), averaged.
- **Code:** Same PoseBusters callback.
- **When:** Same schedule.
- **Saturation:** LOW-MODERATE. Most informative individual PB check. 6% spread with possible signal that additive REPA models have fewer clashes. But n=100 noise (+/-10%) still dominates.

### 13. Diversity (ECFP Fingerprint Distance)

- **Measures:** Mean pairwise Tanimoto distance of ECFP fingerprints — structural variety of the generated set.
- **Formula:** `datamol.pdist(mols, fp_type="ecfp", fpSize=2048).mean()` over all valid molecule pairs.
- **Code:** `MolecularDiversity` at `metrics.py:170-204`
- **When:** Validation: NOT computed. evaluate.py only: n=1000 with bootstrap CI.
- **Saturation:** LOW. Informative metric, typical range 0.7-0.9 for diverse molecular sets. A model generating repetitive structures would score low; a model covering chemical space well would score high. One of the two most important unmeasured metrics.

### 14. FCD (Frechet ChemNet Distance)

- **Measures:** Distributional distance between generated and training molecules in ChemNet embedding space. Lower = generated distribution is closer to training distribution.
- **Formula:** Frechet distance of ChemNet neural network activations: `fcd_torch.FCD()(gen_smiles, train_smiles)`. Analogous to FID in image generation.
- **Code:** Only in `evaluate.py:274-293`
- **When:** Validation: NOT computed. evaluate.py: once, no bootstrap CI.
- **Saturation:** LOW. Gold-standard distributional metric for molecular generation. Captures both quality and diversity in a single number. Sensitive to subtle chemical differences that binary metrics miss. The most important unmeasured metric.

### 15. Atom Fractions (C, N, O) — training only

- **Measures:** Proportion of each element type among all atoms in generated molecules.
- **Formula:** `count(atoms_of_symbol) / count(all_atoms)` per element.
- **Code:** `AtomFractionMetric` at `metrics.py:360-387`
- **When:** Validation only (every 1000 steps, n=100). NOT in evaluate.py.
- **Saturation:** HIGH. Carbon always dominates organic molecules. These stabilize very early and don't differentiate models. Redundant with atom type distribution metric.

### 16. Validation Losses (coords, atomics, REPA)

- **Measures:** Flow matching loss components evaluated on held-out validation data.
- **Formula:** coords: `MSE(pred_coords, true_coords) / (n_atoms * 3)`. atomics: cross-entropy. REPA: `-cos_sim.mean()`.
- **Code:** `LightningTabasco.validation_step()` via respective interpolant/loss classes.
- **When:** Every validation epoch, on full validation set. NOT in evaluate.py (not a generation quality metric).
- **Saturation:** N/A — these are training diagnostics, not generation quality metrics. All runs converge to same loss, confirming REPA doesn't help or hurt the flow matching objective.

---

## Key Takeaways

1. **Saturated metrics (useless for comparison):** Validity, Connectivity, Uniqueness, Lipinski, Atom Type Distribution, Atom Fractions — all at ceiling for every model.

2. **Noise exceeds signal at n=100:** For every metric with non-zero spread between models, the epoch-to-epoch noise (+/-5-20%) exceeds the inter-model spread (0-7%). No statistically reliable conclusions possible from validation metrics alone.

3. **Most important unmeasured metrics:** FCD and Diversity are never computed during training. These are the most likely to differentiate models — they need to be computed via evaluate.py at n=1000.

4. **Best candidates for discrimination:** PB Intersection, PB Steric Clash, QED (at n=1000 with CIs), FCD, Diversity.
