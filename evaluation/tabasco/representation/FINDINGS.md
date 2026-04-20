# Tabasco Representation Quality Probes — Findings

**Date:** 2026-04-18  |  **Data:** 1000 mols from GEOM val split  |  **Max atoms:** 39

## Motivation

The REPA paper uses ImageNet linear-probe accuracy as the canonical measure of
representation quality. We have no equivalent. This suite builds two probes
adapted to small-molecule generators, each on its own axis:

- **P3 — atom-type classification** (per-atom, fine-grained chemistry)
- **P4 — RDKit descriptor regression** (per-molecule, mean-pooled, four targets)

Probes are run against ten sources: three frozen encoders (CheMeleon, MACE,
a Dummy coord-MLP baseline) and seven trained checkpoints (one no-REPA baseline
plus six REPA variants spanning both encoders, both combination modes, and
both same/fused hidden-state plumbing).

## Results

See [results.md](results.md) for the full tables. Headline numbers:

### P3 — atom-type (test accuracy)

| source | acc | macro-F1 | notes |
|---|---:|---:|---|
| CheMeleon (frozen) | 1.000 | 1.000 | reference — pretrained foundation model |
| MACE (frozen)      | 0.997 | 0.846 | strong, slightly worse on rare types (Cl/S/Br) |
| Dummy (frozen)     | 0.696 | 0.121 | coord-MLP, no atom-type info — near majority class |
| baseline (no REPA) | 0.999 | 0.993 | model sees atomics as input, so trivially high |
| all REPA variants  | 0.998–1.000 | 0.977–1.000 | no ablation signal here |

**Takeaway.** Atom-type is trivially saturated by any model that takes
atomics as input, so P3 does not distinguish between REPA variants. It's
still useful as a sanity check and for frozen-encoder comparison:
CheMeleon > MACE > Dummy, matching existing FINDINGS in
[../chemeleon/FINDINGS.md](../chemeleon/FINDINGS.md) and
[../mace/FINDINGS.md](../mace/FINDINGS.md).

### P4 — descriptor regression (R² on held-out molecules)

| source | MolWt | LogP | NumRings | NumRotBonds |
|---|---:|---:|---:|---:|
| CheMeleon (frozen) | **0.993** | **0.988** | **0.995** | **0.989** |
| MACE (frozen)      | 0.022 | 0.307 | 0.081 | −0.014 |
| Dummy (frozen)     | 0.822 | 0.160 | 0.432 | 0.345 |
| **baseline** (no REPA) | 0.920 | 0.713 | 0.742 | 0.636 |
| chemeleon_additive_same | 0.938 | 0.672 | 0.746 | 0.600 |
| chemeleon_tradeoff_same | 0.930 | 0.565 | 0.638 | 0.532 |
| chemeleon_additive_fused | **0.943** | 0.669 | 0.736 | 0.606 |
| chemeleon_tradeoff_fused | 0.894 | 0.633 | 0.733 | 0.597 |
| mace_additive | 0.935 | 0.703 | 0.704 | 0.582 |
| **mace_tradeoff** | **0.955** | **0.730** | 0.732 | **0.666** |

**Key observations:**

1. **Baseline descriptor scores are already high** (R² 0.64–0.92). The model
   sees atom coordinates + atom types, so linear probes extract descriptor
   information fairly easily from the student's internal representation.
   This leaves a narrow margin for REPA to improve.

2. **`mace_tradeoff` is the only variant that beats the baseline across three
   of four targets.** MolWt (+0.035), LogP (+0.017), NumRotBonds (+0.030) all
   improved; NumRings essentially tied. MACE aligns the student's hidden
   states toward a genuinely 3D-geometric teacher, which descriptors —
   particularly flexibility-like (RotBonds) — partially depend on.

3. **CheMeleon REPA variants are flat or slightly worse.** Because CheMeleon
   is 2D-only and descriptors are computed from 2D graph structure, the
   teacher's signal overlaps almost entirely with what the baseline already
   learns. Aligning adds noise without new information. The
   _fused_ plumbing (concatenating coord + atom hidden states before the
   projector) edges out the _same_ plumbing on MolWt.

4. **MACE frozen is the opposite shape from CheMeleon frozen**: near-zero
   R² on descriptors yet 0.997 atom-type accuracy. MACE encodes atom identity
   + local geometry; descriptors (molecule-level scalars) are essentially
   orthogonal to its output without a much richer head. This is consistent
   with the "severe bottleneck" finding in
   [../mace/FINDINGS.md](../mace/FINDINGS.md).

5. **Tradeoff > additive for MACE, opposite for CheMeleon.** `mace_tradeoff`
   wins; `chemeleon_tradeoff_*` lose to their additive counterparts. Plausible
   reading: when the teacher (CheMeleon) is saturated and 2D, additive keeps
   the generative loss dominant and folds in a small signal; when the teacher
   (MACE) is 3D-geometric and underused, tradeoff forces the student to
   actually adopt its representation, paying a generative-loss cost that
   translates to better probe numbers.

## Methodology

- **Clean endpoint probing**: hidden states are extracted at `t = 1.0` with
  `x_t = x_1` (real data, no noise), mirroring how REPA itself evaluates
  alignment during training.
- **Linear probes only**: `sklearn.LogisticRegression` for classification,
  `sklearn.Ridge` for regression. No deep heads — staying close to the REPA
  paper methodology so gains are attributable to representation quality,
  not probe capacity.
- **Train/test split**: 80/20 per probe, stratified for classification. Same
  random seed across sources so splits are identical.
- **Mean pooling** for descriptor probe: sum across unmasked atoms, divide by
  atom count. Padding and '*' dummy atoms are excluded at label time.

## Reproducing

```bash
source .venv/bin/activate
export PROJECT_ROOT=$(pwd)/src/tabasco
python playground/tabasco/probes/run_all.py --n_mols 1000
```

Results (machine-readable) → [results.json](results.json), [results.md](results.md).

## Open follow-ups

- **MoleculeNet-style downstream tasks** (BACE, BBBP, Tox21) would test
  whether representations help transfer to tasks the model was never trained
  on. Requires external data — deferred past this canonical pass.
- **Fine-grained layer-wise probing** on the student. Currently we probe
  `hidden_states_coord` at the final layer. Intermediate layers could reveal
  where REPA's signal is absorbed.
- **Conformer-sensitivity probe**. Extend with a pair of near-identical
  conformers; measure cosine sim of mean-pooled student reps. Should be >0.99
  for 2D-aligned students and <0.99 for 3D-aligned (MACE) students.
- **Larger test set**. 1000 mols is enough to separate variants by descriptor
  R², but atom-type macro-F1 is noisy on rare types. 5000+ would stabilise it.
