# n=256 convergence — gen vs rep notes (2026-05-17)

Synthesis of the correlation table and per-pair envelope plots in this
directory.

- Tables: [n256_convergence_gen_vs_rep_correlation.md](n256_convergence_gen_vs_rep_correlation.md), `.csv`
- Figures: `evaluation/proteina/representation/figures/paper/n256_convergence/`
  - `gen_vs_rep_correlation_{pdb,afdb}.png` — 5 rep × 3 gen scatter grid
  - `gen_vs_rep_envelope_per_pair_{cath_C,cath_A,cath_T,if}_top1_vs_{fid_pdb,fid_afdb,designability}.png` — 12 per-pair envelope figures

Generation metrics: FID vs PDB, FID vs AFDB, designability rate.
Representation metrics: CATH-C / CATH-A / CATH-T top1 acc, IF top1 acc, dihedral MAE (all best-layer at t=1.0, n=256 sampling).
Arrow rule in per-pair plots: shared step closest to `TARGET_STEP=400k`.

## Does better representation predict better generation?

**Yes, several pairs survive the step confound.** Partial Spearman ρ (Spearman partial correlation controlling for training step):

- IF top1 → designability on PDB: **+0.60** (p=2e-4)
- Dihedral MAE → designability on PDB: **−0.67** (p=2e-5)
- CATH (any level) → FID (vs PDB or AFDB) on AFDB-trained models: **−0.70 to −0.77** (p≤1e-5)

Most naïve correlations are mostly the step confound — controlling for step typically halves them. Always look at the partial column in [n256_convergence_gen_vs_rep_correlation.md](n256_convergence_gen_vs_rep_correlation.md).

**Two notable sign reversals:**
- PDB dihedral MAE ↔ FID is *positive* (lower MAE → higher FID). Driven by REPA-MPNN runs achieving low MAE but middling FID.
- AFDB IF top1 ↔ FID is *positive*. Opposite of PDB. Suggests IF top1 is not a robust proxy across reference distributions; MPNN-target REPA runs (where the target is residue identity) inflate IF without proportionally improving FID.

**Dataset asymmetry:** dihedral MAE predicts designability on PDB but not AFDB; CATH-T predicts FID on AFDB but not PDB. No single rep proxy is universal across datasets in this sweep.

**Across CATH levels:** all three (C, A, T) give nearly identical partial ρ on AFDB FID (−0.70 to −0.74). They track the same underlying signal — fineness of CATH level is not what's doing the work.

## Which REPA config is best?

No single winner — depends on what you're optimizing for.

### Mid-training speedup (REPA paper's central claim) → **REPA L9 GearNet**

At step 400k on PDB, ΔFID vs baseline (from per-pair envelope):
- L9 GearNet: **ΔFID = −138** ← largest by far
- L9 MPNN: ΔFID = −52
- L4 GearNet / L4 MPNN: positive (hurts FID early)

L9 (deep-layer) variants are the only ones showing the dramatic "REPA reaches baseline-converged FID much earlier" story. L9 GearNet is the cleanest example of the convergence-speedup phenomenon in our data.

### Final-state generation on PDB → **REPA L4 MPNN**

End-of-training designability:
- baseline @ 1500k: 0.612
- L4 MPNN @ 1600k: **0.713** ← best
- L9 GearNet @ 900k: 0.612 (tied with baseline)
- L4 GearNet @ 900k: 0.516 (worse than baseline)
- L9 MPNN: only trained to 400k on PDB — can't compare

But L4 MPNN's mid-training behavior is *worse* than baseline (hurts FID at 400k, only pulls ahead late). Opposite of the speedup narrative.

### On AFDB → **L9 GearNet** wins more consistently

All four REPA variants improve over baseline on FID. L9 GearNet has the strongest CATH-T / IF / dihedral correlations with generation quality (partial Spearman ρ ≈ −0.7).

### Other observations

- **L4 GearNet is a clear loser on PDB** across the board (worse FID at 400k, worse designability at 900k).
- **Random-init GearNet (L4 GearNet-rand)** regresses toward baseline — confirms that encoder weights matter, not just the auxiliary loss structure.

## Key data gap

**REPA L9 MPNN on PDB only goes to step 400k.** Based on AFDB behavior, L9 MPNN at full training could plausibly be the best variant overall. Worth finishing that run before drawing final paper conclusions if compute allows.

## One-line recommendation

L9 GearNet for the speedup story; extend L9 MPNN on PDB before the paper if at all possible.
