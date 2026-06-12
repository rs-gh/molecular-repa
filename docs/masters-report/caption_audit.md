# Caption audit — the reviewer's minimal-data lens

**Date:** 2026-06-12. **Scope:** every `\caption{}` reachable by `make wordcount`.
**Lens:** read each as a reviewer of a paper submission. The question is not "is this
sentence nice?" but **"what is the minimum I need to trust this float and present it to a
fellow reviewer?"** Anything past that is the author narrating to themselves and belongs in
the body or methods.

Word counts are texcount `Sum`. Proposed counts are estimates (~). **Only 6.2 is edited in
place; everything below is proposed text awaiting your approval.**

## What a results caption minimally contains
1. **The claim** — one sentence, and only what *this* float can actually show (a snapshot
   table cannot claim "persistent"; a single-seed panel cannot claim significance).
2. **Non-obvious reading keys** *not already in the table header / axis*: column abbreviations,
   panel (a)/(b) labels, orientation (higher=better), which cells are coloured.
3. **Trust-the-number lineage** — n seeds, sample size (backbones/molecules), checkpoint/step,
   the statistic and what `±`/bands mean, the reference/baseline set, the eval slice.
4. **Cross-refs** to where fuller detail lives.

## What a reviewer does NOT want in a caption
- Justification of design choices ("the latest step all four variants share", "a representative
  point not a transient swing", "the only smoothing available") → methods.
- The result re-narrated in prose when the table already shows it.
- Interpretation / implication / forward-pointers to analysis → body.
- The colour-significance legend → defined once up front, never repeated.
- Magnitude disclaimers repeated across captions → say once.
- Metric definitions already standardised in Ch3.

---

## Summary

| File | Now | Reviewer-min | Δ |
|------|-----|--------------|---|
| `table_rep_quality` **(done)** | 150 | **64** | **−86** |
| `table_rep_quality_afdb` | 156 | ~105 | −51 |
| `table_13m_ranges` | 96 | ~55 | −41 |
| `table_rep_quality_full` | 133 | ~92 | −41 |
| `table_cknna_matrix` | 83 | ~45 | −38 |
| `fig01_fid_des_convergence` | 73 | ~40 | −33 |
| `table_ode` | 127 | ~95 | −32 |
| `fig02_representation` | 78 | ~47 | −31 |
| `table_genrep_corr_afdb` | 89 | ~62 | −27 |
| `table_concentration` | 115 | ~88 | −27 |
| `fig5_1_tabasco_curves` | 80 | ~53 | −27 |
| `fig04_fid_des_gen_vs_rep` | 88 | ~62 | −26 |
| `fig03_alignment` | 79 | ~60 | −19 |
| `table_setup_cath` | 77 | ~60 | −17 |
| `table_ss_composition` | 98 | ~82 | −16 |
| `table_tabasco_gen` | 93 | ~78 | −15 |
| `table_speedup` | 79 | ~70 | −9 |
| `table_genrep_corr` | 70 | ~64 | −6 |
| `table_proteina_13m` | 85 | ~80 | −5 |
| `table_sampler` | 56 | ~53 | −3 |
| `table_tabasco_probe` | 47 | ~45 | −2 |

**Done: −86. Proposed additional: ≈ −466.** Total reachable ≈ **−550 words** with zero loss of
any number a reviewer needs.

**Leave as-is (already minimal, or definitions a reviewer needs to read the float):**
`table_eval_protein` (15), `table_eval_smallmol` (16), `table_profiling_diagnostics` (12),
`table_profiling_mol` (49), `table_profiling_full` (81), `table_profiling_protein` (78),
`table_setup_datasets` (27), `table_setup_model` (20), `table_tabasco_setup` (17). The two inline
method-figure captions in `report-draft.tex` describe schematics — keep.

---

## Proposed minimal captions (verbatim)

Each block: the **cut** (what a reviewer doesn't need), then the proposed replacement text.

### `table_rep_quality_afdb` (156 → ~105)
*Cut:* the "noise-dominated, so we lean on fold" restatement; the long PDB-vs-AFDB methodology
defense compressed to one clause.
> **On AFDB, REPA's representation gain concentrates on fold structure.** AFDB counterpart to
> Table~\ref{tab:proteina-rep}. REPA-GearNet lifts the CATH fold probes as on PDB, so the fold
> routing replicates; the per-residue picture does not — inverse-folding is flat and the dihedral
> probe shows no consistent REPA effect. (n≤256 AFDB-trained; best-layer linear probe at
> n_train=1000 on the cross-database blinded set; mean over the 700K–1.2M window; Δ from baseline;
> n=1 seed. AFDB models are single-seed, so we window-average to smooth per-step noise (the PDB
> tables use a 3-seed 1.0M checkpoint). Random control omitted: it trained only to 500K.)

### `table_13m_ranges` (96 → ~55)
*Cut:* "This lets the reader gauge…" + the worked AFDB-L4-GearNet example (body material).
> **Per-seed min–max spread behind the 1.3M centerpiece (Table~\ref{tab:proteina-13m}).**
> Min–max range across generation seeds 42/1042/2042 (default sde sampler n=0.45), absolute units;
> the centerpiece reports their mean. Columns match the centerpiece: T-W = (fJSD-A, fS-A),
> T-D = pwTM, S-W = ssJSD2D (W), S-D = ssJSD2D (D) and β%. (†single seed, no range.)

### `table_rep_quality_full` (133 → ~92)
*Cut:* the colour-legend sentence (defined up front).
> **Full PDB representation-quality ablation across all six variants.** The remaining
> encoder×depth combinations behind Table~\ref{tab:proteina-rep}, at the same 1.0M checkpoint. The
> encoder-routed pattern holds: GearNet wins the fold probes (CATH C/A/T), ProteinMPNN edges
> inverse-folding (IF). At matched layer 4, trained GearNet (L4-GearNet) beats the random control
> (L4-random) on every probe, so the learned-vs-random gap is not an artefact of injection depth.
> (Baseline absolute, REPA rows Δ-from-baseline, ± half the seed range where >1 seed; n=3 seeds
> except (†) L4-GearNet and L4-MPNN single-seed. Best-layer linear probe at n_train=1000,
> cross-database for IF/dihedral, cleantrain for CATH.)

### `table_cknna_matrix` (83 → ~45)
*Cut:* "Every REPA row exceeds baseline…" (restates bold); "Absolute values small… read the
pattern" (already in `fig03`); redundant "bootstrap medians".
> **Every REPA variant raises per-residue alignment to all three encoders, including the two it
> was never aligned to.** Across-layer peak per-residue CKNNA (bootstrap median, ×10³); superscript
> = peak trunk layer; † = each model's own target. k=10, step 1.0M, n≤256 PDB-trained; n=1 seed.

### `fig01_fid_des_convergence` (73 → ~40)
*Cut:* the "baseline catches up on FPSD after ≈1M, we examine in §X" prose → compressed to a
one-clause honesty caveat (the figure shows the curves).
> **REPA accelerates FPSD in both data regimes, and designability on PDB.** Strongest variant per
> regime; random-encoder control in grey. PDB FPSD converges by ≈1M (§\ref{sec:proteina-convergence}).
> (Mean over 3 seeds, single-seed at trajectory tails; 1,125 backbones/seed, 250 for designability.)

### `table_ode` (127 → ~95)
*Cut:* "Sampling the trunk's learned distribution directly…" → one clause; "is shown to track the
trade-off" gloss.
> **Under deterministic (ODE) sampling, REPA lifts the floor above baseline on AFDB, while on PDB
> it redistributes it.** ODE sampling (no sampler noise) isolates the learned distribution. On AFDB
> the GearNet encoders and L9-MPNN raise both designability and FPSD; L4-MPNN raises FPSD only. On
> PDB the trained encoders raise designability but worsen FPSD across the board — the floor
> redistributes rather than lifts. Designable pwTM = mean pairwise TM over the designable subset
> (higher = more concentrated). (Step 1.3M; REPA cells coloured for Des and FPSD only; n=1 seed;
> 1,125 backbones/seed, 250 for designability. *AFDB L4-random has no checkpoint past 800K.)

### `fig02_representation` (78 → ~47)
*Cut:* "gap that never closes" restates the bold; "mere regularisation from the auxiliary loss"
trimmed.
> **REPA lifts trunk representation quality asymptotically.** REPA variants open a gap over baseline
> that never closes; the random control gains least (genuine transfer, not regularisation).
> (Probe protocol and ablations in Appendix~\ref{app:leakage}; mean over 3 probe seeds where the
> band was run, shaded interval, single-seed elsewhere.)

### `table_genrep_corr_afdb` (89 → ~62)
*Cut:* method restatement → "same method as Table~X".
> **On AFDB too, better trunk representations track better generation.** AFDB counterpart to
> Table~\ref{tab:proteina-genrep-corr} (same partial-Pearson method). Generation is FPSD-AFDB and
> designability; probes read at n_train=1000 on the cross-database blinded set. (n=42 AFDB
> checkpoints pooled over baseline, REPA variants, and control; generation 3-seed mean, single-seed
> for under-trained variants (AFDB L9-GearNet, L4-MPNN) and tails; probe single-seed; 1,125
> backbones/seed, 250 for designability.)

### `table_concentration` (115 → ~88)
*Cut:* "load-bearing" and "no small-sample artefact" editorial — **keep the n=188 vs n=20** (that
is exactly the data a reviewer needs to trust 0.69 vs 0.13).
> **On PDB, REPA concentrates the β-rich designable folds onto fewer modes; on AFDB it does not.**
> The effect is specific to the β≥25 bin: PDB-L9-MPNN climbs 0.29 (400K)→0.69 (1.3M) while its
> baseline diversifies to 0.13; lower-β bins and AFDB-L4-GearNet barely move. Designable-subset
> pairwise TM (lower=more diverse); columns bin by β-strand fraction (%). Cells pool the designable
> subset over all seeds (2–3/cell) and five lengths. The β≥25/1.3M bin holds n=188 (L9-MPNN) vs
> n=20 (baseline); the 400K β≥25 bins (n≈11–19) are directional.

### `fig5_1_tabasco_curves` (80 → ~53)
*Cut:* in-training-estimate caveat compressed to a clause (kept because it explains the Table-X
discrepancy a reviewer would otherwise query).
> **REPA does not accelerate generation quality over the baseline.** On GEOM-drugs, baseline and
> both REPA variants track together and reach near-ceiling on every axis within the first few
> thousand steps, leaving no headroom to fill. (100 molecules/epoch, in-training estimate — values
> differ slightly from the final-checkpoint Table~\ref{tab:tabasco-gen}; n=1 seed, one run per variant.)

### `fig04_fid_des_gen_vs_rep` (88 → ~62)
*Cut:* "x = student CATH-A accuracy, higher to the right" (redundant with axis).
> **At equal compute, our strongest variant achieves better representation and generation.** The
> dashed arrow marks the matched-compute comparison at 400K: against baseline, REPA-L9-MPNN
> (strongest PDB variant) has higher fold accuracy (CATH-A) with lower FPSD (a) and higher
> designability (b). (Markers ≈400K-spaced from 100K, 400K point bordered; generation n=3 seeds,
> single-seed trajectory tails; probe single-seed; 1,125 backbones/seed, 250 for designability.)

### `fig03_alignment` (79 → ~60)
*Cut:* light — keeps the (a)/(b) keys and the single "read the pattern, not the magnitude"
disclaimer (this is its home; removed from `cknna_matrix`).
> **REPA aligns the trunk with domain encoders, including those it was not trained against.**
> (a) Each variant lifts per-residue alignment to its own target above baseline. (b) Alignment is
> cross-encoder: a GearNet-aligned trunk also moves toward ProteinMPNN and ESM2. Absolute alignment
> is small throughout, so read the pattern, not the magnitude. (Step 1M, t=1.0; per-residue CKNNA,
> k=10; bootstrap medians, 5–95% subsample bands; n=1 seed.)

### `table_setup_cath` (77 → ~60)
*Cut:* the "enough classes… learnable" gloss — keep the bucket counts (the data that justifies CATH-A).
> **We read fold-level representation off CATH \emph{architecture}: Class is too coarse, Topology
> too finely bucketed.** Class has 4 buckets (one at 55%; baseline already 0.73); Topology ~1400
> imbalanced buckets; Architecture sits between, so CATH-A is our headline fold metric (C and T in
> the appendix). (Baseline probe acc. is a linear probe on the un-aligned trunk.)

### `table_ss_composition` (98 → ~82)
*Cut:* compress the f_E provenance sentence.
> …(unchanged head)… The strand column f_E underlies the centerpiece β-strand% column
> (Table~\ref{tab:proteina-13m}), here per-seed at 700K. …(unchanged lineage parenthetical)…

### `table_tabasco_gen` (93 → ~78)
*Cut:* the "do not make the comparison unfair" defense → fold the §-ref in.
> …(unchanged head through the MACE tradeoff sentence)… (1,000 sampled molecules; higher-is-better
> except FCD (↓). Baseline trained ~2× longer; quality plateaus early (§\ref{sec:tabasco-results});
> n=1 seed, one run per variant.)

### Light / skip
- `table_speedup` (−9): tighten the acceleration definition only; the method is needed.
- `table_genrep_corr` (−6): "with the raw value alongside" → "(raw alongside)".
- `table_proteina_13m` (−5): centerpiece — the "compare within a regime block" instruction and the
  FPSD-reference note are load-bearing for a reviewer; only cosmetic trims.
- `table_sampler` (−3), `table_tabasco_probe` (−2): already minimal.

---

## Execution
Top 8 (`rep_quality_afdb`, `13m_ranges`, `rep_quality_full`, `cknna_matrix`, `fig01`, `ode`,
`fig02`, `genrep_corr_afdb`) = **−287** for eight edits, plus 6.2's −86 already banked = **−373**
with no reviewer-relevant data lost. The rest is polish.
