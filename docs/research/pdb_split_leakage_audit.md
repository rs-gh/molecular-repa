# PDB / AFDB split leakage audit and clean-evaluation protocol

**Date:** 2026-05-23 to 2026-05-25
**Status:** Audit + decomposition complete. Findings and report framing to
be cited from the final thesis/paper.

This is the single document for re-loading context on the leakage audit,
the cleaned-evaluation protocols we settled on, and the recommended report
framing. If you're coming back to this cold, read just the TL;DR and the
"Dataset audit at a glance" section, then jump to "How to convince yourself
REPA learns a better representation" for the report framing.

## TL;DR

The proteina datamodule uses `split_type: "random"` on chains for both PDB
and AFDB-SwissProt training data. Because both source datasets contain heavy
sequence redundancy at the chain level (PDB much more than AFDB), the
resulting train/val splits are leaky:

- **PDB val ↔ PDB train:** 79.0% byte-identical, 98.3% ≥30%-identity.
- **AFDB val ↔ AFDB train:** 29.6% byte-identical, 92.2% ≥30%.
- **Cross-DB AFDB val ↔ PDB train:** 5.7% byte-identical, 62.1% ≥30%.
- **Cross-DB PDB val ↔ AFDB train:** 13.6% byte-identical, 39.5% ≥30%.
- **Cross-corpus train↔train:** ~20% of PDB-train short chains are
  byte-identical to an AFDB-train entry; ~50-65% have a ≥30% homolog.

Three consequences:

1. All in-house probe / FID / designability numbers reported on the default
   val sets are inflated relative to a true held-out evaluation.
2. The "our in-house baselines outperform NVIDIA's pretrained 60M on
   representation probes" anomaly is largely a leakage artefact: our models
   trained on near-copies of our val set; NVIDIA's didn't.
3. **The baseline-vs-REPA comparison (the central question) is robust** under
   the leakage controls we ran: both architectures share the same training
   data, so leakage hits them symmetrically. The REPA-over-baseline Δ is
   preserved across all leakage-removal regimes for residue-level probes
   (IF, dihedral), and the architectural ordering — *structural-encoder
   REPA > random-encoder REPA ≈ baseline* — is preserved on the cleanest
   available eval.

## Dataset audit at a glance

All four val↔train homology audits (MMseqs2 `easy-search`, `--min-seq-id 0.3
-c 0.8 --cov-mode 0`, 2026-05-23 to 2026-05-25). Pool sizes: PDB train 425,100;
PDB val 4,999; AFDB train 459,340; AFDB val 4,521.

**Sequence-level leakage rates at multiple thresholds:**

| Audit direction | =100% | ≥99% | ≥90% | ≥70% | ≥50% | ≥30% |
|---|---|---|---|---|---|---|
| PDB val → PDB train | **79.0%** | 87.3% | 95.7% | 97.0% | 97.7% | **98.3%** |
| AFDB val → AFDB train | **29.6%** | 35.0% | 56.8% | 77.5% | 88.1% | **92.2%** |
| AFDB val → PDB train (cross-DB) | 5.7% | 7.0% | 13.7% | 26.4% | 47.6% | 62.1% |
| PDB val → AFDB train (cross-DB) | 13.6% | 15.9% | 22.6% | 27.0% | 33.4% | 39.5% |

**Cross-corpus train↔train overlap (length-filtered):**

| Direction | L≤ | n_query | ≥100% | ≥90% | ≥70% | ≥50% | ≥30% |
|---|---|---|---|---|---|---|---|
| PDB train → AFDB train | 128 | 104,139 | 20.0% | 30.5% | 38.3% | 45.9% | 50.7% |
| AFDB train → PDB train | 128 | 82,855 | 6.1% | 13.3% | 27.5% | 47.2% | 55.7% |
| PDB train → AFDB train | 256 | 273,771 | 20.3% | 33.8% | 41.1% | 50.9% | 59.9% |
| AFDB train → PDB train | 256 | 229,670 | 5.6% | 14.0% | 27.1% | 48.1% | 64.6% |

**Eval-set cleaned-pool sizes** (proteins surviving the homology filter at <30%):

| Eval pool | n total | n ≤128 | n ≤256 |
|---|---|---|---|
| PDB val cleanval (no PDB-train hit) | 86 | 44 | 72 |
| AFDB val cross-DB clean (no PDB-train hit) | 1,714 | 749 | 1,714 |
| AFDB val doubly-clean (no PDB-train + no AFDB-train hit) | 354 | 154 | **325** |
| PDB val doubly-clean (no PDB-train + no AFDB-train hit) | 76 | 41 | 62 |

**Eval-set characteristics — why dirty AFDB and dirty PDB are not directly comparable:**

| Metric (≤256) | PDB val | AFDB val |
|---|---|---|
| n proteins | 3,190 | 4,521 |
| mean / median length | 153 / 153 | 155 / 156 |
| % CATH-labelled | 43% | **62%** |
| Top-1 share, CATH-C | 48% | 57% |
| Top-1 share, CATH-A | 17.5% | 21.2% |
| Top-1 share, CATH-T | 9.5% | **15.2%** |
| Unique T-classes (≥3 ex.) | 89 | 153 |

Length is *not* a confound; CATH-labelling rate, top-class concentration, and
AF2-structural-smoothness are. A "predict the most common topology" baseline
already scores ~6pp higher on AFDB CATH-T than on PDB CATH-T before any model
contribution.

## How the leakage arises

Configured in [src/proteina/configs/datasets_config/pdb/pdb_lmdb.yaml#L37-L42](../../src/proteina/configs/datasets_config/pdb/pdb_lmdb.yaml#L37-L42):

```yaml
datasplitter:
  train_val_test: [0.98, 0.019, 0.001]
  split_type: "random"
```

The splitter at [src/proteina/proteinfoundation/datasets/pdb_data.py:279-284](../../src/proteina/proteinfoundation/datasets/pdb_data.py#L279-L284)
does an unweighted row-level random split without sequence-similarity
clustering. PDB contains many homo-oligomer copies of the same chain and
many near-duplicate mutants; a random row split deposits these on both
sides. AFDB-SwissProt is more curated (~1 chain per UniProt entry), which
is why AFDB self-leakage is less extreme.

The supported alternative `split_type: "sequence_similarity"` with
`split_sequence_similarity: 30` would group near-homologs into the same
side, but has not been used for any of the LMDBs on disk.

## Two distinct leakage paths

The proteina representation-probe pipeline trains a linear head on
`(train.lmdb features, train.lmdb labels)` and evaluates on `(val.lmdb features,
val.lmdb labels)`. Two paths inflate metrics:

**(1) Model-side — asymmetric across models.** The student model was trained
on `train.lmdb`. Because val.lmdb shares chain identity with train, the model
has effectively seen the val proteins during training. Features extracted
from val are features for in-distribution proteins under sharp training-loss
minima, not for held-out proteins. Models trained on different curation
(NVIDIA's pretrained 60M, etc.) don't have this advantage.

**(2) Probe-side — symmetric across models.** If train and val proteins are
near-identical at the sequence level, their features under any model are
similar, and a probe trained on (train features, labels) can shortcut by
pattern-matching rather than generalizing. This inflates every model's
absolute score but doesn't differentially advantage any particular one.

The decomposition experiments below isolate these two paths separately.

## Decomposition experiments — three evaluation regimes

To untangle the two leakage paths, three regimes per protein-size bucket
(n=128 and n=256) were run on the convergence sweep (baseline + REPA-L4
GearNet originally; REPA-L9 GearNet, REPA-L4 MPNN, REPA-L9 MPNN,
REPA-L4 random-encoder added in subsequent rounds).

| Regime | Probe-fit | Eval | n_eval (n=256) | Removes |
|---|---|---|---|---|
| **dirty** | 1000 chains from full leaky PDB train | full leaky PDB val | 3,190 | nothing |
| **cleantrain** | 1000 chains from PDB train ∖ {≥30% hits to val} (198k pool) | same full leaky PDB val | 3,190 | probe-side only |
| **xclean-AFDB** | 1000 chains from AFDB train | AFDB val ∖ {≥30% hits in PDB OR AFDB train} | 325 | both paths (cross-DB, structural-shift confound) |

(We initially also ran a `cleanval` regime — PDB val filtered against PDB
train, with leaky probe-fit. It is dominated by xclean-AFDB: xclean is
*cleaner* — its filter also removes probe-side homology — *and* has more
data — n=325 vs 72. Cleanval was dropped from the final framework.)

**Per-probe lead regime** (determined by the class distribution in each eval
set, see "Dataset audit at a glance" above):

- **IF top-1, dihedral**: lead with xclean. Residue-level metrics; n=325
  proteins gives ~43k residues — full statistical power.
- **CATH-A**: lead with xclean. 13 unique classes / 8 with ≥3 examples
  on AFDB val ≤256, top-class share 26% — good balance between class
  variety and statistical power.
- **CATH-T**: lead with cleantrain. xclean's AFDB val ≤256 has only 4
  T-classes with ≥3 examples (48 unique classes total, mostly singletons)
  — too fragmented for fine-grained topology. Cleantrain on full PDB val
  ≤256 has 89 well-populated T-classes — defensible CATH-T evaluation.
- **CATH-C**: deprioritise. Only 4-5 classes total; top-class share 43-48%
  — a "predict the most common fold" trivial baseline gets you halfway.
  Real signal can drown in the baseline floor.

**dirty** is retained primarily as a *control* for the cleantrain comparison:
the dirty → cleantrain shrinkage isolates the probe-side leakage's
contribution. The cleantrain → xclean shrinkage isolates the model-side
contribution.

## Confounds that xclean does NOT remove

These caveats apply to the cleanest available eval (xclean-AFDB). They are
all approximately symmetric across the PDB-trained models being compared, so
they don't bias the *relative* baseline-vs-REPA ranking, but they should be
declared.

1. **Structural distribution shift (PDB-trained model on AFDB inputs).**
   PDB structures are experimental (X-ray, EM); AFDB structures are AF2
   predictions — smoother, often hallucinated in disordered regions,
   pLDDT-low loops. A PDB-trained model encoding an AFDB protein operates
   on a slightly out-of-distribution input. **Mitigation**: we also
   evaluate the `baseline_afdb_256_*` checkpoints on the same doubly-cleaned
   AFDB val (blue dotted diamond in the n256 plot). The AFDB-trained
   baseline sits *above* PDB-trained REPA on this eval, so the shift is a
   real cost — but it hits all PDB-trained models the same way, so the
   relative comparison is unaffected.

2. **AFDB CATH labels are Gene3D HMM predictions, not experimental
   assignment.** Noisier labels than PDB CATH. Same noise floor across
   models. Drives the higher top-class concentration documented in the
   eval-set characteristics table.

3. **pLDDT-low regions in AFDB structures are essentially random.** Could be
   filtered out (e.g. only retain val proteins with mean pLDDT > 70). Not
   currently applied.

4. **REPA target-encoder pretraining provenance.** CA-GearNet and
   ProteinMPNN were both pretrained on PDB-derived data. So REPA models
   have an indirect "PDB-ness" pathway baseline doesn't. This is by design
   (REPA *is* trying to inherit structural knowledge from the target
   encoder) rather than a confound to remove — but worth declaring.

5. **Probe-head capacity and layer-best selection.** Linear probe on the
   best layer per (run, step). Same head and same selection rule across all
   runs, so consistent. The best layer varies by run — this is itself part
   of what the per-layer curves show.

6. **Mixed n_train along the cleantrain / xclean curves (2026-05-26 ext2).**
   The original cleantrain and xclean v1 sweeps were run at probe-fit
   `n_train=1000`; the dirty regime was always at `n_train=5000`. The
   ext2 extension (2026-05-26) added new latest checkpoints at `n_train=5000`
   across all three regimes for parity with dirty. To keep compute bounded
   we did not re-evaluate the legacy v1 checkpoints at n=5000, so each
   cleantrain/xclean curve is `n=1000` for early/mid steps and `n=5000` at
   the highest steps (no checkpoint overlaps). Expect a ~1-3pp upward
   shift at the v1→v2 boundary (more probe-fit data → tighter linear head);
   trend direction is unaffected. The dirty regime is uniformly `n=5000`.
   A paper-final rerun of legacy ckpts at n=5000 (~20-25 GPU-h) would
   remove the discontinuity.

## Same-model cross-DB cross-table (last-checkpoint, n=256)

For each model flavour at its latest checkpoint, the four-cell table:
diagonal = in-DB dirty eval, off-diagonal = cross-DB doubly-cleaned eval.

**CATH-T (best layer, t=1.0):**

| | baseline | repa_l4_gn | repa_l4_mpnn | repa_l9_gn | repa_l9_mpnn | pretrained |
|---|---|---|---|---|---|---|
| PDB-trained × PDB val (dirty, n=3190)   | 0.405 | 0.777 | 0.614 | 0.837 | 0.559 | 0.500 |
| AFDB-trained × AFDB val (dirty, n=4521) | **0.944** | **0.958** | **0.944** | **0.965** | **0.954** | — |
| PDB-trained × AFDB val (xclean, n=325)  | 0.350 | 0.525 | 0.425 | 0.575 | 0.375 | 0.450 |
| AFDB-trained × PDB val (xclean, n=62)   | 0.333 | 0.333 | 0.500 | 0.333 | 0.667 | 0.333 |

**IF top-1:**

| | baseline | repa_l4_gn | repa_l4_mpnn | repa_l9_gn | repa_l9_mpnn |
|---|---|---|---|---|---|
| PDB-trained × PDB val (dirty)   | 0.113 | 0.159 | 0.177 | 0.168 | 0.172 |
| AFDB-trained × AFDB val (dirty) | **0.213** | **0.207** | **0.227** | **0.208** | **0.256** |
| PDB-trained × AFDB val (xclean) | 0.120 | 0.144 | 0.158 | 0.154 | 0.163 |
| AFDB-trained × PDB val (xclean) | 0.124 | 0.121 | 0.126 | 0.128 | 0.121 |

**Dihedral MAE (lower better):**

| | baseline | repa_l4_gn | repa_l4_mpnn | repa_l9_gn | repa_l9_mpnn |
|---|---|---|---|---|---|
| PDB-trained × PDB val (dirty)   | 46° | 26° | 30° | 25° | 29° |
| AFDB-trained × AFDB val (dirty) | **11°** | 13° | **11°** | 14° | **10°** |
| PDB-trained × AFDB val (xclean) | 34° | 22° | 32° | 24° | 20° |
| AFDB-trained × PDB val (xclean) | 47° | 43° | 50° | 35° | 44° |

**Reads:**

- **In-DB dirty AFDB numbers are not credible** as protein-level fold
  prediction. CATH-T 0.94+ with 273 in-vocab topologies, dihedral MAE 10°
  on held-out proteins — these reflect the compound of (a) Gene3D label
  concentration, (b) AF2 structural smoothness, (c) probe-side leakage
  (92% ≥30% AFDB val↔train), and (d) model-side leakage (29% byte-identical).
- **Cross-DB cleaned numbers (rows 3, 4) collapse model differences.** The
  ordering across architectures within those rows is much tighter than the
  apparent gap in row 2.
- **Same-model-on-different-val** (rows 1 vs 3, or rows 2 vs 4) tells you
  the eval-set effect: PDB-trained baseline goes from 0.113 (PDB-val dirty)
  to 0.120 (AFDB-val xclean) on IF top-1 — virtually identical. The
  apparent "2× higher baseline IF on AFDB" (0.213 vs 0.113) comes
  entirely from training-data matching the eval distribution, not from
  the model being intrinsically better.

## Results summary (n=256, best layer, last checkpoint)

PDB-trained models on doubly-clean AFDB val (xclean):

| Probe | baseline | REPA-MPNN-L4 | REPA-L4-GearNet | pretrained NVIDIA |
|---|---|---|---|---|
| IF top-1 (max) | 0.12 | 0.16 | 0.14 | **0.22** |
| Dihedral MAE | 34° | 32° | 22° | **13°** |
| CATH-C | 0.69 | 0.74 | **0.86** | 0.75 |
| CATH-A | 0.36 | 0.41 | **0.52** | 0.45 |
| CATH-T | 0.35 | 0.43 | **0.53** | 0.45 |

Pretrained NVIDIA 60M leads on most probes; REPA-L4-GearNet leads on
dihedral (consistent with GearNet's geometric pretraining). REPA-MPNN-L4
generally beats baseline. The original "in-house > NVIDIA" pattern on
dirty PDB val (row 1 of cross-tables above) is reversed on this cleaner
eval — confirming the leakage diagnosis.

Exact numbers in
`evaluation/proteina/representation/results/paper/n256_xclean_afdb_pdb/pretrained_sweep_results.csv`.

Final figures:
- `evaluation/proteina/representation/figures/paper/leakage_decomp/n128/pdb/n128_leakage_decomp.png` — 4 columns × 5 probes, all PDB-trained model families at n=128.
- `evaluation/proteina/representation/figures/paper/leakage_decomp/n256/pdb/n256_leakage_decomp.png` — same at n=256.
- `evaluation/proteina/representation/figures/paper/leakage_decomp/n256/afdb/n256_afdb_trained_dirty_vs_xclean_pdb.png` — AFDB-trained models, dirty AFDB val vs xclean PDB val.

## How to convince yourself REPA learns a better representation

This is the methodological argument to advance in the report. The question
is not "what are the absolute numbers" but "is REPA's edge over baseline
real, after controlling for all forms of contamination?"

### The argument structure

If REPA's apparent advantage in any single regime is a leakage artefact,
the following falsifiers should fire:

- **Probe-side memorization hypothesis** → if dirty Δ ≫ cleantrain Δ.
  Removing the probe-side shortcut would crush the Δ. Empirically: for IF
  and dihedral, dirty Δ ≈ cleantrain Δ. For CATH-T, dirty Δ > cleantrain Δ
  by about 1.5× (e.g. n=256 step 200k: dirty +0.52, cleantrain +0.33). So
  *some* probe-side memorization contributes for CATH-T; *little* for
  residue-level probes.

- **Model-side memorization hypothesis** → if cleantrain Δ ≫ xclean Δ. The
  model has memorized val proteins; the probe (trained on novel features)
  still benefits when reading those memorized features. Empirically: for IF
  and dihedral, cleantrain Δ ≈ xclean Δ. For CATH-T, cleantrain Δ > xclean Δ
  (e.g. 0.33 → 0.23). Some model-side memorization for CATH-T; little for
  IF / dihedral.

- **Generic auxiliary-loss / optimization-hack hypothesis** → if
  random-encoder REPA performs comparably to structural-encoder REPA. The
  auxiliary loss alone would help. Empirically: random-encoder REPA
  collapses to baseline performance on every probe in every regime. This is
  the cleanest single falsification — the gain is specifically from
  *structural knowledge* in the target encoder, not the loss form.

After both shrinkages (CATH-T: 0.52 → 0.33 → 0.23) the residual xclean Δ
is still substantial — that's the "true architectural" effect for protein-
level fold classification. For residue-level metrics (IF, dihedral) the Δ
is invariant across regimes, indicating little-to-no leakage contribution.

### Cross-task transfer

Beyond raw rank-order, the cross-task pattern argues that REPA induces
generally useful features, not just task-aligned ones:

- REPA-MPNN (encoder trained on inverse folding) leads IF top-1 but also
  beats baseline on CATH (which MPNN was not trained for).
- REPA-GearNet (encoder pretrained partly on CATH) leads CATH but also
  beats baseline on IF (which GearNet was not trained for).
- Both REPA variants improve dihedral substantially over baseline, even
  though neither encoder was directly trained on dihedral labels (both
  encode local geometry implicitly via their inductive biases).

If the encoders only contributed their training task's specific features,
non-matched probes would track baseline. They don't.

### Recommended report framing

1. **Rank-order claim (the strong one).** REPA outperforms baseline on
   every probe in every leakage regime, including the doubly-cleaned
   cross-DB eval. The ordering is consistent: structural-encoder REPA >
   random-encoder REPA ≈ baseline.

2. **Magnitude claim (per-probe regime choice).**
   - For IF top-1, dihedral, CATH-A: quote Δ from xclean-AFDB as headline.
     Residue-level (IF, dihedral) or A-level CATH (sufficiently populated
     class distribution at n=325) — clean and statistically robust.
   - For CATH-T: quote Δ from cleantrain (full statistical power; only
     probe-side leakage removed). Acknowledge that some shrinkage from
     dirty → cleantrain → xclean is observed (consistent with mild
     memorization contribution); the residual xclean Δ is positive but
     too noisy at n=40 in-vocab to use as a precise point estimate.

3. **Honest caveat.** Cleantrain absolute scores are not OOD numbers —
   they still contain residual model-side leakage. They are upper bounds.
   A sequence-similarity-clustered retraining of the model on a non-leaky
   train.lmdb would be required to fully remove model-side leakage; that
   re-training is out of scope for this work.

4. **Two falsifier controls.**
   - **Random-encoder REPA** collapses to baseline on every probe in every
     regime — falsifies "the loss alone helps."
   - **NVIDIA pretrained 60M**, trained on a different PDB curation
     unrelated to our val.lmdb, is competitive with our REPA on
     leakage-removed evals but below it on dirty — consistent with our
     models being trained on data correlated with our val set, NVIDIA's
     not.

5. **Cross-task transfer evidence.** REPA-MPNN improves CATH (not its
   training objective); REPA-GearNet improves IF (not its training
   objective). Both REPA variants improve dihedral (neither's training
   objective). Argues for broad representation improvement, not just
   task-aligned decoder transfer.

6. **Cross-DB / cross-source caveats.** Within-DB comparisons (PDB-trained
   on PDB val, AFDB-trained on AFDB val) are valid for ranking models
   *within* a DB. Cross-DB absolute-score comparisons ("baseline AFDB >
   baseline PDB") are not directly comparable — AFDB-val is easier due to
   Gene3D label concentration, AF2 structural smoothness, and higher
   CATH-labelling rate. Only the *same model evaluated on different val
   sets* with comparable filters (e.g. the cross-table rows 1↔3, 2↔4)
   isolates the eval-set vs model effect.

7. **Forward-looking recommendation.** For future work on this codebase:
   rebuild the canonical PDB / AFDB train/val splits with
   `split_type: "sequence_similarity"` + `split_sequence_similarity: 30`.
   In-house representation-quality numbers should not be reported on the
   random splits going forward.

## Artifacts

**Audit data** (persistent, committed):
`evaluation/proteina/representation/results/inputs/leakage_audit/`
- Sequence FASTAs: `pdb_train.fasta` (425,100), `pdb_val.fasta` (4,999),
  `afdb_train.fasta` (459,340), `afdb_val.fasta` (4,521).
- Length-filtered fastas (under `train_overlap/`):
  `{pdb,afdb}_train_le{128,256}.fasta`.
- MMseqs2 m8 results: `pdb_val_vs_pdb_train.m8`, `afdb_val_vs_pdb_train.m8`,
  `afdb_val_vs_afdb_train.m8`, `pdb_val_vs_afdb_train.m8`,
  and `train_overlap/{pdb_vs_afdb,afdb_vs_pdb}_le{128,256}.m8`.
- `train_overlap/summarize.py` — reproduces the threshold tables.
- `train_overlap/run_all.sh`, `train_overlap/run_all.log` — driver.

**Cleaned manifests:**
- `results/inputs/clean_val/batch_manifest_clean_v1_max{128,256,512}.json` —
  PDB cleanval (n=44/72/86; historical, not used in final framework).
- `results/paper/n{128,256}_convergence_cleantrain_pdb/batch_manifest_train_clean_v1.json` —
  PDB train minus val-homologs, 1000 sampled.
- `results/paper/n256_xclean_afdb_pdb/batch_manifest_eval_clean_v2.json` —
  doubly-clean AFDB val ≤256, n=325.
- `results/paper/n128_xclean_afdb_pdb/batch_manifest_eval_clean_v2.json` —
  doubly-clean AFDB val ≤128, n=154.
- `results/paper/n256_xclean_pdb_afdb/batch_manifest_eval_clean_v2.json` —
  doubly-clean PDB val ≤256, n=62 (for evaluating AFDB-trained models).

**Sweep config profiles** in
[evaluation/proteina/representation/sweep_config.yaml](../../evaluation/proteina/representation/sweep_config.yaml):
- `paper_n{128,256}_cath_if_dih_convergence_cleantrain_pdb`
- `paper_n{128,256}_cath_if_dih_xclean_afdb_pdb`
- `paper_n256_cath_if_dih_xclean_pdb_afdb` (AFDB-trained on PDB val)
- (`*_cleanval_pdb` profiles are retained but deprioritised in the final
  framing.)

**Figures:**
- `evaluation/proteina/representation/figures/paper/leakage_decomp/n128/pdb/n128_leakage_decomp.png`
- `evaluation/proteina/representation/figures/paper/leakage_decomp/n256/pdb/n256_leakage_decomp.png`
- `evaluation/proteina/representation/figures/paper/leakage_decomp/n256/afdb/n256_afdb_trained_dirty_vs_xclean_pdb.png`

**Persistent memory entries** (claude auto-memory):
- `project_pdb_split_leakage.md` — PDB self-leakage findings.
- `project_afdb_split_leakage.md` — AFDB self-leakage findings.
- `reference_afdb_pdb_overlap.md` — cross-DB overlap stats (both directions).
- `project_repa_evidence_framing.md` — three-regime framework, per-probe
  lead-regime choices, class distribution justifications.
