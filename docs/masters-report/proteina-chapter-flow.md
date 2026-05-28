# Proteina chapter (Ch~\ref{ch:proteina-study}) — flow / figure / claim register

Working doc for the Proteina study chapter. We iterate here, then port to
`report-draft.tex`. Iteration may interleave (doc ↔ tex). Last updated 2026-05-28.

Companion to the findings docs — this is the *report-construction* layer, those
are the *evidence* layer:
- [proteina_narratives.md](../research/proteina_narratives.md) — the claims + evidence
- [proteina_narratives_handoff_2026-05-27.md](../research/proteina_narratives_handoff_2026-05-27.md)

## Chapter numbering (use \ref labels, never hard-coded numbers)

Current draft resolves to: Ch1 Intro · Ch2 Background+Related (merged) ·
**Ch3 Evaluation** (`ch:evaluation`) · **Ch4 Profiling** (`ch:profiling`) ·
**Ch5 Tabasco** (`ch:tabasco-study`) · **Ch6 Proteina** (`ch:proteina-study`) ·
Ch7 Conclusions (`ch:conclusions`). (The 8-chapter spine comment is stale —
Background and Related Work were merged.)

## Format conventions

Each paragraph is planned as a **topic sentence (the actual proposed opener) +
the claim it makes + coverage + confidence + figure/table it leans on.** Prose
flows as paragraphs, not bullets, when ported.

- **Confidence**: ✓✓ strong, multi-regime · ✓ solid, often single-regime ·
  ⚠ couched / coverage-limited
- **Coverage**: which {encoder × dataset × scale × sampler} the claim survives.
  Default everything to **n256** (headline regime); n128 is a scale-check only.
- **Figure/table status**: `EXISTS` · `EXISTS✎` (exists, needs redesign) ·
  `PLOT` (data on disk, no figure yet) · `ANALYSIS` (compute needed) ·
  `DATA` (must run jobs) · `OPT` (optional / nice-to-have)
- **Headline-metric rule**: headline *figures* index on **{FID, designability}**
  (FID = distribution-match flagship, designability = quality flagship); the
  full super-column suite lives in *tables*. Picking few metrics for figures is
  deliberate — we have many and each captures a different facet.

---

# Part 0 — Cross-chapter dependencies (what must land before Ch6 reads)

Ch6 assumes the reader already owns three things. Listed here so we build them
in the right home and Ch6 doesn't re-introduce.

## Ch3 (Evaluation, `ch:evaluation`) must establish
- **The generation-quality super-column taxonomy.** Five groups:
  *quality* (designability%, scRMSD, pLDDT) ·
  *tertiary-whole* T-W (FID-PDB/AFDB, fJSD-A/T/C, fS-A/fS-C) ·
  *tertiary-designable* T-D (#clusters, pwTM on designable subset) ·
  *secondary-whole* S-W (ssJSD-2D, H/E fractions over whole set) ·
  *secondary-designable* S-D (SS composition / β-rich % of designable subset).
  Note the Proteina-paper trade-offs (designability ↔ diversity) here. This
  taxonomy is **load-bearing**: Ch6's headline trade-off is literally "T-W up,
  T-D down", legible only if T-W vs T-D is already a reader primitive.
- **Seed of the imaging-vs-molecular argument** (item 3; see Notes). The
  content–geometry *coupling* difference + discrete/continuous duality →
  representation and generation-quality are both *multi-factorial* in molecules
  → no single number suffices → motivates the super-columns; mode collapse is
  harder to reason about. Pay-off deferred to Ch6 / Ch7.
- **PDB vs AFDB as two data regimes** (partial; the "what"): PDB = experimental
  & scarce; AFDB = AF2-predicted, abundant, more in-silico-designable
  [NEEDS-CITATION]. The "why we study both" reinforced in Ch6 §6.1.

## Ch4 (Profiling, `ch:profiling`) must establish
- **The encoder-target metric set** — *its own instrument*, distinct from the
  Ch6 student battery (they overlap only on CATH + IF):
  RankMe / participation-ratio / effective-rank (Q3 trainability),
  projector-gap "headroom" (Q2 reachability),
  linear-probe recoverability principle (Q1 information content).
  Source: [encoder_profiling/proteina/FINDINGS.md](../../encoder_profiling/proteina/FINDINGS.md).
- **Per-encoder verdicts + predictions**: CA-GearNet usable (~0.35 headroom),
  ESM2 mid-layers usable but saturated (~0.08), MC-GearNet-Edge unusable
  (eff-rank 1.1/3072). The chapter ends with *predictions* Ch6 then tests.
- **The probe principle Ch6 reuses**: Ch4 introduces "linear-probe
  recoverability of structural labels"; Ch6 instantiates it on the *student*
  with a fuller battery (adds dihedral, contact, distance). Continuity is real
  (CATH+IF in both); Ch6 says "same principle, finer probes, applied to the
  student."

**Open placement question**: does the *student* rep-quality battery
(CATH/IF/dihedral/contact/distance) get introduced in Ch3, Ch4, or inline in
Ch6? Recommendation: **inline in Ch6 §6.2 at first use** — matches REPA's habit
(it introduces its linear probe + CKNNA at Fig2/3, no separate metrics chapter),
and the battery is protein-/student-specific. Revisit if Tabasco (Ch5) needs the
same battery — if so, promote to a shared methods home.

---

# Part 1 — Figure & table register (the visual-asset plan)

REPA-paper analogue in brackets. "Index" = which §6.x uses it.

## Figures

| # | What it shows | REPA-analogue | Status | Index | The one-line claim (caption seed) |
|---|---|---|---|---|---|
| **6.1 HEADLINE** | FID vs training step, baseline vs REPA, **2-panel PDB + AFDB** | Fig 1 | `PLOT` (data + early plots) | §6.4 | "On AFDB the gap never closes (durable, 6.5–13×); on PDB REPA leads early then the long baseline catches up." |
| 6.2 (problem) | Student rep quality (CATH-T, IF) vs **training step**, **baseline only**, + NVIDIA-60M ceiling | Fig 2a | `PLOT` | §6.2 | "Flow-matching alone never makes Proteina's hidden states more structural — baseline is flat across 1.8M steps." |
| 6.3 (solution) | Same axes, **+ REPA variants** rising; encoder rank-order | Fig 3a | `PLOT` | §6.2 | "REPA makes the student's reps structural, as a persistent gap; GearNet > MPNN > random." |
| 6.4 CKNNA | Per-layer CKNNA, baseline (noise floor) vs REPA (L8 peak); + off-diagonal panel (align→GearNet also lifts ESM2) | Fig 2b/3b | `EXISTS✎` (heatmaps; want line version) | §6.3 | "Alignment is REPA-induced, peaks mid-stack, and generalises across encoders (Platonic)." |
| 6.5 gen-vs-rep | Envelope/correlation: x = best-layer rep metric, y = gen metric, points = (run,step) | Fig 3c | `EXISTS` | §6.5 | "Representation quality and generation quality co-move; the link is encoder-matched." |
| 6.6 trade-off | T-D (designable #clusters / pwTM) vs training step, baseline vs REPA, by encoder, PDB+AFDB; + designable-vs-whole pwTM gap growing | (none — our finding) | `ANALYSIS` (data on disk) | §6.6 | "REPA concentrates designable samples onto few encoder-preferred folds; gated by encoder×dataset; MPNN-AFDB is the falsifier." |
| 6.7 gallery | Generated backbones, baseline vs REPA (optionally over steps) | Fig 4/6 | `DATA` `OPT` | §6.1/§6.6 | illustrative ("what the model makes"); upgrade to show concentration *if* visually obvious |
| 6.8 rep-vs-t | Rep quality / CKNNA vs timestep t, baseline vs REPA | Fig 7 | `DATA` `OPT` `LOW` | §6.3 | "REPA's rep advantage holds across noise levels" — only if cheap |

Note on 6.2/6.3 axis: REPA's 2a/3a are per-*layer* at fixed iter; **we use
per-*training-step*** because our signal is the over-training divergence
(baseline flat / REPA rising). We *can* also show per-layer (CKNNA matrix has
all 10 layers) but the step-axis is the money shot. **Split problem→solution**
(2a baseline-only, 3a +REPA) per the REPA rhetorical device — strong for us
because the baseline is flat.

## Tables

| # | What it shows | REPA-analogue | Status | Index | Claim |
|---|---|---|---|---|---|
| 6.1 | Proteina-60M + REPA config (layers, dim, heads, λ, projector depth, encoders) | Table 1 | `EXISTS` | §6.1 | setup reference |
| **6.2 CENTERPIECE** | All metrics × encoder variants {baseline, GN-L4, GN-L9, MPNN-L4, MPNN-L9, rand-L4} at **fixed step (700K)**, **γ=0.45**, super-column grouped; one block per dataset (PDB, AFDB) | Table 2 | `ANALYSIS✓` (700K snapshot computed — Part 5) | §6.4/§6.6 | the whole story at a glance |
| 6.3 speedup | Steps for REPA to reach baseline's *best* FID → **N×**; AFDB-GearNet only (PDB has no clean speedup — see Part 5) | Table 3 | `ANALYSIS✓` (computed — Part 5) | §6.4 | acceleration number (AFDB-GearNet 6.5–13×, durable) |
| 6.4 sampler | Metrics × γ {ODE,0,.35,.45,.5,1}, GearNet, PDB+AFDB | (none) | `EXISTS✎` (figures → tabularise) | §6.7 | "gains robust in the γ∈[0.35,0.5] band; break at extremes" — **GearNet only** |
| 6.5 rep table (opt) | CATH-C/A/T, IF, dihedral × encoder, Δ vs baseline | (in Fig 5a region) | `PLOT` | §6.2 | encoder rank-order on rep quality, numerically |

We deliberately **skip** REPA's Fig 5b/5c (model-size) — single 60M size. A true
Fig 5a (encoder-quality→FID scatter) is **too sparse** (~3 encoder families with
trained students); carry that claim via Table 6.2 rank-order + §6.5 instead.

---

# Part 2 — Section-by-section flow (topic sentences + claims)

Order leads with **representation** (cleanest evidence), then convergence, then
the bridge, then the anatomy/trade-off, then robustness. The "REPA is a *family*
of interventions, not one" theme threads §6.5–6.7 and crystallises in §6.6.

## §6.1 Setup

1. "We integrate REPA into Proteina, a 60M-parameter non-equivariant
   flow-matching Cα-backbone generator, attaching a trainable 3-layer projector
   at a chosen trunk layer that aligns to a frozen structural encoder." —
   *config.* `Table 6.1`. ✓✓
2. "We study the two encoders profiling (Ch~\ref{ch:profiling}) flagged as
   viable targets — CA-GearNet (global contact-graph topology) and ProteinMPNN
   (local inverse-folding environment) — at two injection depths (L4, L9), with
   a random-weights GearNet as a falsifier control." — *what we trained.* ✓✓
3. "We train in two data regimes that bracket the field's data-scarcity problem:
   PDB (experimental, scarce) and AFDB (AF2-predicted, abundant, and more
   in-silico-designable [CITE])." — *regime motivation; ties to intro spine —
   REPA is a data-efficiency tool, tested in the real-but-scarce and the
   synthetic-but-abundant regimes.* ✓✓ [NEEDS-CITATION]
4. "Unless noted we evaluate with the Ch~\ref{ch:evaluation} suite at the
   γ=0.45 SDE sampler, deferring sampler-robustness to §6.7." — *protocol.* ✓✓

## §6.2 REPA improves the student's representation quality  [PROBLEM → SOLUTION]

*(Problem — Fig 6.2, baseline only)*
1. "Left to the flow-matching loss alone, Proteina's hidden states never become
   more structurally decodable: across 1.8M steps the baseline's CATH-fold and
   inverse-folding probe accuracy stays flat (CATH-T ≈0.30, IF ≈0.12)." —
   *the problem.* Coverage: n256 PDB. ✓✓ `Fig 6.2`

*(Solution — Fig 6.3, +REPA)*
2. "Under REPA the student's representations become markedly more structural,
   and as a *persistent gap* rather than a head-start — the baseline never
   catches up (CATH-T 0.30→0.75 for L9-GN)." — *core rep result; note the
   contrast with the gen side, which is acceleration not gap.* n256 PDB. ✓✓ `Fig 6.3`
3. "The gain rank-orders by encoder, and the order is axis-specific: GearNet
   dominates fold-level decodability (CATH-T +0.34) while MPNN leads per-residue
   geometry (dihedral 16.4°, IF top-1)." — *the GearNet→global / MPNN→local
   mirror; previews §6.6.* n256 PDB. ✓✓ `Table 6.5`
4. "A random-weights encoder gives the smallest gain on every probe — the effect
   needs structural knowledge in the target, not merely an auxiliary loss —
   though at n256 the random gain is small-positive, not zero." — *falsifier
   #1; honest refinement.* n256 PDB. ✓ (n128 compresses this — see §6.7)

## §6.3 REPA improves representation alignment (CKNNA)

1. "Beyond decodability, REPA pulls the trunk's geometry toward the encoder's:
   per-residue CKNNA, at the noise floor (≤0.003) for the baseline, rises 20–25×
   under REPA-GearNet-L9." — *alignment is REPA-induced; cleaner causal story
   than the REPA paper's own (their baseline already drifted up).* n256 PDB,
   step 1M. ✓✓ `Fig 6.4`
2. "Alignment peaks in the upper-middle stack (≈L8) and collapses at the final
   velocity-output layer — where we align is not where the representational peak
   ends up, matching REPA's image-domain observation." — *per-layer profile.* ✓
3. "Aligning to GearNet alone also raises alignment to MPNN and ESM2 — REPA
   yields generically more encoder-like reps (Platonic convergence), with ESM2
   the most universal attractor (L9-GN→ESM2 0.046 > →GearNet 0.024)." —
   *headline CKNNA finding; defuses the 'CKNNA-to-own-target is tautological'
   objection.* n256 PDB, 1M. ✓ `Fig 6.4` (off-diagonal panel)
4. "These gains are residue-level: the per-protein mean-pooled matrix is flat —
   a guardrail against over-claiming fold-level representational benefit." —
   *caveat; consistent with a per-residue loss.* ✓✓

## §6.4 REPA accelerates convergence on generation metrics

1. "The acceleration is sharp and durable on synthetic data: REPA-GearNet
   reaches the AFDB baseline's *best-ever* FID-AFDB **6.5–13× earlier** (L4-GN
   @100K, L9-GN @200K vs baseline @1300K) and keeps improving past it (FID 282
   vs the baseline's floor of 386)." — *headline acceleration; AFDB-GearNet is
   the clean number.* `Fig 6.1`, `Table 6.3`. ✓✓ [computed 2026-05-28 — Part 5]
   ⚠ **On PDB there is NO clean 'reaches-baseline-final-FID-faster' story** —
   the PDB baseline keeps improving FID-PDB to 288 (1.6M) while every REPA
   variant plateaus higher (L9-GN best 319; MPNN-L4 at matched 1.6M only 329).
   Frame PDB as *step-matched* acceleration (§6.4 claim 2), not a speedup factor.
2. "On PDB the gain is *step-matched* acceleration: at 700K, REPA-L9-GN reaches
   FID-PDB 319 vs the baseline's 437 (−27%) — a level the baseline does not
   match until ~1.5M (≈2× step-efficiency) — and most learned variants beat the
   step-matched baseline on FID/fJSD-A through ~1M." — *PDB = acceleration, not
   asymptotic dominance.* `Table 6.2`. ✓✓ [computed — Part 5]
3. "The regime split is the headline the two-dataset design buys us: on AFDB the
   advantage is *durable* (baseline FID-AFDB floors at 386 and never closes the
   gap; REPA-GearNet sits at 252–313 from 200K on), whereas on PDB the
   long-trained baseline matches and edges below REPA late." — *durable on
   synthetic, transient on real — same intervention, opposite asymptotics.* ✓✓
   `Fig 6.1`

## §6.5 Does representation quality predict generation quality?  [the bridge]

1. "Representation and generation quality co-move: traced over training, REPA's
   rising probe accuracy tracks its falling FID while the baseline stays flat on
   both." — *the co-movement (our 3c).* `Fig 6.5`. ✓
2. "Across checkpoints the relationship is strong: better student reps predict
   lower FID and higher designability (xclean-PDB Pearson: IF→designability
   r=0.69, CATH-A→FID r=−0.49; xclean-AFDB CATH-A→FID r=−0.55)." — *quantified.*
   `Fig 6.5`. ✓ [computed — Part 5]
   ⚠ **Step-controlled caveat**: partialling out training step, the robust
   survivors are IF/local-rep→designability (PDB 0.53, AFDB 0.47) and
   CATH/fold-rep→FID on AFDB (−0.60); on PDB the CATH→FID link largely washes
   out (much of the raw co-movement is the common "both improve with training"
   trend). This *backs* the "no formal mediation" scope in claim 4.
3. "Its encoder-matched structure is the strongest evidence the gen gain runs
   through representation: GearNet's fold-rep advantage surfaces as a
   fold-*distribution* gen advantage; MPNN's local-rep advantage as a
   *designability* advantage." — *axis-matched prediction-confirmation;
   validates the Ch4 framework.* ✓
4. "We stop short of a formal mediation claim: the random-encoder control
   brackets the relationship (small rep gain → small gen gain), but co-movement
   is not causation." — *honest scope.* ⚠

## §6.6 The anatomy of the generation gain  [super-columns + the trade-off — BIG]

1. "Decomposed across the metric suite, REPA's effect is not uniform: strongest
   and most encoder-robust on whole-set distribution, most nuanced on
   designable-subset diversity." — *framing.* `Table 6.2`
2. (T-W/S-W) "Whole-set diversity improves under nearly every variant — fold
   entropy (fS-A, fS-C) almost always rises, and SS-distribution match
   (ssJSD-2D) is the single most robust REPA effect, holding even for the random
   control at n128." — *the generic core.* ✓✓ broad coverage. `Table 6.2`
3. (quality) "Per-sample quality improves encoder-specifically: MPNN accelerates
   designability and per-residue quality across both datasets, GearNet less
   consistently." — *mirrors §6.2 rank-order.* ✓ `Table 6.2`
4. (S-D) "On PDB, learned-encoder REPA shifts the *designable* subset's
   secondary-structure composition β-ward — but this is dataset-specific:
   AFDB-MPNN shifts the opposite way (α-ward)." — *precise wording — NOT 'REPA
   makes more sheets' generically.* ✓✓ (PDB vs AFDB explicit)
5. (T-D — the headline trade-off) "The cost sits in tertiary diversity *within
   the designable subset*: REPA concentrates designable samples onto a few
   encoder-preferred modes, so whole-set fold coverage rises (fS-A↑) while
   distinct designable folds fall (#clusters/pwTM↓)." — *the big finding.* ✓✓
   `Fig 6.6`
6. "The trade-off is gated by which SS-mode the encoder×dataset combination
   favours: β-rich modes are fold-narrow, so configs that concentrate β-rich
   (GearNet on both datasets, MPNN on PDB) lose designable diversity, while
   MPNN-on-AFDB concentrates α-rich/mixed and preserves it — the clean
   falsifier." — *mechanism + falsifier #2; 'REPA is a family of
   interventions'.* ✓✓ `Fig 6.6`
7. "The concentration is confined to the designable subspace and grows with
   training (designable-vs-whole pwTM gap 0→+0.25 over 400–1000K), which is why
   step-matched comparisons read it as a late diversity 'cliff' while whole-set
   diversity stays comparable." — *resolves the step-matched-vs-absolute
   tension; cite Exp A.* ✓✓

## §6.7 Robustness: datasets, encoders, samplers, scale

1. "The convergence and distribution-match gains hold across both data regimes,
   differing in magnitude/durability (durable on AFDB, catch-up on PDB)." — ✓✓
2. "They hold across the sampler noise band γ∈[0.35,0.5], breaking only at the
   extremes — distribution-match collapses at full-temperature γ=1,
   designability at γ=0 — shown for **GearNet only** (MPNN sampler ablation
   pending)." — *robustness + explicit coverage hole.* ✓ (GearNet) / ⚠ (MPNN)
   `Table 6.4`
3. "Across encoders the whole-set gains are generic but the directional effects
   (β-shift, diversity trade-off) are encoder×dataset conditional — REPA is a
   family of interventions, not one." — *the thread crystallises.* ✓✓
4. "At n128 the learned-vs-random separation compresses (random helps nearly as
   much), so we anchor headline claims at n256 and read n128 as a
   scale-robustness check." — *n128's role.* ✓✓
5. (opt) "Training-dynamics ablations (λ, batch size) show the gain is stable
   across reasonable settings." — *minor; if time.* `OPT`

## §6.8 (optional) What Proteina shows — bridge to Ch7
Two-to-three sentences distilling: REPA improves rep quality (persistent) and
accelerates generation (durable on synthetic data); the gain is real but is a
*family* of encoder×dataset-conditional interventions, with a designable-fold
diversity trade-off. Hand to Ch~\ref{ch:conclusions}. Or fold into §6.6/§6.7.

---

# Part 3 — Data / analysis checklist

**Ready (read/plot from disk):**
- [ ] Fig 6.1 headline FID convergence, 2-panel PDB+AFDB (early plots exist; finalise)
- [ ] Fig 6.2/6.3 rep-over-training, split baseline-only / +REPA (CSV: `representation/results/paper/n256_convergence_*`)
- [ ] Fig 6.5 gen-vs-rep envelope+correlation (exist: `joint/figures/paper/n256_convergence/`)
- [ ] §6.5 Spearman/Pearson ρ (read `joint/results/paper/n256_convergence_gen_vs_rep_correlation.{md,csv}`)
- [ ] Table 6.5 rep rank-order numbers (from rep sweep)

**Needs analysis (data on disk, compute needed):**
- [ ] Table 6.3 speedup factors — "steps to baseline asymptotic FID" from convergence sweeps (n256 PDB ~159 ckpts, AFDB ~86)
- [ ] Table 6.2 centerpiece — fixed-step (700K) all-metric × encoder, PDB + AFDB
- [ ] Fig 6.6 trade-off — pick metric form (designable #clusters vs designable-pwTM); + designable-vs-whole pwTM gap panel
- [ ] Fig 6.4 — decide heatmap vs cleaner per-layer line plot; add off-diagonal panel

**Needs data (jobs):**
- [ ] MPNN sampler ablation — PDB submitted (jobs 29735627-31), AFDB queued (`n256_afdb_sampler_ablation_ext`). Gates §6.7 cross-encoder sampler claim. Re-run `build_sampler_regime_robustness.py` + `clean_variance_jsonl.py` when they land.
- [ ] (OPT) Fig 6.8 rep-quality-vs-t — probe student at multiple t
- [ ] (OPT) Fig 6.7 sample gallery — generate backbones from ckpts
- [ ] (OPT) length-extrapolation — n128-trained model generating >128-residue proteins (n128's best report use; investigate if time)

**Research debt:**
- [ ] Citation: AFDB more in-silico-designable than PDB (Proteina paper? AF2 confidence work?)

---

# Part 4 — Notes & decisions

## Decisions locked (2026-05-28)
- Headline figure = **FID convergence** (Fig 6.1), caption motivates the two
  regimes.
- **Thread the imaging-vs-molecular argument** Ch3 seed → Ch6/Ch7 pay-off.
- **Elevate the T-D trade-off** to a full subsection (§6.6); give both
  falsifiers (random-GN, MPNN-AFDB) explicit space.
- Centerpiece Table 6.2 = γ=0.45 fixed-step; sampler ablation = separate Table 6.4.
- Headline figures index on {FID, designability}; tables carry full suite.
- Lead with representation; reorder rep → convergence → bridge → anatomy → robustness.

## Imaging vs molecular ↔ spine connection  (remember this)
In images the pixel grid is *both* content and geometry (RGB, one narrow,
contiguous feature space), so a *single* SSL encoder (DINOv2) is a near-complete
alignment target. In molecules the two are **decoupled** — a continuous 3D
coordinate field *and* a discrete, periodic-table-sized type space, organised in
a hierarchy (primary→secondary→tertiary). So (a) representation is
multi-factorial → no single encoder captures all factors → **which encoder you
pick routes REPA to a different factor** (GearNet→tertiary/topology,
MPNN→local), which *is* our encoder×dataset-conditionality finding; and (b)
generation-quality is multi-factorial → the super-column suite, harder
mode-collapse reasoning. This is the *a-priori* reason to expect "REPA is a
family of interventions" and is the conceptual root that the spine's
"transferability is about representation statistics, not domain" rests on in a
multi-factor domain. Safe framing: index on the **content–geometry coupling**
difference + **discrete/continuous duality**; do NOT claim "images are
single-factor." (Also saved as memory `project_imaging_vs_molecular_repa`.)

## Excluded material (do not put in the chapter)
- **H1 / loss-saturation / cos_sim-saturation** story — refuted by denoised
  data; drop entirely. The 700K "cliff" is real in #clusters/SS-composition (H2)
  but has no loss-balance explanation.
- "REPA → bimodal scRMSD" — refuted; the real effect is 2–4Å marginal-zone
  *polarization* (use that if scRMSD comes up, not bimodality).

## Coverage holes that force couching
- Sampler robustness: **GearNet-only** today (MPNN pending).
- n128 **GearNet** rep probes missing (only MPNN/random on disk) → no n128
  encoder rank-order yet.
- AFDB-trained rep eval too small (n≈62 clean) → rep-quality is a *PDB-trained*
  finding; don't lean on AFDB rep numbers.
- CKNNA = single step (1M), single dataset (PDB), single t — a snapshot.

---

# Part 5 — Computed numbers (2026-05-28)

Source: `generation/results/paper/n256_convergence_{pdb,afdb}/sweep_results.clean.jsonl`
(γ=0.45 `sde_n0.45`, 3 seeds/cell, seed-mean). In-distribution FID = FID-PDB for
PDB models, FID-AFDB for AFDB models. Speedup = baseline's first step reaching
its own cumulative-best FID ÷ the variant's first step reaching that value.

## Speedup (Table 6.3) — AFDB-GearNet is the clean story

| dataset | metric | variant | baseline best | REPA reaches @ | **speedup** | REPA's own best | durable? |
|---|---|---|---|---|---|---|---|
| AFDB | FID-AFDB | **L4-GN** | 386 @1300K | **100K** | **13.0×** | 282 | ✓ (baseline floors at 386) |
| AFDB | FID-AFDB | **L9-GN** | 386 @1300K | **200K** | **6.5×** | 313 | ✓ |
| AFDB | FID-AFDB | MPNN-L9 | 386 @1300K | 400K | 3.2× | 352 | ~ |
| AFDB | FID-AFDB | MPNN-L4 | 386 @1300K | never | — | 416 | ✗ (MPNN-AFDB ≠ distribution shaper) |
| AFDB | fJSD-A | L4-GN / L9-GN | (early) | 200K | ~2× | — | ✓ mid-traj |
| PDB | FID-PDB | *all variants* | **288 @1600K** | **never** | **—** | L9-GN 319; MPNN-L4 329 | ✗ baseline overtakes |

**PDB honest framing**: baseline FID-PDB keeps dropping (700K 437 → 1.5M 318 →
1.6M 288); REPA plateaus higher. So *no* "reaches-baseline-final faster". The
real PDB claim is **step-matched**: L9-GN @700K = 319 vs baseline 437 (−27%),
and the baseline only reaches ~319 by ~1.5M (≈**2×** step-efficiency *to REPA's
plateau level*), but its longer training then edges below. MPNN-L4 trained the
full 1.6M and still only hit 329 > 288 → on FID-PDB the baseline is
asymptotically as good or better. **Do not quote a PDB speedup factor.**

## 700K snapshot (Table 6.2 sanity — the encoder split is crystal-clear)

**PDB @700K** (FID-PDB; rand = random-GearNet control):

| family | Des% | scRMSD | pLDDT | FID-PDB | fJSD-A | fS-A | #Clust | pwTM | ssJSD2D | E%(β) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.46 | 2.96 | 0.61 | 437 | 1.11 | 5.48 | 77 | 0.27 | 0.35 | 0.21 |
| L4-GN | 0.70 | 3.36 | 0.66 | 508 | 0.73 | 7.11 | 78 | 0.38 | 0.14 | 0.23 |
| **L9-GN** | 0.60 | 3.07 | 0.66 | **319** | **0.53** | **7.79** | 104 | 0.25 | 0.16 | 0.17 |
| MPNN-L4 | 0.64 | **2.77** | 0.66 | 349 | 0.98 | 5.96 | 73 | 0.35 | 0.19 | 0.20 |
| MPNN-L9 | 0.51 | 2.71 | 0.63 | 418 | 0.79 | 5.85 | 110 | 0.26 | 0.21 | 0.18 |
| rand (ctrl) | **0.38** | 4.07 | 0.58 | 471 | 0.55 | 5.53 | 54 | 0.17 | 0.22 | 0.15 |

**AFDB @700K** (cross-DB FID-PDB shown; in-dist FID-AFDB in speedup table):

| family | Des% | scRMSD | pLDDT | FID-PDB | fJSD-A | fS-A | #Clust | pwTM | ssJSD2D |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.72 | 2.11 | 0.69 | 463 | 2.07 | 4.59 | 128 | 0.22 | 0.23 |
| **L4-GN** | 0.77 | 2.34 | 0.74 | **253** | **0.61** | 6.86 | 92 | 0.26 | 0.09 |
| L9-GN | 0.73 | 2.67 | 0.72 | 272 | 0.69 | **7.35** | 77 | 0.33 | **0.06** |
| MPNN-L4 | 0.69 | 2.20 | 0.69 | 452 | 1.92 | 4.62 | 127 | 0.15 | 0.21 |
| **MPNN-L9** | 0.77 | **1.93** | 0.72 | 326 | 1.45 | 4.90 | **149** | 0.20 | 0.24 |

Reads straight off the table for §6.6: **GearNet → distribution/SS-match**
(FID, fJSD-A, fS-A, ssJSD2D all best; reduces #Clust on AFDB 128→77/92);
**MPNN → per-sample quality** (scRMSD/pLDDT/Des best; preserves #Clust on AFDB
149 ≥ baseline 128 — the falsifier); **random** helps fJSD-A/ssJSD2D but *hurts*
Des% (0.38<0.46) and #Clust (54<77) — distribution-regularizer without the
quality/diversity benefit.

## Correlation summary (§6.5) — from `joint/.../gen_vs_rep_correlation.md`

Headline (xclean, all checkpoints, Pearson): IF→designability **0.69** (PDB) /
0.58 (AFDB); CATH-A→FID **−0.49** (PDB) / −0.55 (AFDB); dihedral→designability
−0.54 (PDB). Step-controlled partial-Spearman survivors: IF→designability (PDB
0.53, AFDB 0.47), CATH-A→FID (AFDB −0.60); on PDB the CATH→FID link washes out
(cath_T partial −0.13 n.s.) → much of the raw PDB co-movement is the shared
training-time trend. Backs the "no formal mediation" scope.

## Coverage note
n256, γ=0.45 only. n128 analogues exist (`n128_convergence_*`) for the
scale-robustness check. AFDB-MPNN cells are 1 seed for L4 (L9 is 3-seed).
