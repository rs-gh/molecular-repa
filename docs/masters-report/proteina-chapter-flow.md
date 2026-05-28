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
- **Variant-selection rule**: trajectory/convergence plots show **best learned
  + random control + baseline** (3 lines). Rank-order plots (where the
  rank-order *is* the claim, e.g. §6.2/§6.3) show **all 4 learned + random +
  baseline**. Tables carry the full suite.
- **Multi-seed bands**: where ≥2 seeds exist for a cell, show shaded ±1 SD (or
  min/max) bands on trajectory plots; tables show point estimate ± SD;
  single-seed cells get an asterisk. Most n256 cells are 3-seed; known
  single-seed exceptions to flag: AFDB-L9-GN early steps (100–600K),
  MPNN-L4-AFDB at most steps.

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
| 6.2 (problem) | **3-panel**: IF top-1 · dihedral MAE · CATH-A — student rep quality vs **training step**, **baseline only**, + NVIDIA-60M ceiling on IF & dihedral only (CATH-A ceiling misleading due to cleantrain model-side leakage) | Fig 2a | `PLOT` | §6.2 | "Flow-matching alone never makes Proteina's hidden states more structural — baseline is flat across 1.8M steps on every probe." |
| 6.3 (solution) | Same 3-panel axes, **+ all 4 learned REPA variants + random** rising; encoder rank-order visible per panel | Fig 3a | `PLOT` | §6.2 | "REPA makes the student's reps structural, as a persistent gap; encoder rank-order is axis-specific (GearNet → CATH-A; MPNN → IF and dihedral)." |
| 6.4 CKNNA | Per-layer CKNNA, **best-GN + best-MPNN + baseline** (not all 4 variants); baseline at noise floor, REPA at L8 peak with L9 collapse; + off-diagonal panel (align→GearNet also lifts ESM2) | Fig 2b/3b | `EXISTS✎` (heatmaps; want line version) | §6.3 | "Alignment is REPA-induced, peaks mid-stack, generalises across encoders (Platonic); per-residue not per-protein." |
| 6.5 gen-vs-rep | **Lead**: xclean-PDB **dihedral MAE → designability** envelope/correlation scatter (raw r=−0.54, partial-step r=−0.47 — strongest partial outside IF, with interpretable absolute spread 32°→16°); supplementary panels for CATH-A→Des and IF→Des | Fig 3c | `EXISTS` | §6.5 | "Better student dihedral geometry predicts more designable generations; the link survives controlling for training step." |
| 6.6 trade-off | **3-panel**: (a) T-W up — fS-A vs step (more whole-set fold classes under REPA); (b) T-D down — designable #clusters (or pwTM) vs step (fewer distinct designable folds); (c) mechanism — β-stratified pwTM within designable subset (REPA's β≥25 bin concentrated 0.7–0.9 vs baseline 0.13) | (none — our finding) | `ANALYSIS` (data on disk) | §6.6 | "REPA covers more whole-set fold classes but concentrates the designable subset onto a few encoder-preferred β-rich modes; β-rich is fold-narrow → that's the mechanism. MPNN-AFDB is the falsifier (no concentration)." |
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
| **6.5 rep table** | CATH-C/A/T, IF, dihedral × encoder, Δ vs baseline (point ± SD across seeds where available) | (in Fig 5a region) | `PLOT` (promoted from OPT — companion to Fig 6.2/6.3) | §6.2 | encoder rank-order on rep quality, numerically, with the axis split visible |

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
   the durable number.* `Fig 6.1`, `Table 6.3`. ✓✓ [computed 2026-05-28 — Part 5]
2. "On PDB the acceleration is *transient but real*: REPA-L9 (both GN and MPNN)
   leads on FID-PDB at every step ≤1.2M (−27% at 700K, ≈1.3–1.5× step-
   efficiency to REPA's plateau ~330), and the baseline only matches REPA's
   plateau by ~1.4M — but then continues to improve past it (288 at 1.6M)." —
   *PDB = transient acceleration, not asymptotic dominance.* `Table 6.2`,
   `Table 6.3`. ✓ [computed — Part 5; L9-GN/MPNN trajectories stop at 1.2M,
   pending an extension run for 1.4–1.6M]
3. "The regime split tracks the baseline's own convergence: AFDB's baseline
   *plateaus poorly* (FID-AFDB floors at 386, oscillates 386–534), so REPA's
   advantage is durable; PDB's baseline is a *strong asymptotic learner*
   (reaches FID 288 by 1.6M), so REPA front-loads but the baseline closes.
   Same intervention; regime-dependent asymptotics — REPA helps most where
   the base learner struggles most." — *the two-regime story tied to baseline
   convergence behaviour, not REPA itself. Thread T3.* ✓✓ `Fig 6.1`
3b. ⚠ **Honesty caveat on AFDB durability** (added 2026-05-28 from sampler-
   audit). AFDB-trained models score 2–5× higher on designability than PDB-
   trained ones at *every* γ (baseline Des: AFDB 0.05–0.79 vs PDB 0.00–0.31
   across γ; gap persists with REPA on top). Because designability is measured
   via ProteinMPNN → ESMFold (the same folding-model lineage that produced
   AFDB structures via AF2), AFDB-trained generators are biased toward
   producing structures the proxy is confident about — a *proxy–data
   alignment* artefact. This refines but does not invalidate the AFDB
   durability claim: REPA's AFDB gains on **FID, fJSD, ssJSD-2D, rep-quality
   probes, and CKNNA** are all robust to this caveat (none of those metrics
   share the ProteinMPNN/ESMFold lineage), but the *designability* component
   of durability is partly proxy-friendly by construction. — *the honesty
   layer on Thread T3; seeded in Ch~\ref{ch:evaluation} §3.4 NotA, paid off
   here.* ✓ from sampler-audit baseline tables.

## §6.5 Does representation quality predict generation quality?  [the bridge]

1. "Representation and generation quality co-move: traced over training, REPA's
   rising probe accuracy tracks its falling FID while the baseline stays flat on
   both." — *the co-movement (our 3c).* `Fig 6.5`. ✓
2. "Lead: across checkpoints, lower student dihedral error predicts higher
   designability (xclean-PDB Pearson r=−0.54, step-controlled partial r=−0.47)
   — the per-residue rep axis cleanly predicts per-sample gen quality, on
   absolute scales that read meaningfully (baseline 32° → MPNN-L9 16°)." —
   *the §6.5 headline pair.* `Fig 6.5`. ✓ [computed — Part 5]
3. "Supplementary: the same monotone relationship holds across other
   rep×gen axis pairs — IF→designability has the largest raw correlation
   (r=0.69, partial 0.53; small absolute spread); CATH-A→FID is the
   per-chain analogue (raw −0.49 on PDB, −0.55 on AFDB)." — *carry as
   complementary panels or a small table; the axis split mirrors §6.2/§6.3.*
   ✓ [computed — Part 5]
4. ⚠ **Step-controlled caveat**: partialling out training step, the robust
   survivors are dihedral/local→designability and CATH/fold→FID on AFDB; on
   PDB the CATH→FID link washes out (much of the raw co-movement is the shared
   "both improve over training" trend). This *backs* the "no formal mediation"
   scope in claim 6.
5. "The encoder-matched structure of the co-movement is the strongest evidence
   the gen gain runs through representation: GearNet's fold-rep advantage
   surfaces as a fold-*distribution* gen advantage; MPNN's local-rep
   advantage as a *designability* advantage. Mirrors the §6.2/§6.3 rank-order
   split exactly." — *axis-matched prediction-confirmation; validates the
   Ch~\ref{ch:profiling} framework.* ✓
6. "We stop short of a formal mediation claim: the random-encoder control
   brackets the relationship (small rep gain → small gen gain), but
   co-movement is not causation." — *honest scope.* ⚠

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
   makes more sheets' generically. Thread T2 (encoder-routing).* ✓✓ (PDB vs
   AFDB explicit)
4b. (γ-invariance of the β-direction — added 2026-05-28) "The β-shift
   direction (encoder × dataset) is **sampler-invariant**: γ-spread within a
   given (encoder, dataset) is ≤0.07 on Edes%, while the encoder-direction
   split holds at every γ ∈ {ODE, 0, 0.35, 0.45, 0.5, 1} — AFDB L9-GN
   Edes% = 0.15–0.21 across all 6 γ values vs AFDB-MPNN-L9 Edes% = 0.11–0.16,
   bracketing the AFDB baseline (0.12–0.16) in opposite directions at every
   noise level. The encoder routes REPA to a representational-alignment
   *direction* that the sampler dial cannot substitute for or reach." —
   *Direct empirical evidence for the imaging-vs-molecular thread: encoder
   choice and sampler choice are orthogonal axes in this domain. Pays off
   Ch~\ref{ch:conclusions} (the multi-factorial central claim). Thread T2.*
   ✓✓ from sampler audit.
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
8. (Mechanism disentangling — added 2026-05-28 from sampler audit) "The
   trade-off has two components that disentangle cleanly: the *clustering*
   component is driven by the encoder's **geometric inductive bias** — random-
   weight GearNet *also* reduces designable #Clust on PDB, so this is not a
   learned-feature effect. The candidate mechanism is GearNet's relational
   graph-convolution over residue-contact graphs: the input graph carries
   fold-topology information even with random weights, and graph-conv
   smoothing then concentrates similar geometries. The *direction* of the
   SS-shift (β-rich vs α-rich) requires learned features — random-GN doesn't
   shift β, so the direction component is genuinely learned-representation-
   driven. Caveat: no random-MPNN run to fully isolate 'graph-conv inductive
   bias in general' from GearNet-specific design choices." — *Thread T4
   refined; T1 (learned features) + T2 (encoder routing) disentangled at the
   mechanism level. MPNN-AFDB falsifier consistent with both halves: MPNN's
   geometry doesn't cluster the manifold the same way (no #Clust loss) AND
   its learned features point at a different SS-direction (α-ward).* ✓ from
   audit Claim E.

## §6.7 Robustness: datasets, encoders, samplers, scale

Structure as five subsections.

### §6.7.1 Across datasets
1. "The two-regime contrast established in §6.4 holds across the metric suite,
   not just FID: AFDB-trained REPA durably beats baseline on FID/fJSD/ssJSD2D
   throughout training; PDB-trained REPA leads step-matched through ~1.2M and
   converges with the baseline thereafter." — *durable vs transient.* ✓✓

### §6.7.2 Across sampler noise levels  (dedicated subsection — fully rewritten 2026-05-28 after the ext sweep landed; supersedes prior γ=1 failure-mode framing)

1. (Take 1 — scaffold) "REPA shifts the *levels* of the Des↔T-D tradeoff but
   doesn't change the γ-shape: REPA and baseline navigate γ identically
   (γ=0 mode-collapsed high Des + huge pwTM; γ=mid moderate Des + peak
   #Clust; γ=ODE/γ=1 low Des + designable pool too small to cluster). So
   γ=0.45 results compose with the sampler dial rather than depending on it
   — REPA is a training-time intervention orthogonal to inference-time
   sampling choices." — *enables the rest of the chapter's γ=0.45-anchored
   claims to read robustly. Thread T5 (sampler scaffold).* ✓✓

2. (Take 2 — MAIN finding) "REPA's fJSD-A distribution-match advantage is
   sampler-invariant for **learned encoders** but near-chance for the
   **random** control across all 6 γ values (L4-random on PDB fJSD-A wins:
   2/4, 3/4, 2/4, 2/4, 2/4, 3/4 ≈ 52%; L4-GN ≈ 75%; L9-MPNN ≈ 70%). This
   validates that REPA does mechanistic work via the encoder's learned
   structural knowledge — not auxiliary-loss regularisation — and that the
   encoder-selectivity finding from §6.6 is noise-robust." — *Thread T1
   (encoder-selectivity) reinforced at γ-invariance. Pairs with the §6.6
   ablation table, where random helps T-W/S-W but not fJSD-A/quality —
   that selectivity now also holds at every γ.* ✓✓ `Fig: per-encoder
   Δ fJSD-A vs γ, learned vs random`

3. (Take 3 — ODE Des-floor, refined to AFDB) "AFDB shows a clean ODE
   designability-*floor* effect: every REPA encoder boosts Des at
   deterministic sampling (Δ +0.06 to +0.16 absolute). On PDB only L9-GN
   robustly wins at ODE and the absolute Δ is tiny because the PDB
   baseline's ODE Des is near-zero at early steps (≤0.04 at 100–400K). This
   dataset asymmetry tracks the broader AFDB-designability bias (see
   §6.4 claim 3b; Ch~\ref{ch:evaluation} §3.4 proxy NotA)." — *AFDB-specific
   robustness claim; the PDB-equivocation is itself evidence for the
   proxy-data alignment story. Thread T3 paid off here.* ✓ (AFDB) ⚠ (PDB
   equivocal)

4. (γ-failure mode, simplified) "Only one γ-regime failure survives the
   larger sweep: γ=0 designability loss for **GearNet**-aligned REPA on
   **both** datasets (L9-GN PDB 2/5, AFDB 0/3; random-GN PDB also loses).
   MPNN preserves at γ=0. Mechanism candidate from the audit: GearNet's
   geometric alignment broadens the very narrow γ=0 mode-collapse basin
   the baseline lives in. — *encoder-specific, not dataset-specific.
   Thread T1 at the negative edge: encoder choice routes a *limitation*
   too, not only the gains.* ✓

5. (Retraction — supersedes earlier framing) "The earlier 'γ=1 distribution-
   match collapse on PDB' claim is RETRACTED as a single-cell PDB-L9-GN-700K
   artifact. With the full multi-encoder grid, every PDB REPA encoder
   except the originally-quoted cell wins or ties on fJSD-A at γ=1 (L4-GN
   4/5, MPNN-L4 4/6, MPNN-L9 4/5). One bad row got promoted to a regime
   claim; simplifies the chapter — only the §6.7.2 §4 GearNet-γ=0 failure
   remains."

6. **Coverage / reproducibility**: PDB step-matched at γ=0.35/0.5 required
   recovering baseline rows from the older clean.jsonl (the 2026-05-28
   refresh dropped them; raw doesn't have them). Audit doc records this.
   Backfill (14 baseline cells at γ ∈ {0.35, 0.5}) would let
   `build_sampler_regime_robustness.py` auto-emit PDB tables but is not
   load-bearing for the revised claims above. Full claim-by-claim re-audit:
   [proteina_sampler_regime_audit_2026-05-28.md](../research/proteina_sampler_regime_audit_2026-05-28.md).

### §6.7.3 Across encoders
1. "Across encoders, the whole-set gains are generic but the directional
   effects (β-shift, T-D trade-off) are encoder×dataset conditional. REPA is
   a *family* of interventions, not one — the thread that has run through
   §6.2 (axis-specific rep gain), §6.5 (axis-matched co-movement), §6.6
   (encoder×dataset gated trade-off with γ-invariant β-direction, and the
   mechanism disentangling into geometric inductive bias + learned features),
   and §6.7.2 (γ-invariant encoder-selectivity on fJSD-A) crystallises
   here." — *the T2 thread fully formed; payoff sets up Ch~\ref{ch:conclusions}.*
   ✓✓

### §6.7.4 Across scale (n128)
1. "At n128 the learned-vs-random separation compresses (random helps nearly
   as much on rep CATH and on T-W metrics), so we anchor headline claims at
   n256 and read n128 as a scale-robustness check — gains direction is
   consistent, magnitude smaller." — ✓✓
2. (OPT) "Length-extrapolation: an n128-trained model evaluated at >128
   residues would test out-of-distribution generation; data not yet
   collected." — *future-work flag.* `OPT`

### §6.7.5 (OPT) Training dynamics
1. "λ, batch size, projector depth, alignment-layer choice — stability check.
   Models trained briefly; deprioritise unless extended training lands. Not
   the crux of the chapter." — `OPT`

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
- [x] **MPNN+random sampler ablation — DONE (2026-05-28).** PDB jobs 29735627-31 COMPLETED 21/21; AFDB ext also landed. Coverage: PDB now has all 5 REPA variants × 5 γ × 3 steps (baseline lost γ=0.35/0.5 in clean regen — recoverable from git commit 51fddb6); AFDB has 4 families × 5 γ × ~5 steps (full grid). Full re-audit in [proteina_sampler_regime_audit_2026-05-28.md](../research/proteina_sampler_regime_audit_2026-05-28.md). Five chapter-bearing findings (§6.4, §6.6, §6.7.2, §6.7.3) integrated above.
- [ ] **(OPT) Backfill 14 PDB baseline cells at γ=0.35/0.5** — would unlock `build_sampler_regime_robustness.py` auto-emit of PDB tables, but not load-bearing for the revised §6.7.2 claims.
- [ ] **PDB L9 extension (1.3–1.6M)** — gen-eval at later ckpts for `repa_l9_256_per_residue_bs24_2gpu` and `repa_mpnn_l9_256_per_residue`. Checkpoints on disk; need inference + designability sweep (~6h/ckpt). **Cheap variant**: 1 seed × 2 ckpts × 2 configs = 4 jobs ≈ 24 GPU-hours, decides whether to invest more. Sharpens §6.4 PDB framing — if L9 keeps dropping past 1.2M, "transient acceleration" softens further; if it plateaus, current framing stands.
- [ ] (OPT) Fig 6.8 rep-quality-vs-t — probe student at multiple t
- [ ] (OPT) Fig 6.7 sample gallery — generate backbones from ckpts
- [ ] (OPT) length-extrapolation — n128-trained model generating >128-residue proteins (n128's best report use; investigate if time)

**Research debt:**
- [ ] **Citation: AFDB-trained generators are biased toward in-silico-designable structures by proxy-data alignment.** Our own data shows the effect (baseline Des AFDB 2–5× PDB at every γ); we need a literature citation for the *mechanism* (ProteinMPNN/ESMFold share folding-model lineage with AFDB's AF2 source). Likely homes: Proteina paper (designability discussion); AF2 confidence-filtering work; recent backbone-generator papers that compare PDB-trained vs AFDB-trained on designability. Anchor for Ch~\ref{ch:evaluation} §3.4 NotA and §6.4 claim 3b.

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

## Sampler-audit integration (2026-05-28) — thread map
After the sampler-ablation extension landed, we triaged the new patterns by
*does this confirm/contradict a load-bearing thread in the chapter?* not by
*is this an interesting observation?* Net five updates above, each tagged to
its thread:

- **Take 2 → §6.7.2 main finding** (thread T1: encoder-selectivity). REPA's
  fJSD-A advantage is γ-invariant for learned encoders but near-chance for
  random across all 6 γ values. Pairs with the §6.6 ablation table — the
  "random helps some metrics but not fJSD-A/quality" pattern now holds at
  every noise level.
- **Take 4 → §6.6 claim 4b** (thread T2: multi-factorial reps). The
  β-direction (encoder × dataset) is γ-flat — encoder choice and sampler
  choice are *orthogonal* axes in this domain. Direct empirical hook for the
  imaging-vs-molecular argument (Ch~\ref{ch:evaluation} §3.2 → pays off
  Ch~\ref{ch:conclusions}).
- **Mechanism disentangling → §6.6 claim 8** (threads T1 + T4 disentangled).
  Random-weight GearNet also reduces designable #Clust on PDB → the
  *clustering* component of the T-D trade-off is geometric-encoder-bias-
  driven (graph-conv inductive bias over residue contact graphs); the
  *direction* of the SS-shift requires learned features. MPNN-AFDB falsifier
  consistent with both halves. Caveat: no random-MPNN run to fully isolate
  GearNet-specific from graph-conv-general.
- **Take 3 → Ch~\ref{ch:evaluation} §3.4 NotA + §6.4 claim 3b** (thread T3:
  regime asymmetry, honesty layer). AFDB-trained models are 2–5× more
  designable at every γ because the designability proxy (ProteinMPNN →
  ESMFold) shares folding-model lineage with AFDB (AF2). This *refines* but
  does not invalidate the AFDB durability claim — REPA's gains on FID,
  fJSD, ssJSD-2D, rep-quality probes, and CKNNA do not share that lineage,
  so they are clean; only the designability component of durability is
  proxy-friendly by construction. Seeded in Ch3, paid off in §6.4.
- **Take 1 → §6.7.2 scaffold sentence** (thread T5). Sampler composability —
  REPA shifts levels, not γ-shape of Des↔T-D tradeoff. Enables the rest of
  the chapter's γ=0.45-anchored claims to read robustly. Mild but enabling.
- **Retraction A → §6.7.2 claim 5**. The "γ=1 distribution-match collapse on
  PDB" framing is retracted as a single-cell PDB-L9-GN-700K artifact;
  simplifies §6.7.2.

**Memory written**: `project_afdb_designability_proxy_alignment` (saved
2026-05-28). Full audit: [proteina_sampler_regime_audit_2026-05-28.md](../research/proteina_sampler_regime_audit_2026-05-28.md).

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
| PDB | FID-PDB | L9-GN | ~330 plateau | 700K @319 | **~1.3–1.5×** step-eff. | 319 (1.2M) | ✗ baseline matches by ~1.4M, edges below by 1.6M (288) |
| PDB | FID-PDB | L9-MPNN | ~330 plateau | 1.1M @334 | **~1.3×** step-eff. | 334 (1.1M) | ✗ same shape |
| PDB | FID-PDB | MPNN-L4 (full 1.6M) | 329 plateau | 1.6M @329 | budget-matched | 329 | ✗ baseline asymptotically better (288 vs 329) |

**PDB framing**: REPA-L9 (both GN and MPNN) leads on FID-PDB at every matched
step ≤1.2M, often substantially (−27% at 700K). The baseline only matches
REPA's plateau (~320–334) by ~1.4M (interpolating 1.3M=416, 1.5M=318) — a
~1.3–1.5× step-efficiency *to that plateau level*. Beyond that the baseline
continues to improve (288 @1.6M), so on FID-PDB it asymptotically edges below
REPA. **Frame PDB as *transient acceleration*** (not "no speedup"); the regime
contrast vs AFDB tracks the *baseline's own* convergence behaviour — AFDB
baseline plateaus poorly, PDB baseline is a strong asymptotic learner. The
L9 trajectories stop at 1.2M; an extension run (1.4M, 1.6M ckpts) would
sharpen this further.

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
