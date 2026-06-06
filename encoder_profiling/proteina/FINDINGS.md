# Proteina REPA Encoder Comparison

Cross-encoder summary. Per-encoder depth lives in:
- [gearnet/FINDINGS.md](gearnet/FINDINGS.md) — CA-GearNet (CATH-classifier-trained)
- [gearnet_random/](gearnet_random/) — CA-GearNet random-init baseline (3 seeds)
- [esm/FINDINGS.md](esm/FINDINGS.md) — ESM-2 650M
- [mc_gearnet/FINDINGS.md](mc_gearnet/FINDINGS.md) — MC-GearNet-Edge
- [pw_gearnet/FINDINGS.md](pw_gearnet/FINDINGS.md) — ProteinWorkshop GearNet-Edge
- [mpnn/FINDINGS.md](mpnn/FINDINGS.md) — ProteinMPNN CA-only (inverse-folding)

## How to read these findings

Every per-encoder file is organised around three questions. Together they form the actual selection rubric for REPA targets — each on its own is misleading.

- **Q1 — What information does the encoder encode?** Upper bound on what REPA can transfer (residue identity, 3D geometry, structural / sequence context, protein-level identity). If the encoder doesn't carry information X, REPA can never teach the student X.
- **Q2 — How much of that is reachable from cheap inputs?** A 3-layer MLP from `(onehot, position) → encoder` — per-residue, no neighbours — absorbs most of the per-residue cosine alignment. The proteina student's projector reads off a transformer hidden state that is computed from *strictly more*: `(onehot, position) + noisy 3D coords + cross-residue self-attention + diffusion timestep`. So:
  - **Saturation floor** = cosine reachable from identity + index alone.
  - **Gap above it** = cosine the student can only reach by leveraging coords, attention, or timestep. That's REPA's actual operating budget.
  - For 3D-aware encoders (CA-GearNet, PW) the headroom is mostly coord-driven. For sequence-only encoders (ESM2) the headroom is mostly cross-residue attention context (the encoder ignores coords by construction).
- **Q3 — Is the encoder a tractable optimisation target?** Sparsity, rank collapse, dead dims, and norm explosions degrade the gradient signal even when the embedding is informative.

The diagnostic value lives in the **gap between Q1 and Q2 = REPA headroom**, modulated by Q3 conditioning:

```
            pretraining quality
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   RankMe / PR ↑       residue/protein Δ ↑
        │                       │
        ▼                       │
  mean-dir cos ↓                │
        │                       │
        ▼                       ▼
        projector gap = REPA headroom
                    │
                    ▼
         empirical val-loss gain
```

Caveat: a large gap alone is necessary but not sufficient. ESM2's +0.053 gap is the largest in the field, but it's *sequence-only* — the headroom is conformation-invariant and can't guide 3D geometry. Read the gap alongside Q1 to understand *what kind* of information sits in the headroom.

## Setup
- Run: 2026-05-04, slurm job 28852949, A100, 20:54 wall (re-run with RankMe metric).
- Original 2026-04-29 numbers used a mislabelled effective-rank metric (entropy on σ² of centered Z); now switched to RankMe (Garrido et al. 2023, ICML — Roy–Vetterli effective rank on raw σ of uncentered Z) for cross-paper comparability.
- Inputs: 200 PDB train proteins, `--random-seed 0`, shared LMDB.
- Pipeline: [_probes/lib.py](_probes/lib.py).
- Reproduce: `sbatch hpc-scripts/proteina/encoder_profiling/rerun_rankme.sh full`.
- Re-collate: `python encoder_profiling/proteina/collate.py` →
  [comparison.csv](comparison.csv), [figures/](figures/).

## Headline comparison

| Encoder                         | Embed dim | RankMe       | PR    | Dims@95% | AA probe | 3D@1Å | Δ within−between | Mean-dir | Best projector | **Gap**    |
|---------------------------------|----------:|-------------:|------:|---------:|---------:|------:|-----------------:|---------:|---------------:|-----------:|
| **ca-gearnet** (CATH-cls)       |       512 |        256.4 |  37.9 |      119 |    0.137 |  0.27 |            0.222 |    0.425 |          0.434 | **+0.009** |
| ca-gearnet-random (3-seed avg)  |       512 | 99.8 ± 4.5   |   1.5 |     38.7 |    0.127 |  1.00 |            0.035 |    0.952 |          0.954 |   +0.003   |
| **esm2-650M** (last layer)      |      1280 |       1030.5 |  43.3 |      993 |    0.998 |   N/A |            0.098 |    0.671 |          0.723 | **+0.053** |
| **mc-gearnet-edge**             |      3072 |         12.0 |   1.0 |        1 |    0.146 |  0.77 |            0.043 |    0.855 |          0.858 |   +0.002   |
| **pw-gearnet** (torsional)      |      3072 |        207.7 |   5.5 |       36 |    0.928 |  0.86 |            0.102 |    0.710 |          0.723 |   +0.013   |
| **proteinmpnn-ca** (inv-fold)   |       128 |         84.9 |  33.3 |       92 |    0.327 |  0.73 |            0.022 |    0.835 |          0.850 |   +0.014   |

**3D@1Å** = cosine self-similarity after a 1 Å Gaussian coordinate perturbation (lower = more 3D-sensitive; N/A for sequence-only ESM2). Recorded 2026-06-06 from saved per-encoder `results/*/results.json` (`sigma_1.0A`); GearNet 0.27 is the modal value across three runs (0.269/0.271/0.287, σ≈0.055). The main-text Table 4.3 currently shows a categorical 3D column; these numerics can replace it or move to the appendix.

Columns map to the three questions:
- **Q1 evidence**: AA probe acc (Q1.1 residue identity), Δ within−between (Q1.4 protein-level identity); per-encoder files break this down further into 3D sensitivity (Q1.2) and structural/sequence context (Q1.3).
- **Q2 evidence**: Mean-dir, Best projector, **Gap = best − mean-dir** (a structural property of the encoder — see note below).
- **Q3 evidence**: RankMe, participation ratio (PR), dims-for-95%-variance — three complementary spectral diagnostics. RankMe (entropy of normalised σ) is RankMe paper's published metric; PR `(Σλ)²/Σλ²` is sharper at penalising tail concentration; Dims@95% is the projector-bottleneck check. Per-encoder files add sparsity, norms, and dead dims.

### What the "Gap" column is — and isn't

`Gap = Best projector − Mean-dir` measures *how much (onehot + position) lifts cosine over a constant prediction* on this encoder. Worth being explicit:

- **What it is**: an *estimate of what could be learned* — a structural property of the encoder that upper-bounds how much room exists above the cheap-input floor for any input to extract more cosine. A small gap means the encoder's variance is so tight that even rich inputs probably can't push much further; a large gap means there is room in principle.
- **What it isn't**: REPA's actual operating budget. The budget is `training_cos − best_(onehot+pos)_floor`, observable only from wandb training logs, not from this probe.

The two quantities to keep separate:

| Quantity | Meaning | Where it lives |
|----------|---------|----------------|
| `Best − Mean-dir` (this column) | How much *cheap inputs* lift cosine over a constant prediction. Estimate of *what could be learned* if richer inputs (coords / attention / timestep) helped further. | This table — measurable from the encoder alone. |
| `training_cos − best_(onehot+pos)` | How much the *student transformer* lifts cosine over what a cheap-input MLP achieves. The actual REPA contribution. | wandb logs — only visible during training. |

Workflow: the table tells you which encoders *could* offer headroom; the training cosine vs the `best_(onehot+pos)` floor tells you whether the student *actually used* that headroom. A run that plateaus at the floor is a run where REPA contributed nothing the projector wouldn't do alone.

## Verdicts

| Encoder                 | Verdict     | Deciding metric(s) |
|-------------------------|-------------|--------------------|
| **ca-gearnet** (trained)| **usable**  | **Q1**: 3D-sensitive (cos 0.37 at 0.5 Å), Δ 0.222 vs random 0.035 (≈6×). **Q3**: RankMe 256 vs random 100 (≈2.6×); PR 38 vs random 1.5 (≈25×) — pretraining sharpens the spectrum, with most of the gain in the bulk rather than the tail. **Q2**: gap small (+0.009) because mean-dir is already low (0.43) — embeddings are spread, projector matches without coord signal. REPA still has measurable headroom; what it teaches is geometric, not chemical. |
| ca-gearnet-random       | floor       | Reference for the row above. Random init has RankMe 100 / 512 (~20% of full rank — not collapsed, but tightly clustered: mean-dir 0.95, PR 1.5). **Q2 saturated**: every embedding ≈ the centroid, projector trivially saturates (gap +0.003). The "rank" is in the breadth of the cap on the sphere, not in any spread of variance. |
| **esm2-650M** (L33)     | **usable, but use mid-layers** | **Q2**: largest projector gap in the field (+0.053). **Q1**: AA-probe 0.998 confirms last-layer collapse to AA identity (cf. [esm/FINDINGS.md](esm/FINDINGS.md), recommends layers 24–30 for richer reps). Sequence-only — Q1.2 (3D sensitivity) is N/A by construction; gap is sequence-context, not 3D. |
| **mc-gearnet-edge**     | **unusable**| **Q3 catastrophe**: RankMe 12 / 3072, PR 1.0, **95% variance in 1 dim** (collapse), 507 dead dims, mean L2 norm 1.5×10⁶ (norm explosion). **Q2**: gap +0.002 — projector barely beats the mean-direction constant baseline. Confirms [mc_gearnet/FINDINGS.md](mc_gearnet/FINDINGS.md). |
| **pw-gearnet** (torsional) | **borderline** | **Q3**: RankMe 208 / 3072 (severe under-utilisation, ~17× MC-GearNet); PR 5.5 still low; 95% variance in 36 dims. **Q1**: AA-probe 0.928 (identity-driven), strong SS-Δ. **Q2**: gap +0.013 — ~4× random but well below ESM2. Usable if no better option, but ESM2 mid-layers and CA-GearNet are stronger choices. |
| **proteinmpnn-ca** (inv-fold) | **borderline, empirical test pending** | **Q3 cleanest of any 3D-aware encoder we've probed**: 0% sparsity, 0 dead dims, RankMe 84.9 / 128 = 66% utilisation, tight bounded norms (3.13 ± 0.20). **Q2**: gap +0.014, slightly above CA-GearNet but on a much higher floor (mean-dir 0.835 vs 0.425) — embeddings cluster tight. **Q1**: less coord-sensitive than CA-GN (1 Å noise → cos 0.726 vs 0.269), and protein-specificity is **10× weaker** (Δ 0.022 vs 0.222). AA probe 0.327 — moderate per-token identity leakage from inverse-folding pretraining. n=128 + n=256 runs in flight to settle. See [mpnn/FINDINGS.md](mpnn/FINDINGS.md). |

## Random-init baseline interpretation

Read the random row as the **architecture-only floor**. A Q1 / Q2 / Q3 metric that moves substantially off this floor is one where pretraining contributed; a metric that doesn't move is one where REPA has nothing useful to align against:

- **Q3.2 RankMe**: trained CA-GearNet 256 vs random 100 — only 2.6× separation. RankMe weights raw σ rather than σ², so it's less sensitive to the long tail of small singular values that the old "entropy on σ²" metric heavily penalised; under that older metric the ratio looked like 23×. **Participation ratio** tells the larger story (38 vs 1.5, ~25×) — pretraining concentrates variance into a moderate number of equally-weighted directions, which is the property that PR is designed to detect. ✓
- **Q1.4 Δ within−between**: trained 0.222 vs random 0.035 — pretraining encodes per-protein structural identity. ✓
- **Q1.1 AA probe**: trained 0.137 vs random 0.127 — *no meaningful gain*. Expected: CA-GearNet has no residue features at input; it cannot learn AA identity.
- **Q2 projector gap**: trained +0.009 vs random +0.003 — barely separated. Most of the trained signal is *already absorbed* by the (one-hot + position) projector input. This is the tightest leash on REPA's contribution and is consistent with the modest REPA gains seen in 128/256-residue runs.

Implication: when picking a REPA target encoder, the projector gap (Q2) is the most predictive single metric — ESM2's +0.053 is an order of magnitude above the others, and matches ESM-REPA's empirical advantage in val-loss curves. But always read it alongside Q1 to understand whether the headroom is geometry, sequence-context, or identity, since only the first directly serves a 3D generative model.

## Reframing through the iREPA lens (Singh et al. 2025)

Singh et al. ("What matters for Representation Alignment", arXiv:2512.10794) studied 27 vision encoders as REPA targets and found that **spatial structure** — pairwise patch-token cosine $S_{ij}$ tracking object/region relationships — correlates with generation FID at $|r| > 0.85$, while **global information** (ImageNet linear-probe accuracy) correlates only at $|r| = 0.26$. Their fix (iREPA) is a 4-line change: replace the per-token MLP projector with a `Conv2d(k=3, p=1)` and add instance-norm across the spatial dimension. The same recipe gives consistent gains across all 27 encoders.

That framing is a useful second pass over the verdicts above. The Q1/Q2/Q3 rubric still does the work — the iREPA lens just gives a sharper *causal* story for why ESM2's large +0.053 gap doesn't translate to strong REPA outcomes the way CA-GearNet's tiny +0.006 does.

### Three concepts, not two

Mapping our probes onto the paper's two axes requires being careful — we have measurements for at least three distinct things, and only two of them line up with the paper's axes:

- **Global info** (paper's concept): a property of the *whole input* read from a pooled vector — e.g. "what class is this image" via linear probe. Our closest analogue is the **within-vs-between protein cosine Δ** (Q1.4): do per-residue embeddings agree across residues that they belong to the same protein? Higher Δ ⇒ more pooled whole-input signal.
- **Per-token information richness** (NOT one of the paper's axes): can a linear probe read each token's own identity off its embedding? Our **AA-probe** (Q1.1) measures this. A token can be highly individuated (high probe accuracy) without carrying any whole-input signal and without its similarities to other tokens being structurally meaningful.
- **Spatial structure** (paper's concept): does the *pairwise* $S_{ij}$ between residues within one protein align with known structural relationships (CA–CA contact, SSE co-membership, 3D distance, sequence distance)?

Per-token richness and spatial structure are easy to confuse but distinct: ESM2-last has near-perfect per-token AA-identity readout, but its $S_{ij}$ is dominated by "are these two residues the same amino acid" — a categorical relation that's mostly orthogonal to 3D structure. So high per-token richness does not buy spatial structure.

It would be a mistake to read "AA probe acc 0.998" as "ESM2 has high global info." It does not — that probe is per-token, not global. ESM2's actual global-info proxy (within-vs-between Δ 0.098) is modest; lower than CA-GearNet's 0.222.

### Reframed table

| Encoder | Global info (within-vs-between Δ) | Per-token richness (AA probe) | Spatial structure | REPA verdict |
|---|---|---|---|---|
| **ca-gearnet** (trained) | **Strong** (Δ 0.222, 6× random) | Low (probe 0.137; CA-only input) | **Strong** (0.5 Å noise → cos 0.367; per-AA helix/sheet/loop Δ) | Usable |
| ca-gearnet-random | Floor (Δ 0.035) | Floor (probe 0.128) | Floor | Reference |
| **esm2-650M** (L33) | Modest (Δ 0.098) | **Very high** (probe 0.998) | **None by design** (sequence-only) | Saturated, gap +0.053 but conformation-invariant |
| **mc-gearnet-edge** | Collapsed (Δ 0.043) | Collapsed (probe 0.146) | Collapsed (RankMe 12/3072; 95% variance in 1 dim; norm 1.5e6) | Unusable |
| **pw-gearnet** (torsional) | Modest (Δ 0.102) | High (probe 0.928; AA in input) | **Strong** (3D-sensitive; torsional pretraining → SSE Δ) | Borderline |
| **proteinmpnn-ca** (inv-fold) | Floor-like (Δ 0.022) | Moderate (probe 0.327; CA-only input — inverse-folding leak) | **Weak** (1 Å noise → cos 0.726, ~3× smoother than CA-GN; SSE Δ ~0.025) | Borderline, empirical test pending |

### Reading the table

- **Spatial structure tracks REPA viability.** Both viable encoders (CA-GearNet, PW-GearNet-torsional) are spatial-structure-strong; both unusable-for-Q1-reasons encoders (CheMeleon-equivalent on the proteina side is ESM2-last) have no useful spatial structure. The paper's central claim transfers: pick by relational signal, not by pooled or per-token signal.
- **Global info does not track viability.** CA-GearNet (strong global Δ 0.222) and ESM2 (modest Δ 0.098) sit on opposite ends of viability — but for opposite reasons. CA-GN wins on spatial; ESM2 has none. The pooled signal is roughly orthogonal to the question of REPA quality.
- **Per-token richness does not track viability either.** ESM2 has the highest per-token richness (0.998) and is unusable as a 3D REPA target. CA-GearNet has the lowest (0.137) and is the most usable. Per-token AA discriminability is not what REPA needs — it cares about *between*-token relationships, not *within*-token identity.
- **MC-GearNet-Edge sits outside the two-axis picture entirely.** Every signal it might carry is destroyed by Q3 pathology (norm 1.5e6, rank 1.1/3072). Optimisation tractability is a third practical axis we hit because some structural encoders weren't trained to be drop-in differentiable targets — separate from the question of what kind of signal an encoder *would* carry if it were well-conditioned.
- **The ESM2 paradox dissolves.** Q2 alone says "+0.053 is the largest gap, ESM2 is the best target." The iREPA lens says: that gap is sequence-context headroom, dominated by per-token AA identity matching, with no relational structure to align against. The student denoiser needs relational signal, not categorical signal. So the gap exists but it's the wrong gap.

### What this changes for picks

- Per-token AA-identity probes and pooled within-vs-between Δ are both essentially uninformative about REPA target quality. The thing to evaluate is *between*-residue relational structure: does $S_{ij}$ track CA–CA contact, SSE co-membership, 3D distance?
- This de-prioritises "bigger ESM" as a candidate — adding sequence-only per-token richness will not help. It prioritises 3D-aware structural encoders (CA-GearNet, PW-GearNet, future structural-self-supervised models) and earlier ESM2 layers (24–30 per [esm/FINDINGS.md](esm/FINDINGS.md)) where last-layer collapse hasn't yet flattened the spatial signal.
- The iREPA projector recipe (replace `src/proteina/` REPA's per-token projector MLP with a neighbour-mixing layer over the residue chain or 3D graph + axis-instance-norm across residues) is a low-risk follow-up the framing predicts should help. Out of scope here, kept on the radar.

### What's missing if we wanted to verify the paper's claim quantitatively

The paper's spatial-structure axis is **per-protein pairwise $S_{ij}$ binned by a known structural relation** — for us: CA–CA contact (e.g. <8 Å), same-SSE co-membership, 3D-distance bins, sequence-distance bins. None of these are computed today. The closest existing measurement is the perturbation probe (Q1.2: 0.5 Å noise → cosine), which captures *aggregate* geometric sensitivity but not *which* relational structure the encoder represents.

Adding it would be ~100 lines reusing existing precomputed embeddings and existing protein structures (DSSP for SSE, CA distance from coords for contact). The deliverable would be a two-axis scatter — best relational-$S_{ij}$ delta on x, REPA headroom on y, one point per encoder — i.e. the paper's correlation plot rebuilt in our domain. Out of scope for this commit.
