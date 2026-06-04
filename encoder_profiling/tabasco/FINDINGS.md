# Tabasco REPA Encoder Comparison

Cross-encoder summary. Per-encoder depth lives in:
- [chemeleon/FINDINGS.md](chemeleon/FINDINGS.md) — CheMeleon (chemprop, frozen, 2D bond-graph)
- [mace/FINDINGS.md](mace/FINDINGS.md) — MACE-OFF small (frozen, 3D equivariant)

## How to read these findings

Every per-encoder file is organised around three questions. Together they form the actual selection rubric for REPA targets — each on its own is misleading. (Same framing as [proteina's cross-encoder summary](../proteina/FINDINGS.md); the metrics differ but the questions don't.)

- **Q1 — What information does the encoder encode?** Upper bound on what REPA can transfer (atom identity, 3D geometric sensitivity, chemical / topological context, molecule-level identity). If the encoder doesn't carry information X, REPA can never teach the student X.
- **Q2 — How much of that is reachable from cheap inputs?** A small MLP from `(atom-onehot) → encoder` — per-atom, no neighbours — absorbs most of the per-atom cosine alignment. The tabasco student's projector reads off a transformer hidden state computed from *strictly more*: `(atom-onehot) + noisy 3D coords + cross-atom self-attention + flow-matching timestep`. So:
  - **Saturation floor** = cosine reachable from atom identity alone (or from random inputs, when those approach the same number).
  - **Gap above it** = cosine the student can only reach by leveraging coords, attention, or timestep. That's REPA's actual operating budget.
  - For 3D-aware encoders (MACE) the headroom would be coord-driven if the encoder discriminated conformers strongly. For 2D encoders (CheMeleon) the headroom is conformation-invariant by construction.
- **Q3 — Is the encoder a tractable optimisation target?** Sparsity, rank collapse, dead dims, and norm explosions degrade the gradient signal even when the embedding is informative.

The diagnostic value lives in the **gap between Q1 and Q2 = REPA headroom**, modulated by Q3 conditioning:

```
            pretraining quality
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   RankMe / PR ↑       atom-type / molecule Δ ↑
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

Caveat: a large gap alone is necessary but not sufficient. CheMeleon's projector gap is non-trivial, but the headroom is *2D bond-graph context* — conformation-invariant and unable to guide 3D geometry. Read the gap alongside Q1 to understand *what kind* of information sits in the headroom.

## Setup
- Investigations: CheMeleon 2026-03-18 (with follow-up 2026-04-02), MACE 2026-03-19 (with follow-up 2026-04-02).
- Inputs: GEOM train molecules and QM9 train molecules (~100–500 mols per probe; per-encoder files give exact counts).
- Pipelines: per-encoder `investigate.py` / `explore_mace.py` + `probe_and_saturation.py`.
- These pre-date the Q1/Q2/Q3 unified probe library used by proteina; the numbers below were produced by the per-encoder scripts and reorganised into Q1/Q2/Q3 form rather than re-collected.

## Headline comparison

| Encoder           | Embed dim | Rank (note)                          | Sparsity | Atom probe acc | Atom-type Δ within−between | 3D-aware? | Mean-dir | Best projector | **Gap**    |
|-------------------|----------:|--------------------------------------|---------:|---------------:|---------------------------:|:---------:|---------:|---------------:|-----------:|
| **CheMeleon** (2048-d) | 2048 | RankMe 1166 (GEOM) / 1195 (QM9)¹ | 93.8% | 1.000 | 0.093 (QM9) | **No** (2D-only) | ~0.43 | 0.471 | **+0.04**  |
| **MACE-OFF small** (192-d) | 192 | RankMe 40.6                | 0.0% | 1.000 | 0.260 | Yes (weak) | ~0.86 | 0.863 | **+0.005** |

¹ Re-run 2026-06-04 (after fixing a venv drift — rogue `rdkit-pypi 2022.9.5` shadowed the locked `rdkit 2025.9.3`, so the LMDBs would not depickle). Now RankMe = `exp(H(p))` with `p = σᵢ / Σσ`, the same metric as MACE's 40.6, so the two are directly comparable. **Higher is not better here**: CheMeleon's rank is high because its variance is *diffuse and sparse* (93.8% zeros, 500 dims for 90% variance), not collapsed — it overflows the 128-d tabasco projector input (a bottleneck), the opposite failure mode to MC-GearNet's collapse. The old stale value was `138 = exp(H(p))` with `p = σᵢ² / Σσ²` (the old proteina convention); discard it.

Columns map to the three questions:
- **Q1 evidence**: Atom probe acc (Q1.1), atom-type Δ within−between (Q1.3 — same-element / different-environment discrimination), 3D-aware? (Q1.2). Per-encoder files break Q1 down further (molecule-level identity Q1.4, conformer sensitivity, etc.).
- **Q2 evidence**: Mean-dir, Best projector, **Gap = best − mean-dir** (a structural property of the encoder).
- **Q3 evidence**: Rank (RankMe for both, comparable as of the 2026-06-04 CheMeleon re-run), Sparsity. Per-encoder files add norms, dead dims, and the threshold-based rank for cross-checking.

### What the "Gap" column is — and isn't

`Gap = Best projector − Mean-dir` measures *how much (atom-onehot) lifts cosine over a constant prediction* on this encoder. Worth being explicit:

- **What it is**: an *estimate of what could be learned* — a structural property of the encoder that upper-bounds how much room exists above the cheap-input floor for any input to extract more cosine. A small gap means the encoder's variance is so tight that even rich inputs probably can't push much further; a large gap means there is room in principle.
- **What it isn't**: REPA's actual operating budget. The budget is `training_cos − best_(atom-onehot)_floor`, observable only from wandb training logs, not from this probe.

The two quantities to keep separate:

| Quantity | Meaning | Where it lives |
|----------|---------|----------------|
| `Best − Mean-dir` (this column) | How much *cheap inputs* lift cosine over a constant prediction. Estimate of *what could be learned* if richer inputs (coords / attention / timestep) helped further. | This table — measurable from the encoder alone. |
| `training_cos − best_(atom-onehot)` | How much the *student transformer* lifts cosine over what a cheap-input MLP achieves. The actual REPA contribution. | wandb logs — only visible during training. |

Workflow: the table tells you which encoders *could* offer headroom; the training cosine vs the `best_(atom-onehot)` floor tells you whether the student *actually used* that headroom. A run that plateaus at the floor is a run where REPA contributed nothing the projector wouldn't do alone.

## Verdicts

| Encoder       | Verdict      | Deciding metric(s) |
|---------------|--------------|--------------------|
| **CheMeleon** | **unusable as a 3D-REPA target** | **Q1.2 catastrophic**: identical embeddings for all conformers (L2 = 0.000) — REPA cannot teach 3D geometry. **Q3.1 catastrophic**: 93.8% sparsity (ReLU) — cosine operates over ~130 active dims of 2048, gradients dominated by activation pattern. **Q3.2**: 500 dims for 90% variance vs 128-d projector input — bottleneck. **Q1.1 strong** (probe 1.000) but trivial in QM9 (max 9 atoms). **Q2 floor 0.43 with gap +0.04** — the gap exists, but it's 2D bond-graph context, conformation-invariant, useless for 3D guidance. |
| **MACE-OFF small** | **borderline; saturated** | **Q1.2 weak-positive**: cos 0.998 between conformers — geometry-aware but the 3D signal is small (local environments dominate, MACE was trained on energies). **Q3 clean**: 0% sparsity, dense gradients, RankMe 40.6 fits projector easily, no dead dims, no norm explosion. **Q1.1 / Q1.3 strong**: probe 1.000, atom-type Δ = 0.260 (3× CheMeleon). **Q2 saturated**: mean-dir floor ≈ 0.86, gap +0.005. Random and atom-onehot inputs both reach ~0.86 — *the projector saturates without any input signal*. Even a perfect transformer can lift cosine ≤ 0.14 above the floor; most of that headroom is local geometry, not global conformation. |

## Why no random-init baseline (cf. proteina)

The proteina table includes a `gearnet-random` row as the architecture-only floor. Tabasco doesn't have an equivalent: CheMeleon and MACE-OFF are off-the-shelf pretrained checkpoints, not architectures we own. A random-init MACE/CheMeleon would be informative but isn't trivially available, and the proteina random-init experiment showed the architecture-only contribution to most metrics is small — the diagnostic loss is acceptable.

## Implications for tabasco REPA design

Reading across the two encoders:

1. **Neither is a clean win.** CheMeleon fails Q1.2 (no 3D signal at all) and Q3 (sparsity + bottleneck). MACE passes Q3 but saturates so hard at Q2 that there's almost no operating budget. The choice is between *informative-but-2D* and *3D-aware-but-saturated*.
2. **CheMeleon's gap is not 3D headroom.** Its +0.04 gap is real, but it represents 2D bond-graph context the encoder exposes. Aligning to CheMeleon teaches atom identity and bond-graph topology — for QM9 with max 9 atoms, atom identity is trivially learnable already; for GEOM the topology signal could matter, but it cannot guide 3D structure.
3. **MACE's gap is real 3D headroom but tiny.** The conformer-pair cosine (0.998) tells you the gradient for 3D structure is weak; the projector saturation (gap +0.005) tells you the projector has no room to extract even that.
4. **Consider supplementary signals.** Both encoders motivate auxiliary objectives that don't go through cosine alignment: a direct atom-type CE loss (cheaper than CheMeleon-REPA, same effect on Q1.1), or an explicit pairwise-distance loss for 3D geometry (which neither encoder provides cleanly).
5. **The MACE saturation is a calibration warning.** Any MACE-REPA training that reports `cos > 0.86` is hitting the projector floor — it does not by itself indicate the transformer learned anything. Genuine learning requires `training_cos > 0.86 + ε` and ideally `> 0.90`.

For per-encoder methodology, raw numbers, and historical context, see the per-encoder FINDINGS files linked above.

## Reframing through the iREPA lens (Singh et al. 2025)

Singh et al. ("What matters for Representation Alignment", arXiv:2512.10794) studied 27 vision encoders as REPA targets and found that **spatial structure** — pairwise patch-token cosine $S_{ij}$ tracking object/region relationships — correlates with generation FID at $|r| > 0.85$, while **global information** (ImageNet linear-probe accuracy) correlates only at $|r| = 0.26$. Their fix (iREPA) is a 4-line change: replace the per-token MLP projector with a `Conv2d(k=3, p=1)` and add instance-norm across the spatial dimension. The same recipe gives consistent gains across all 27 encoders.

That framing is a useful second pass over the verdicts above. The Q1/Q2/Q3 rubric still does the work — the iREPA lens just gives a sharper *causal* story for what the headroom numbers were measuring.

### Three concepts, not two

Mapping our probes onto the paper's two axes requires being careful — we have measurements for at least three distinct things, and only two of them line up with the paper's axes:

- **Global info** (paper's concept): a property of the *whole input* read from a pooled vector — e.g. "what class is this image" via linear probe. Our closest analogue is the **within-vs-between molecule cosine Δ**: do per-atom embeddings agree across atoms that they belong to the same molecule? Higher Δ ⇒ more pooled whole-input signal.
- **Per-token information richness** (NOT one of the paper's axes): can a linear probe read each token's own identity off its embedding? Our **atom-ID probe** measures this. A token can be highly individuated (high probe accuracy) without carrying any whole-input signal and without its similarities to other tokens being structurally meaningful.
- **Spatial structure** (paper's concept): does the *pairwise* $S_{ij}$ between tokens within one molecule align with known structural relationships (ring co-membership, bond-graph distance, 3D distance)?

Per-token richness and spatial structure are easy to confuse but distinct: an encoder can have near-perfect per-token atom-ID readout while its $S_{ij}$ is dominated by "are these two atoms the same element" — a categorical relation that's mostly orthogonal to 3D geometry. So high per-token richness does not buy spatial structure.

### Reframed table

| Encoder | Global info (within-vs-between Δ) | Per-token richness | Spatial structure | REPA verdict |
|---|---|---|---|---|
| **CheMeleon** | Modest (Δ 0.093 QM9 / 0.044 GEOM) | High (atom-ID probe 1.000) | **None** (conformer L2 = 0.000; 2D bond-graph; 93.8% sparsity; projector saturates ~0.47 from random) | Unusable |
| **MACE-OFF small** | Not directly probed (see follow-up below) | High (atom-ID probe 1.000; same-element-different-env Δ 0.260) | **Weak-positive** (conformer cos 0.998 — sensitive but tiny; dense; RankMe 40.6/192) | Saturated, gap +0.005 |

### Reading the table

- **Spatial structure tracks REPA viability — but neither encoder has much of it.** CheMeleon has none (2D-only by construction), MACE has the right kind but very little magnitude. The paper's central claim ("spatial structure is what matters") predicts both should be poor REPA targets, which matches our verdicts.
- **Per-token richness does not track viability.** Both encoders score atom-ID probe 1.000 (trivial in QM9, but still). That signal is uninformative about REPA quality — it's per-atom self-identity, not between-atom relational signal.
- **Global info is barely above random.** CheMeleon's modest mol-level Δ is consistent with "encoder distinguishes molecules a little, but not via geometric structure." MACE wasn't directly measured here.
- **MACE has the right *kind* of signal, just very little magnitude.** Conformer cos 0.998 means almost-no variation across conformers; the framing predicts MACE-large or any 3D encoder with stronger conformer sensitivity should do better.

### What this changes for picks

- Per-token atom-ID probes and pooled mol-level Δ are both essentially uninformative about REPA target quality. The thing to evaluate is *between*-atom relational structure: does $S_{ij}$ track ring co-membership, bond-graph distance, 3D contact?
- This de-prioritises "bigger CheMeleon" — adding per-token richness or 2D context won't help. It prioritises 3D-aware encoders with strong conformer sensitivity, dense per-atom embeddings, and high RankMe — i.e. the MACE direction, but with stronger 3D signal magnitude.
- The iREPA projector recipe (replace `src/tabasco/` REPA's per-token projector MLP with a neighbour-mixing layer over the molecular graph + axis-instance-norm across atoms) is a low-risk follow-up the framing predicts should help. Out of scope here, kept on the radar.

### What's missing if we wanted to verify the paper's claim quantitatively

The paper's spatial-structure axis is **per-molecule pairwise $S_{ij}$ binned by a known structural relation** — for us: bond-graph distance, ring co-membership, 3D Euclidean distance, functional-group co-membership. None of these are computed today. The closest existing measurement is the same-element-different-env Δ (0.260 for MACE, 0.093 for CheMeleon) which is one slice of the relational structure but not directly comparable to the paper's metrics.

Adding it is ~100 lines reusing existing precomputed embeddings. The deliverable would be a two-axis scatter — best relational-$S_{ij}$ delta on x, REPA headroom on y, one point per encoder — i.e. the paper's correlation plot rebuilt in our domain. Out of scope for this commit.

### Aside: filling the one missing global-info number

MACE's mol-level within-vs-between Δ is the only directly measurable cell in the table that we don't have. Easy follow-up: ~30-line script over existing precomputed embeddings, no encoder rerun. Not blocking — the paper argues this column is approximately uninformative for REPA target quality regardless.
