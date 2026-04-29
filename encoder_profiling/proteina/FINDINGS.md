# Proteina REPA Encoder Comparison

Cross-encoder summary. Per-encoder depth lives in:
- [gearnet/FINDINGS.md](gearnet/FINDINGS.md) — CA-GearNet (trained)
- [gearnet_random/](gearnet_random/) — CA-GearNet random-init baseline (3 seeds)
- [esm/FINDINGS.md](esm/FINDINGS.md) — ESM-2 650M
- [mc_gearnet/FINDINGS.md](mc_gearnet/FINDINGS.md) — MC-GearNet-Edge
- [pw_gearnet/FINDINGS.md](pw_gearnet/FINDINGS.md) — ProteinWorkshop GearNet-Edge

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
   eff rank ↑          residue/protein Δ ↑
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
- Run: 2026-04-29, slurm job 28596016, A100, 12:33 wall.
- Inputs: 200 PDB train proteins, `--random-seed 0`, shared LMDB.
- Pipeline: [_probes/lib.py](_probes/lib.py).
- Reproduce: `sbatch hpc-scripts/proteina/encoder_profiling/run_all_encoders.sh`.
- Re-collate: `python encoder_profiling/proteina/collate.py` →
  [comparison.csv](comparison.csv), [figures/](figures/).

## Headline comparison

| Encoder                         | Embed dim | Eff rank   | AA probe acc | Δ within−between | Mean-dir | Best projector | **Gap**    |
|---------------------------------|----------:|-----------:|-------------:|-----------------:|---------:|---------------:|-----------:|
| **ca-gearnet** (trained)        |       512 |       77.5 |        0.137 |            0.222 |    0.425 |          0.432 | **+0.006** |
| ca-gearnet-random (3-seed avg)  |       512 | 3.3 ± 0.3  |  0.128 ± .00 |    0.035 ± .001  |    0.952 |          0.954 |   +0.003   |
| **esm2-650M** (last layer)      |      1280 |      360.6 |        0.998 |            0.098 |    0.671 |          0.724 | **+0.053** |
| **mc-gearnet-edge**             |      3072 |        1.1 |        0.150 |            0.043 |    0.855 |          0.853 |   −0.002   |
| **pw-gearnet** (torsional)      |      3072 |       12.2 |        0.927 |            0.102 |    0.710 |          0.719 |   +0.009   |

Columns map to the three questions:
- **Q1 evidence**: AA probe acc (Q1.1 residue identity), Δ within−between (Q1.4 protein-level identity); per-encoder files break this down further into 3D sensitivity (Q1.2) and structural/sequence context (Q1.3).
- **Q2 evidence**: Mean-dir, Best projector, **Gap = best − mean-dir** (a structural property of the encoder — see note below).
- **Q3 evidence**: Eff rank (the conditioning metric most tightly coupled to Q2 saturation); per-encoder files add sparsity, norms, and dead dims.

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
| **ca-gearnet** (trained)| **usable**  | **Q1**: 3D-sensitive (cos 0.37 at 0.5 Å), Δ 0.222 vs random 0.035 (≈6×). **Q3**: eff rank 77.5 vs random 3.3 (≈23×) — pretraining unlocks rank. **Q2**: gap small (+0.006) because mean-dir is already low (0.43) — embeddings are spread, projector matches without coord signal. REPA still has measurable headroom; what it teaches is geometric, not chemical. |
| ca-gearnet-random       | floor       | Reference for the row above. Random init collapses to eff rank 3.3 and mean-dir 0.95 (**Q3 collapse → Q2 saturated**): every embedding ≈ the centroid, projector trivially saturates (gap +0.003). |
| **esm2-650M** (L33)     | **usable, but use mid-layers** | **Q2**: largest projector gap in the field (+0.053). **Q1**: AA-probe 0.998 confirms last-layer collapse to AA identity (cf. [esm/FINDINGS.md](esm/FINDINGS.md), recommends layers 24–30 for richer reps). Sequence-only — Q1.2 (3D sensitivity) is N/A by construction; gap is sequence-context, not 3D. |
| **mc-gearnet-edge**     | **unusable**| **Q3 catastrophe**: eff rank 1.1/3072 (collapse), 507 dead dims, mean L2 norm 1.5×10⁶ (norm explosion). **Q2**: gap −0.002 (projector cannot beat constant baseline). Confirms [mc_gearnet/FINDINGS.md](mc_gearnet/FINDINGS.md). |
| **pw-gearnet** (torsional) | **borderline** | **Q3**: eff rank 12.2/3072 (severe under-utilisation, but ~10× MC-GearNet). **Q1**: AA-probe 0.927 (identity-driven), strong SS-Δ. **Q2**: gap +0.009 — ~3× random but well below ESM2. Usable if no better option, but ESM2 mid-layers and CA-GearNet are stronger choices. |

## Random-init baseline interpretation

Read the random row as the **architecture-only floor**. A Q1 / Q2 / Q3 metric that moves substantially off this floor is one where pretraining contributed; a metric that doesn't move is one where REPA has nothing useful to align against:

- **Q3.2 eff rank**: trained CA-GearNet 77.5 vs random 3.3 — pretraining unlocks rank. ✓
- **Q1.4 Δ within−between**: trained 0.222 vs random 0.035 — pretraining encodes per-protein structural identity. ✓
- **Q1.1 AA probe**: trained 0.137 vs random 0.128 — *no meaningful gain*. Expected: CA-GearNet has no residue features at input; it cannot learn AA identity.
- **Q2 projector gap**: trained +0.006 vs random +0.003 — barely separated. Most of the trained signal is *already absorbed* by the (one-hot + position) projector input. This is the tightest leash on REPA's contribution and is consistent with the modest REPA gains seen in 128/256-residue runs.

Implication: when picking a REPA target encoder, the projector gap (Q2) is the most predictive single metric — ESM2's +0.053 is an order of magnitude above the others, and matches ESM-REPA's empirical advantage in val-loss curves. But always read it alongside Q1 to understand whether the headroom is geometry, sequence-context, or identity, since only the first directly serves a 3D generative model.
