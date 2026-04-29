# Proteina REPA Encoder Comparison

Cross-encoder summary. Per-encoder depth lives in:
- [gearnet/FINDINGS.md](gearnet/FINDINGS.md) — CA-GearNet (trained)
- [gearnet_random/](gearnet_random/) — CA-GearNet random-init baseline (3 seeds)
- [esm/FINDINGS.md](esm/FINDINGS.md) — ESM-2 650M
- [mc_gearnet/FINDINGS.md](mc_gearnet/FINDINGS.md) — MC-GearNet-Edge
- [pw_gearnet/FINDINGS.md](pw_gearnet/FINDINGS.md) — ProteinWorkshop GearNet-Edge

## Setup
- Run: 2026-04-29, slurm job 28596016, A100, 12:33 wall.
- Inputs: 200 PDB train proteins, `--random-seed 0`, shared LMDB.
- Pipeline: [playground/proteina/_encoder_probes/lib.py](../../playground/proteina/_encoder_probes/lib.py).
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

`Mean-dir` = test-set cos to the centroid; `Best projector` = test-set cos achieved by a 3-layer MLP trained from `onehot+pos` to the encoder's embedding (REPA-trainable cap). **Gap = best − mean-dir** is the headroom REPA has after the projector absorbs identity & position signal.

## Verdicts

| Encoder                 | Verdict     | Deciding metric(s) |
|-------------------------|-------------|--------------------|
| **ca-gearnet** (trained)| **usable**  | Eff rank 77.5 vs random 3.3 (≈23×); Δw-b 0.222 vs 0.035 (≈6×). Projector gap is small (+0.006) because mean-dir is already low (0.43) — embeddings are spread, so the projector matches without coord signal. REPA still has measurable headroom. |
| ca-gearnet-random       | floor       | Reference for the row above. Random init collapses to eff rank 3.3 and mean-dir 0.95: every embedding ≈ the centroid, so the projector trivially saturates (gap +0.003). |
| **esm2-650M** (L33)     | **usable, but use mid-layers** | Largest projector gap in the field (+0.053). AA-probe 0.998 confirms last-layer collapse to AA identity (cf. [esm/FINDINGS.md](esm/FINDINGS.md), recommends layers 24–30). Sequence-only — no 3D guidance. |
| **mc-gearnet-edge**     | **unusable**| Eff rank 1.1/3072 (collapse), 507 dead dims, mean L2 norm 1.5×10⁶ (norm explosion), gap −0.002. Confirms [mc_gearnet/FINDINGS.md](mc_gearnet/FINDINGS.md). |
| **pw-gearnet** (torsional) | **borderline** | Eff rank 12.2/3072 (severe under-utilization), AA-probe 0.927 (identity-driven), gap +0.009 — ~3× random but well below ESM2. Same shortlist as before: usable if no better option, but ESM2 mid-layers and CA-GearNet are stronger choices. |

## Random-init baseline interpretation

Read the random row as the architecture-only floor. The metrics that move *substantially* off the floor are where pretraining contributed:

- **Eff rank:** trained CA-GearNet 77.5 vs random 3.3 — pretraining unlocks rank.
- **Δ within−between protein similarity:** trained 0.222 vs random 0.035 — pretraining encodes per-protein structural identity.
- **AA probe:** trained 0.137 vs random 0.128 — *no meaningful gain*. Expected: CA-GearNet has no residue features at input; it cannot learn AA identity.
- **Projector gap:** trained +0.006 vs random +0.003 — barely separated. Most of the trained signal is *already absorbed* by the (one-hot + position) projector input. This is the tightest leash on REPA's contribution and is consistent with the modest REPA gains seen in 128/256-residue runs.

Implication: when picking a REPA target encoder, the projector gap is the most predictive single metric — ESM2's +0.053 is an order of magnitude above the others, and matches ESM-REPA's empirical advantage in val-loss curves.
