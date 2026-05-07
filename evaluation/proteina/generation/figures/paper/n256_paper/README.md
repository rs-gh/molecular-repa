# n=256 Paper-Protocol Sweep

Epoch-matched evaluation of 60M Proteina ProteinTransformer checkpoints at residue
length n=256, organised as ablations parallel to the n=128 paper sweep
([sweep_config.yaml: `n128_paper_layer` / `n128_paper_encoder` / `n128_paper_bs_lr`](../../sweep_config.yaml)).
Plot: [n256_paper_sweep.png](n256_paper_sweep.png) (currently still grouped by the
legacy `pdb / random / afdb` profiles — pending reorg into per-ablation rows).

Common training config (unless noted): 60M params, AdamW, lr=1×10⁻⁴, bs=24,
λ_repa=0.5 (REPA), GearNet-CA target encoder (REPA), per-residue averaging,
cosine similarity, combination_mode=additive, projector depth=2 (PDB) / 3 (AFDB).
Status legend: ✅ evaluated · ⏳ running · 🟡 ckpt available, not evaluated · ❌ no checkpoint.

Samples seen = `epoch × chains/epoch` with chains/epoch = 267,789 (PDB,
50 ≤ L ≤ 256) or 240,000 (AFDB-Swissprot). Same formula as plot x-ticks.

---

## 1. Layer ablation (PDB · L0/L4/L9 vs baseline)

REPA insertion layer at L4-default-encoder, λ=0.5, GearNet-CA, per_residue.

| Layer | Run | Step | Epoch | Samples | Status |
|---|---|---|---|---|---|
| baseline | `baseline_256_ep21` | 400K | 21 | 5.62M | ⏳ rerunning (28955895) |
| 0 | `repa_l0_256_ep26` | 400K | 26 | 6.96M | ✅ |
| 4 | `repa_l4_256_ep22` | 400K | 22 | 5.89M | ⏳ metric-only (28956346) |
| 9 | `repa_l9_256_ep25` | 400K | 25 | 6.69M | ✅ |

Currently lives in [`n256_paper_pdb`](../../results/n256_paper_pdb/).

## 2. Encoder ablation (PDB · L4 across encoders)

Same model + REPA layer (L4), varying the target encoder.

| Encoder | Run | Step | Epoch | Samples | Status |
|---|---|---|---|---|---|
| baseline (no REPA) | `baseline_256_ep21` | 400K | 21 | 5.62M | (shared with §1) |
| GearNet-CA (default) | `repa_l4_256_ep22` | 400K | 22 | 5.89M | (shared with §1) |
| GearNet-CA, **random init** | `repa_l4_256_random_ep17` | 200K | 17 | 4.55M | ⏳ metric-only (28956348) |
| PW-GearNet (structure) | `repa_l4_256_per_residue_pw_structure` | — | — | — | ❌ no run yet |
| PW-GearNet (torsional) | `repa_l4_256_per_residue_pw_torsional` | — | — | — | ❌ no run yet |
| ProteinMPNN | `repa_mpnn_l4_256_per_residue` | — | — | — | ⏳ training, 0 EMA ckpts |
| ESM2-650M (**L9, t=30**) | `repa_esm_l9_t30_256_per_residue` | 300K | 13 | 3.48M | 🟡 layer mismatch¹ |
| MC-GearNet-Edge | `repa_l4_256_per_residue_mc_edge` | — | — | — | ❌ no run; flagged unusable at n=512² |

¹ Only ESM-L9 (with t=30 conditioning) is being trained at n=256; n=128 paper ablation uses ESM-L4.
Comparison would be apples-to-oranges. Either retrain ESM-L4-256 for parity, or footnote the layer in the encoder row.

² Per [project_encoder_characterizations.md](../../../../../home/sr2173/.claude/projects/-home-sr2173-git-molecular-repa/memory/project_encoder_characterizations.md):
MC-GearNet-Edge has effective rank 1.1 / 3072 + norm explosion — likely unusable for REPA.

Existing rows live in [`n256_paper_pdb`](../../results/n256_paper_pdb/) (default L4) and [`n256_paper_random`](../../results/n256_paper_random/) (random-init).

## 3. Lambda ablation (PDB · L4 · GearNet-CA · varying λ)

| λ | Run | Step | Epoch | Samples | Status |
|---|---|---|---|---|---|
| 0.5 (default) | `repa_l4_256_ep22` | 400K | 22 | 5.89M | (shared with §1) |
| 1.0 | `repa_l4_256_per_residue_lambda1` | 100K | 8 | 2.14M | 🟡 only early ckpt available |
| 2.0 | `repa_l4_256_per_residue_lambda2` | — | — | — | ⏳ training, 0 EMA ckpts |

Not yet a registered profile; mirrors `n128`'s deferred lambda ablation note in [sweep_config.yaml:189-191](../../sweep_config.yaml#L189-L191).

## 4. Batch-size + LR ablation (PDB)

**Empty for n=256.** No bs/lr variants registered or trained beyond the default
bs=24 / lr=1e-4. Mirroring n128's [`n128_paper_bs_lr`](../../sweep_config.yaml#L217)
(8 cells: bs ∈ {24, 80} × lr ∈ {1×, 3×} ± REPA at 200K + 400K) is a separate
training-job batch — not feasible without new training runs.

## 5. Averaging ablation (PDB · per_sample vs per_residue)

Not in the n=128 framework, but checkpoints exist for n=256 — proposed addition.

| Layer | Run | Step | Epoch | Samples | Status |
|---|---|---|---|---|---|
| L0 per_residue | `repa_l0_256_ep26` | 400K | 26 | 6.96M | ✅ (shared with §1) |
| L0 per_sample | `repa_l0_256_per_sample` | 300K | 19 | 5.09M | 🟡 ckpt avail, not evaluated |
| L4 per_residue | `repa_l4_256_ep22` | 400K | 22 | 5.89M | (shared with §1) |
| L4 per_sample | `repa_l4_256_per_sample` | 400K | 25 | 6.69M | 🟡 ckpt avail, not evaluated |
| L9 per_residue | `repa_l9_256_ep25` | 400K | 25 | 6.69M | ✅ (shared with §1) |
| L9 per_sample | `repa_l9_256_per_sample` | 300K | 20 | 5.36M | 🟡 ckpt avail, not evaluated |

Three new evaluations needed (`repa_l{0,4,9}_256_per_sample`).

## 6. Dataset ablation (PDB vs AFDB-Swissprot)

| Run | Step | Epoch | Samples | Status |
|---|---|---|---|---|
| `baseline_256_ep21` (PDB) | 400K | 21 | 5.62M | ⏳ rerunning |
| `baseline_afdb_256_ep20` (AFDB) | 200K | 20 | 4.80M | ⏳ metric-only (28956349) |
| `repa_l4_256_ep22` (PDB, L4) | 400K | 22 | 5.89M | ⏳ metric-only |
| `repa_l4_afdb_256_ep20` (AFDB, L4) | 200K | 20 | 4.80M | ✅ |

Currently lives in [`n256_paper_afdb`](../../results/n256_paper_afdb/) (AFDB) and [`n256_paper_pdb`](../../results/n256_paper_pdb/) (PDB).

## 7. External-baseline comparisons (TODO)

Need separate inference plumbing — none currently wired through `evaluate.py`.

| Model | What we need |
|---|---|
| Proteina 60M (NVIDIA NGC pretrained) | Download NGC ckpt; layer count is 12 (ours is 10) — see [project_proteina_60m_layer_mismatch.md](../../../../../home/sr2173/.claude/projects/-home-sr2173-git-molecular-repa/memory/project_proteina_60m_layer_mismatch.md). Inference compatible with our sampler if layer count override is added. Plot by normalised depth when comparing layer-internals. |
| FrameFlow | External codebase / repo. Generate PDBs externally → drop into our metric pipeline via `--ckpts_file` path or as an ad-hoc directory of PDBs. |
| Genie | Same shape as FrameFlow. |

Suggested entry point: a separate `n256_paper_external` profile with one row per model, `metrics: "fid,designability,diversity"`, and PDB pools dropped into `eval_output/<external_name>_step_-1/` so the metric pipeline picks them up without a generation pass.

---

## Pending jobs (after current 5 retries land)

1. **Reorg sweep_config.yaml** — split the data we have across `n256_paper_layer / _encoder / _dataset / _averaging` profiles. No new compute; just re-bucket existing JSONL rows.
2. **Averaging ablation evaluations** (3 new tasks): `repa_l0_256_per_sample` (ep19), `repa_l4_256_per_sample` (ep25), `repa_l9_256_per_sample` (ep20). All have full 1125-PDB pools to generate (~3h each fresh, can use `--no-fast_inference`).
3. **Encoder gap-fill (training)** — PW-structure, PW-torsional, MPNN-L4 (let it train), ESM-L4 retrain. Deferred until training plan is set.
4. **Lambda ablation** — wait for `lambda1` to reach 200K+ and `lambda2` to write its first EMA ckpt.
5. **External baselines** — design separate inference path before kicking off.

---

## Experimental protocol

Generation pool (one per checkpoint, drives all metric families):
- Lengths: {50, 75, 100, 125, 150, 175, 200, 225, 250} (9 bins, step 25)
- Samples per length: 125 → **1,125 PDBs / checkpoint**
- Sampler: 400 inference steps, log schedule, temp=0.45, gt=1/t (paper App. F)
- Eager fp32 inference (torch.compile disabled — SDPA contiguous-bias bug)

Metrics:

| Metric | Pool | Notes |
|---|---|---|
| **FID** (PDB, AFDB) | full 1,125 | GearNet-CA features, 8M reference PDBs |
| **fJSD** (C/A/T) | full 1,125 | Per-feature Jensen-Shannon, paper §4.2 |
| **fS** (C/A/T) | full 1,125 | First-step distance |
| **Designability** | 5 lengths × 50 = 250 | ProteinMPNN(8 seqs, CA model) → ESMFold → scRMSD<2; lengths {50,100,150,200,250} |
| **Diversity** | designable subset (≤250) | Pairwise TM-score + cluster count |
| **Novelty (centroid)** | designable subset | Max TM-score vs training-set centroids; <0.5 = novel |
| **Novelty (Foldseek)** | designable subset | `foldseek easy-search` vs PDB + AFDB-Swissprot DBs; alignment-type=2, sensitivity=9.5, max-seqs=1000 |

Configs:
- Sweep profile: [evaluation/proteina/generation/sweep_config.yaml](../../sweep_config.yaml) (`n256_paper_{pdb,random,afdb}` under `_paper_defaults`)
- Inference: [src/proteina/configs/experiment_config/inference/inference_fid_60m_paper.yaml](../../../../../src/proteina/configs/experiment_config/inference/inference_fid_60m_paper.yaml)
- Per-checkpoint result rows: `../../results/n256_paper_{pdb,random,afdb}/sweep_results.{jsonl,csv,md}`
