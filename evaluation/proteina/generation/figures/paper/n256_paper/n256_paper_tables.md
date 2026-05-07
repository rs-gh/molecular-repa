# n=256 paper-protocol sweep — ablation table

Companion to `n256_paper_sweep.png`. Pool: 1125 PDBs at L∈{50,75,100,125,150,175,200,225,250} × 125 for FID/fJSD/fS_T (N=1125); designability on 50/L × 5 paper lengths {50,100,150,200,250} (N=250); diversity/novelty on the designable subset.

**N per metric:** PDB FID=N=1125, AFDB FID=N=1125, Fold Score (Topo)=N=1125, PDB fJSD (Topo)=N=1125, Designability=N=250, scRMSD mean (Å)=N=250, Diversity (clusters)=designable, Diversity (pairwise TM)=designable.

Best per metric within each ablation block is **bolded**.

| Run | Step | bs | des N | PDB FID (↓) | AFDB FID (↓) | Fold Score (Topo) (↑) | PDB fJSD (Topo) (↓) | Designability (↑) | scRMSD mean (Å) (↓) | Diversity (clusters) (↑) | Diversity (pairwise TM) (↓) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Layer ablation — (L0/L4/L9 vs baseline, PDB, GearNet-CA, λ=0.5, per_residue, bs=24)** |  |  |  |  |  |  |  |  |  |  |  |
| Baseline ep21 | 400K | 24 | 250 | 441.8 | 462.0 | 30.08 | 2.510 | 0.124 | 6.31 | 6.00 | **0.156** |
| REPA L0 ep26 | 400K | 24 | 250 | **265.7** | **318.5** | 34.33 | **1.682** | 0.084 | 7.12 | 4.00 | 0.192 |
| REPA L4 ep22 | 400K | 24 | 250 | 357.4 | 410.6 | **40.35** | 1.866 | 0.320 | 5.35 | 15.20 | 0.172 |
| REPA L9 ep25 | 400K | 24 | 250 | 297.2 | 337.3 | 33.58 | 2.228 | **0.484** | **4.67** | **16.80** | 0.357 |
| **Encoder ablation — (REPA L4 with 6 target encoders, PDB, λ=0.5, per_residue, bs=24)** |  |  |  |  |  |  |  |  |  |  |  |
| Baseline ep21 | 400K | 24 | 250 | 441.8 | 462.0 | 30.08 | 2.510 | 0.124 | 6.31 | 6.00 | **0.156** |
| CA-GearNet ep22 | 400K | 24 | 250 | **357.4** | **410.6** | **40.35** | **1.866** | **0.320** | **5.35** | **15.20** | 0.172 |
| GearNet random ep17 | 200K | 24 | 250 | 600.2 | 673.5 | 38.42 | 2.681 | 0.160 | 6.44 | 8.00 | 0.172 |
| PW-Structure | — | 24 | — | — | — | — | — | — | — | — | — |
| PW-Torsional | — | 24 | — | — | — | — | — | — | — | — | — |
| ProteinMPNN | — | 24 | — | — | — | — | — | — | — | — | — |
| ESM2 | — | 24 | — | — | — | — | — | — | — | — | — |
| **Step-matched reference (encoder ablation, ep17 anchor)** |  |  |  |  |  |  |  |  |  |  |  |
| REPA L9 ep17 | 300K | 24 | 250 | 376.0 | 415.9 | 33.87 | 2.471 | 0.040 | 9.02 | 4.00 | 0.134 |
| **Dataset ablation — (PDB vs AFDB-Swissprot, baseline + REPA L4, GearNet-CA, λ=0.5, per_residue, bs=24)** |  |  |  |  |  |  |  |  |  |  |  |
| Baseline PDB ep21 | 400K | 24 | 250 | 441.8 | 462.0 | 30.08 | 2.510 | 0.124 | 6.31 | 6.00 | **0.156** |
| Baseline AFDB ep20 | 200K | 24 | 250 | 374.5 | 367.4 | 10.05 | 3.193 | 0.656 | **2.17** | **28.00** | 0.212 |
| REPA L4 PDB ep22 | 400K | 24 | 250 | 357.4 | 410.6 | **40.35** | **1.866** | 0.320 | 5.35 | 15.20 | 0.172 |
| REPA L4 AFDB ep20 | 200K | 24 | 250 | **207.5** | **223.6** | 21.08 | 2.129 | **0.684** | 2.91 | 20.40 | 0.229 |
| **λ ablation — (REPA L4, PDB, GearNet-CA, per_residue, bs=24, varying λ)** |  |  |  |  |  |  |  |  |  |  |  |
| λ=0.5 ep22 | 400K | 24 | 250 | **357.4** | **410.6** | **40.35** | **1.866** | **0.320** | **5.35** | **15.20** | **0.172** |
| λ=1.0 | — | 24 | — | — | — | — | — | — | — | — | — |
| λ=2.0 | — | 24 | — | — | — | — | — | — | — | — | — |
| **Averaging ablation — (REPA L0/L4/L9, per_sample vs per_residue, PDB, GearNet-CA, λ=0.5, bs=24)** |  |  |  |  |  |  |  |  |  |  |  |
| L0 per_residue ep26 | 400K | 24 | 250 | **265.7** | **318.5** | 34.33 | **1.682** | 0.084 | 7.12 | 4.00 | 0.192 |
| L0 per_sample | — | 24 | — | — | — | — | — | — | — | — | — |
| L4 per_residue ep22 | 400K | 24 | 250 | 357.4 | 410.6 | **40.35** | 1.866 | 0.320 | 5.35 | 15.20 | **0.172** |
| L4 per_sample | — | 24 | — | — | — | — | — | — | — | — | — |
| L9 per_residue ep25 | 400K | 24 | 250 | 297.2 | 337.3 | 33.58 | 2.228 | **0.484** | **4.67** | **16.80** | 0.357 |
| L9 per_sample | — | 24 | — | — | — | — | — | — | — | — | — |
| **Batch size + LR ablation — (bs ∈ {24,80}? × lr ∈ {1×,3×} × ±REPA)** |  |  |  |  |  |  |  |  |  |  |  |
| (no n=256 bs/lr variants trained yet) | — | — | — | — | — | — | — | — | — | — | — |

## Pending rows (status as of 2026-05-07)

**Layer ablation** — complete (4/4). No pending rows.

**Encoder ablation** — 3/7 cells filled. Pending:
- `repa_l4_256_per_residue_pw_structure` — config exists ([training_repa_l4_256_per_residue_pw_structure.yaml](../../../../../src/proteina/configs/experiment_config/training/256/pw_gearnet/per_residue/training_repa_l4_256_per_residue_pw_structure.yaml)); no run dir on disk → needs training kickoff.
- `repa_l4_256_per_residue_pw_torsional` — same status, training kickoff needed.
- `repa_mpnn_l4_256_per_residue` — training in flight as of screenshot 2026-05-06 (started 0505-20:19); 0 EMA checkpoints yet → wait for ckpt to mature, then evaluate.
- `repa_esm_l4_256_per_residue` — does **not** exist. The on-disk ESM run is `repa_esm_l9_t30_256_per_residue` (L9 + t=30 conditioning), apples-to-oranges with the L4 default. Decision needed: retrain at L4 for parity with n=128, or footnote the layer mismatch and use L9-t30 ep13 (300K) as a placeholder.
- MC-GearNet-Edge — explicitly skip per [project_encoder_characterizations.md](../../../../../home/sr2173/.claude/projects/-home-sr2173-git-molecular-repa/memory/project_encoder_characterizations.md): effective rank 1.1/3072 + norm explosion at n=512, expected to be unusable here too.

**Step-matched reference** — complete (1/1). `repa_l9_256_ep17` lives here as a step/sample-matched comparator for the random-init encoder row (both at ep17 / 200-300K).

**Dataset ablation** — complete (4/4). No pending rows.

**λ ablation** — 1/3 cells filled. Pending:
- `repa_l4_256_per_residue_lambda1` — only step 100K / ep8 EMA available (very early); evaluate as-is, or wait for it to reach step 200K+.
- `repa_l4_256_per_residue_lambda2` — training, 0 EMA checkpoints yet. Wait.

**Averaging ablation** — 3/6 cells filled. Pending:
- `repa_l0_256_per_sample` — last EMA ep19 / 300K (5.09M smp). Ready to evaluate now (close to per_residue ep26 sample-budget).
- `repa_l4_256_per_sample` — last EMA ep25 / 400K (6.69M smp). Ready to evaluate now (sample-matched to per_residue ep22).
- `repa_l9_256_per_sample` — last EMA ep20 / 300K (5.36M smp). Ready to evaluate now.
- All three are ready-to-evaluate quick wins; ~3h fresh-generation each via `--no-fast_inference`.

**Batch size + LR ablation** — 0/8 cells filled. The full block requires new training runs that don't exist for n=256 (no `pdb_lmdb_256_bs80`, no `lr3x` variants registered). Either:
- Mirror n=128's bs∈{24,80} × lr∈{1×,3×} × ±REPA grid (8 cells) — needs ~8 new training launches + ~16h GPU each to reach 200K, OR
- Drop the bs/lr block from n=256 and rely on n=128 for that ablation.
