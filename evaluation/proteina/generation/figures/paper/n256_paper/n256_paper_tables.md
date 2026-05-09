# n=256 paper-protocol sweep — ablation table

Companion to `n256_paper_sweep.png`. Pool: 1125 PDBs at L∈{50,75,100,125,150,175,200,225,250} × 125 for FID/fJSD/fS_T (N=1125); designability on 50/L × 5 paper lengths {50,100,150,200,250} (N=250); diversity/novelty on the designable subset.

**N per metric:** PDB FID=N=1125, AFDB FID=N=1125, Fold Score (Topo)=N=1125, PDB fJSD (Topo)=N=1125, Designability=N=250, scRMSD mean (Å)=N=250, Diversity (clusters)=designable, Diversity (pairwise TM)=designable.

Best per metric within each ablation block is **bolded**.

| Run | Step | bs | samples | des N | PDB FID ↓ | PDB fJSD C ↓ | PDB fJSD A ↓ | PDB fJSD T ↓ | AFDB FID ↓ | AFDB fJSD C ↓ | AFDB fJSD A ↓ | AFDB fJSD T ↓ | fS T ↑ | Des ↑ | scRMSD ↓ | Div clust total ↑ | Div pairwise TM ↓ | Notes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **External reference — (NVIDIA NGC public 60M ckpt; not part of any of our ablations)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Pretrained 60M (NGC) | 1.3M | — | — | 250 | 319.8 | 1.227 | 1.753 | 2.649 | 300.1 | 0.680 | 1.241 | 2.197 | 19.67 | 0.920 | 1.13 | 226 | 0.146 | NGC release `proteina_pretrained_dfs_60m`. **12-layer** vs our **10-layer** 60M (see [project_proteina_60m_layer_mismatch.md](../../../../../home/sr2173/.claude/projects/-home-sr2173-git-molecular-repa/memory/project_proteina_60m_layer_mismatch.md)) — compare via normalized depth, not absolute layer index. Centroid novelty not run for this row. |
| **Layer ablation — (L0/L4/L9 vs baseline, PDB, GearNet-CA, λ=0.5, per_residue, nominal bs=24)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Baseline ep21 | 400K | 24† | 5.74M | 250 | 441.8 | 0.421 | 0.480 | 2.510 | 462.0 | 0.267 | **0.474** | 2.567 | 30.08 | 0.124 | 6.31 | 30 | **0.156** | 🔁 **rerun candidate**: wandb confirms bs=12 → bs=24 at step ~322K (full run reached step 615K). Anchors 3 ablation blocks (layer/encoder/dataset). |
| REPA L0 ep26 | 400K | 24† | 7.08M | 250 | **265.7** | **0.014** | **0.421** | **1.682** | **318.5** | **0.164** | 0.664 | **1.873** | 34.33 | 0.084 | 7.12 | 16 | 0.192 | 🔁 **rerun candidate**: bs=12 → bs=24 at step ~210K (started 04-17, bumped 04-18). Two snaps preserved. |
| REPA L4 ep22 | 400K | 24† | 6.37M | 250 | 357.4 | 0.093 | 0.491 | 1.866 | 410.6 | 0.329 | 0.719 | 1.998 | **40.35** | 0.320 | 5.35 | 76 | 0.172 | 🔁 **rerun candidate**: bs=12 → bs=24 at step ~269K. Anchor for encoder/dataset/λ/averaging blocks — taints all four. |
| REPA L9 ep25 | 400K | 24† | 7.25M | 250 | 297.2 | 0.441 | 0.808 | 2.228 | 337.3 | 0.237 | 0.662 | 2.368 | 33.58 | **0.484** | **4.67** | **84** | 0.357 | 🔁 **rerun candidate**: bs=12 → bs=24 at step ~196K. |
| **Encoder ablation — (REPA L4 with 6 target encoders, PDB, λ=0.5, per_residue, nominal bs=24)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Baseline ep21 | 400K | 24† | 5.74M | 250 | 441.8 | 0.421 | 0.480 | 2.510 | 462.0 | **0.267** | **0.474** | 2.567 | 30.08 | 0.124 | 6.31 | 30 | 0.156 | 🔁 same as layer block (bumped 12→24 at step ~322K). |
| CA-GearNet ep22 | 400K | 24† | 6.37M | 250 | 357.4 | 0.093 | 0.491 | 1.866 | 410.6 | 0.329 | 0.719 | 1.998 | **40.35** | 0.320 | 5.35 | 76 | 0.172 | 🔁 = L4 ep22 (bumped 12→24). |
| GearNet random ep17 | 200K | 24⚠ | 1.44M | 250 | 600.2 | 0.103 | 0.362 | 2.681 | 673.5 | 0.297 | 0.623 | 2.834 | 38.42 | 0.160 | 6.44 | 40 | 0.172 | ⚠️ **avg bs=7.20** over the run (1.44M smp / 200K steps from ckpt `nsamples_processed`), NOT clean bs=24 as the start-date had implied. Likely length-bucketed sampling effect (REPA-specific overhead, smaller microbatch on longer chains). Sample budget is **3.3× smaller than the bs=24 baseline AFDB row** at the same step count — comparing this row vs CA-GearNet ep22 (6.37M) is sample-budget-unfair by 4.4×. |
| PW-Structure | — | 24 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | not trained yet — config exists, no run dir. |
| PW-Torsional | — | 24 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | not trained yet — config exists, no run dir. |
| ProteinMPNN ep26 | 300K | 24 | 7.20M | 250 | **261.2** | **0.056** | **0.334** | **1.642** | **307.7** | 0.310 | 0.751 | **1.918** | 31.80 | **0.412** | **3.94** | **89** | 0.193 | ✅ clean bs=24 (started 05-06, post-bump era); evaluated 2026-05-09 (slurm 29112166). **Best PDB/AFDB FID, fJSD, Designability, scRMSD, clusters in encoder block** — strongly outperforms CA-GearNet ep22 (FID 357.4→261.2; Des 0.320→0.412). Sample budget 7.20M (clean bs=24 × 300K) vs CA-GearNet 6.37M — modest favourable budget gap, but the gap is large enough that ProteinMPNN looks like a real win. |
| ESM2 (L9-t30) steplast | 322K | **12*** | 3.86M | 250 | 314.6 | 0.742 | 0.891 | 2.307 | 318.2 | 0.409 | 0.556 | 2.012 | 30.36 | 0.168 | 6.11 | 42 | **0.149** | ⚠️ **bs=12 throughout** (≠ rest of block at bs=24) — ESM-650M OOMs at bs=24 (smoke job 28898440), trained intentionally via `pdb_lmdb_256_bs12`. ⚠️ **L9-t30 ≠ L4 default** (the encoder block otherwise pins L4) — second axis of incomparability. Despite both handicaps the **PDB FID (314.6) beats CA-GearNet L4 ep22 (357.4)** — directional only, not a fair comparison. Evaluated 2026-05-08 (slurm 28993863). |
| **Step-matched reference (encoder ablation, ep17 anchor)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| REPA L9 ep17 | 300K | 24† | 4.85M | 250 | 376.0 | 0.247 | 0.508 | 2.471 | 415.9 | 0.368 | 0.681 | 2.534 | 33.87 | 0.040 | 9.02 | 8 | 0.134 | 🔁 = earlier ckpt of L9 ep25 (bumped 12→24); same rerun fixes both. |
| **Dataset ablation — (PDB vs AFDB-Swissprot, baseline + REPA L4, GearNet-CA, λ=0.5, per_residue, nominal bs=24)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Baseline PDB ep21 | 400K | 24† | 5.74M | 250 | 441.8 | 0.421 | **0.480** | 2.510 | 462.0 | **0.267** | **0.474** | 2.567 | 30.08 | 0.124 | 6.31 | 30 | **0.156** | 🔁 same as layer block (bumped 12→24 at step ~322K). |
| Baseline AFDB ep20 | 200K | 24 | 4.80M | 250 | 374.5 | 0.825 | 2.242 | 3.193 | 367.4 | 1.236 | 2.437 | 2.977 | 10.05 | 0.656 | **2.17** | **140** | 0.212 | ✅ clean bs=24 throughout (started 04-23). |
| REPA L4 PDB ep22 | 400K | 24† | 6.37M | 250 | 357.4 | **0.093** | 0.491 | **1.866** | 410.6 | 0.329 | 0.719 | **1.998** | **40.35** | 0.320 | 5.35 | 76 | 0.172 | 🔁 = L4 ep22 (bumped 12→24). |
| REPA L4 AFDB ep20 | 200K | 24⚠ | 2.53M | 250 | **207.5** | 0.141 | 0.837 | 2.129 | **223.6** | 0.512 | 1.274 | 2.288 | 21.08 | **0.684** | 2.91 | 102 | 0.229 | ⚠️ **avg bs=12.66** (2.53M / 200K from ckpt `nsamples_processed`), NOT clean bs=24. About **half the sample budget of Baseline AFDB** (4.80M, avg bs=24) at the same step. The headline FID/Des wins versus baseline AFDB are *despite* a 1.9× sample-budget disadvantage — strengthens the REPA-helps story but cross-method comparison is sample-unfair in REPA's favour. |
| **λ ablation — (REPA L4, PDB, GearNet-CA, per_residue, nominal bs=24, varying λ)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| λ=0.5 ep22 | 400K | 24† | 6.37M | 250 | **357.4** | 0.093 | **0.491** | **1.866** | **410.6** | 0.329 | 0.719 | **1.998** | 40.35 | **0.320** | **5.35** | **76** | **0.172** | 🔁 = L4 ep22 (bumped 12→24). |
| λ=1.0 ep26 | 300K | 24 | 7.20M | 250 | 449.1 | **0.036** | 0.556 | 2.004 | 501.8 | 0.249 | 0.961 | 2.269 | 28.94 | 0.192 | 6.42 | 41 | 0.290 | ✅ clean bs=24 (started 05-07, post-bump era); evaluated 2026-05-09 (slurm 29113620 task 1). Worse than λ=0.5 on most metrics despite +0.83M smp budget — λ=0.5 looks like the right setting for 60M-paper at n=256. |
| λ=2.0 ep17 | 200K | 24 | 4.80M | 250 | 398.1 | 0.065 | 0.492 | 1.949 | 463.8 | **0.215** | **0.667** | 2.258 | **43.83** | 0.032 | 9.94 | 7 | 0.181 | ✅ clean bs=24 (started 05-07, post-bump era); evaluated 2026-05-09 (slurm 29113620 task 0). **Best fS_T (43.83 > λ=0.5's 40.35)** but **collapse on Designability (0.032)** and **clusters (7 designable across only 2 length bins)** — strong λ over-weights REPA, hurting samplable quality even as topology score climbs. Sample budget 4.80M is 1.3× smaller than λ=0.5's 6.37M. |
| **Step-matched reference (λ ablation, L4 default at step 300K)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| λ=0.5 ep13 step300k | 300K | 24† | 3.97M | 250 | 497.8 | 0.222 | 0.689 | 2.259 | 557.4 | 0.674 | 1.134 | 2.560 | 38.99 | 0.080 | 8.59 | 20 | 0.159 | 🔁 same run as λ=0.5 ep22 (bumped 12→24 at ~269K), earlier ckpt. Step-matched to λ=1.0 ep26@300K. **Sample-budget caveat:** this row has 3.97M smp (bumped run), λ=1.0 has 7.20M (clean bs=24 × 300K) — step-matched but NOT sample-matched, λ=1.0 has 1.8× more samples. Evaluated 2026-05-09 (slurm 29114035). |
| **Averaging ablation — (REPA L0/L4/L9, per_sample vs per_residue, PDB, GearNet-CA, λ=0.5, nominal bs=24)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| L0 per_residue ep26 | 400K | 24† | 7.08M | 250 | **265.7** | **0.014** | 0.421 | **1.682** | **318.5** | **0.164** | 0.664 | **1.873** | 34.33 | 0.084 | 7.12 | 16 | 0.192 | 🔁 = L0 ep26 (bumped 12→24). |
| L0 per_sample steplast | 381K | 24† | 7.52M | 250 | 452.4 | 0.025 | **0.315** | 2.023 | 482.5 | 0.179 | **0.636** | 2.242 | 28.57 | 0.228 | 5.87 | 44 | 0.230 | 🔁 **rerun candidate**: bs=12 → bs=24 at step ~143K (full run reached step 382K). Per_residue ep26 better than this on FID/fJSD/fS_T but worse on Des; resolved live last-EMA at 381500 (≥ ep19/300K snapshot). Evaluated 2026-05-08 (slurm 28993862 task 0). |
| L4 per_residue ep22 | 400K | 24† | 6.37M | 250 | 357.4 | 0.093 | 0.491 | 1.866 | 410.6 | 0.329 | 0.719 | 1.998 | **40.35** | 0.320 | 5.35 | 76 | **0.172** | 🔁 = L4 ep22 (bumped 12→24). |
| L4 per_sample step400k | 400K | 24† | 7.25M | 250 | 276.7 | 0.092 | 0.586 | 1.832 | 330.8 | 0.327 | 0.886 | 2.082 | 33.59 | 0.280 | 5.77 | 58 | 0.176 | ✅ **rerun done**: explicit ep25/400K snapshot pin (sample-matched to per_residue ep22, ~6.69M smp). bs=12→24 mid-training. **Notably better PDB FID (276.7) than per_residue ep22 (357.4)** but loses on fS_T (33.6 vs 40.4) and Des (0.28 vs 0.32). Re-evaluated 2026-05-08 (slurm 29012689) — earlier attempt at last-EMA resolved to a stale 04-17 symlink at step=56K (dropped). |
| L9 per_residue ep25 | 400K | 24† | 7.25M | 250 | 297.2 | 0.441 | 0.808 | 2.228 | 337.3 | 0.237 | 0.662 | 2.368 | 33.58 | **0.484** | **4.67** | **84** | 0.357 | 🔁 = L9 ep25 (bumped 12→24). |
| L9 per_sample steplast | 385K | 24† | 7.56M | 250 | 499.1 | 0.322 | 0.959 | 2.415 | 569.0 | 0.201 | 1.002 | 2.717 | 30.47 | 0.328 | 5.23 | 70 | 0.211 | 🔁 **rerun candidate**: bs=12 → bs=24 at step ~145K (full run reached step 387K). Per_residue ep25 better on Des/scRMSD/clusters; resolved live last-EMA at 385000. Evaluated 2026-05-08 (slurm 28993862 task 2). |
| **Batch size + LR ablation — (bs ∈ {24,80}? × lr ∈ {1×,3×} × ±REPA)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (no n=256 bs/lr variants trained yet) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | empty block — see Pending Rows below. |
| _**Reference ranges — Proteina paper Table 1** (n=256 lengths 50-275 with γ-tuned guidance; sanity band only, NOT directly comparable)_ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| paper min–max | — | — | — | — | 129.9–933.9 | — | — | 0.680–3.690 | 159.9–855.4 | — | — | 0.910–3.100 | 9.720–30.11 | 0.220–0.990 | — | 64.00–323.0 | 0.350–0.450 | — |

*Reference-range source:* Proteina paper Table 1 (unconditional backbone generation), 17 method rows (FrameDiff, FoldFlow×3, FrameFlow, ESM3, Chroma, RFDiffusion, Proteus, Genie2, ℳ_FS×3, ℳ_FS^no-tri, ℳ_21M×2, ℳ_LoRA). Designability converted from % to rate (0-1). Diversity-clusters maps to the cluster *count* the paper reports (we use `clusters_total`). fJSD C/A levels and scRMSD have no Table 1 counterparts — left blank. **The paper uses n=256 over lengths 50–275 with γ-tuned guidance; we use n=256 over lengths 50–250 unconditional, so the band is a sanity check, not a target.** Notable: our fS_T peak at REPA-L4 PDB ep22 (40.35) sits **above the paper maximum** (30.11) — likely a protocol-pool difference (we use 1125 PDBs over 9 lengths) but worth verifying before claiming outperformance.

**Notes column legend.** ✅ = clean bs=24 throughout (single snap, post-2026-04-18 first ckpt). 🔁 = mid-run bs change (12→24 around 2026-04-18 SDPA-contiguous fix); rerun-from-scratch recommended for clean reporting. 🚫 = wrong bs throughout. † next to bs=24 marks rows whose underlying run was actually bs=12 for some prefix. ⚠ next to bs marks rows whose actual avg bs (from `nsamples_processed`) materially diverges from the nominal bs in the column header.

**Samples column.** All values are **ground truth** read from the EMA ckpt's `nsamples_processed` field (verified 2026-05-08). Pretrained NGC ckpt is external (no comparable counter).

**Bump-step reference.** Per-run bs=12→24 bump steps (and the analysis of how earlier estimates over-counted samples by 0.6–1.5M) are tabulated in [`n256_bump_steps.md`](n256_bump_steps.md). The bump step for each affected row is also woven into its Notes cell below.

## Pending rows (status as of 2026-05-09)

**Layer ablation** — complete (4/4). No pending rows.

**Encoder ablation** — 5/7 cells filled. Pending:
- `repa_l4_256_per_residue_pw_structure` — config exists ([training_repa_l4_256_per_residue_pw_structure.yaml](../../../../../src/proteina/configs/experiment_config/training/256/pw_gearnet/per_residue/training_repa_l4_256_per_residue_pw_structure.yaml)); no run dir on disk → needs training kickoff.
- `repa_l4_256_per_residue_pw_torsional` — same status, training kickoff needed.
- `repa_mpnn_l4_256_per_residue` — **filled** by ep26@300K (slurm 29112166, evaluated 2026-05-09); strong winner across the encoder block.
- `repa_esm_l4_256_per_residue` — does **not** exist. **Filled by `repa_esm_l9_t30_256_steplast`** (slurm 28993863, 2026-05-08) with double footnote (bs=12 + L9-t30 layer mismatch).
- MC-GearNet-Edge — explicitly skip per [project_encoder_characterizations.md](../../../../../home/sr2173/.claude/projects/-home-sr2173-git-molecular-repa/memory/project_encoder_characterizations.md): effective rank 1.1/3072 + norm explosion at n=512, expected to be unusable here too.

**Step-matched reference** — complete (1/1). `repa_l9_256_ep17` lives here as a step/sample-matched comparator for the random-init encoder row (both at ep17 / 200-300K).

**Dataset ablation** — complete (4/4). No pending rows.

**λ ablation** — **complete (3/3)** as of 2026-05-09. λ=0.5 ep22 (bumped run, 6.37M smp); λ=1.0 ep26@300K (slurm 29113620 task 1, clean bs=24, 7.20M smp); λ=2.0 ep17@200K (slurm 29113620 task 0, clean bs=24, 4.80M smp). Step-matched anchor `λ=0.5 ep13 step300k` added (slurm 29114035, 3.97M smp due to mid-run bump) so λ=1.0 ep26@300K can be compared at matched steps even though sample budgets differ. **Net read:** λ=0.5 wins most metrics; λ=2.0 wins fS_T but loses Designability dramatically (0.032) — λ-overweighting is real and harmful past 1.0.

**Averaging ablation** — **complete (6/6)** as of 2026-05-08. L0 per_sample @381K, L9 per_sample @385K (slurm 28993862 tasks 0/2, last-EMA), L4 per_sample @400K (slurm 29012689, explicit step pin after the original last-EMA resolved to a stale 04-17 symlink at step=56K). L4 per_sample is the only sample-matched row (ep25/400K = 6.69M smp); L0/L9 sit slightly past their per_residue sample budgets.

**Batch size + LR ablation** — 0/8 cells filled. The full block requires new training runs that don't exist for n=256 (no `pdb_lmdb_256_bs80`, no `lr3x` variants registered). Either:
- Mirror n=128's bs∈{24,80} × lr∈{1×,3×} × ±REPA grid (8 cells) — needs ~8 new training launches + ~16h GPU each to reach 200K, OR
- Drop the bs/lr block from n=256 and rely on n=128 for that ablation.
