# n=512 ablation table (paper-style scaffold)

> ⚠️ **Protocol divergence — read before comparing.** Unlike the n=128 / n=256 paper tables (1125 PDBs × 9 lengths under the Proteina paper protocol), **n=512 has no paper-protocol sweep on disk**. Rows below are sourced from two non-paper harnesses:
> - **`n512_sm_lite`** (`results/lite/n512_sm_lite/sweep_results.jsonl`) — 4 runs × 1 ckpt, smaller pool, partial designability (only baseline + L4 ran scRMSD; des_N=50 not 250).
> - **`n512_full_eval`** (`results/full_eval/n512_full_eval/inference_fid_60m_*.csv`) — single-row CSVs at one or two step pins per run, **FID + fJSD + fS only** (no Des, no scRMSD, no diversity, no novelty).
>
> Pool size, length grid, designability N, and diversity-pairwise-TM are all different from n=128/n=256. **Treat as a within-table comparison only — do NOT compare row-for-row to the n=128/n=256 tables.** Convergence trace (4 runs × ~12 step pins) lives at `results/lite/n512_convergence_lite/lite_convergence_all.csv`; not enumerated here.

Best per metric within each ablation block is **bolded**. Empty cells (`—`) are data we don't have for n=512.

| Run | Step | bs | samples | des N | PDB FID ↓ | PDB fJSD C ↓ | PDB fJSD A ↓ | PDB fJSD T ↓ | AFDB FID ↓ | AFDB fJSD C ↓ | AFDB fJSD A ↓ | AFDB fJSD T ↓ | fS T ↑ | Des ↑ | scRMSD ↓ | Div clust total ↑ | Div pairwise TM ↓ | Notes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **External reference — (NVIDIA NGC public 60M ckpt; not part of any of our ablations)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Pretrained 60M (NGC) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | ⚠️ no n=512 sweep run for the NGC pretrained ckpt; n=256 reference row is at `n256_paper_tables.md`. |
| **Layer ablation — (L0/L4/L9 vs baseline; sources mixed: `sm_lite` and `full_eval`)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Baseline (sm_lite) | 500K | — | — | 50 | 1317.84 | 2.880 | 3.344 | 6.202 | 1281.73 | 1.937 | 2.454 | 6.195 | 4.453 | 0.000 | 19.508 | — | — | ⚠️ source: `n512_sm_lite`; ckpt=`last-EMA.ckpt` of `proteina_60m_baseline_v2`, seed=42; div_clusters reads 200 but that is the pool cap (50 lengths × 4 = 200), not a real cluster count → suppressed. Novelty (lite-style) = 0.640. |
| Baseline (full_eval) | 742K | — | — | — | 647.97 | 1.288 | 1.567 | 3.794 | 622.05 | 0.655 | **0.982** | 3.717 | **19.366** | — | — | — | — | ⚠️ source: `n512_full_eval/inference_fid_60m_baseline.csv`; ckpt=`last-EMA.ckpt` @ step=742K (epoch=10), seed=5; FID/fJSD/fS only. |
| REPA L0 (sm_lite) | 750K | — | — | — | 943.11 | **0.323** | 1.382 | 3.866 | 897.37 | 0.391 | 1.552 | 3.988 | 7.633 | — | — | — | — | ⚠️ source: `n512_sm_lite`; ckpt=`chk_epoch=00000007_step=000000750000-EMA.ckpt` of `proteina_60m_repa_layer0_v2`, seed=42; designability/scRMSD not run for this row; novelty = 0.750 (highest among sm_lite L0/L4/L9). |
| REPA L0 last-EMA (full_eval) | 420K | — | — | — | 599.14 | 0.842 | 1.394 | 3.149 | 611.07 | 0.452 | 1.010 | 3.106 | 1.842 | — | — | — | — | ⚠️ source: `n512_full_eval/inference_fid_60m_repa_layer0.csv`; `last-EMA.ckpt` @ 420K (epoch=3), seed=5. |
| REPA L0 (full_eval, 840K) | 836500 | — | — | — | **401.81** | 0.533 | **1.035** | **2.536** | **393.84** | **0.435** | 1.031 | **2.622** | 1.927 | — | — | — | — | ⚠️ source: `n512_full_eval/inference_fid_60m_repa_layer0_840k.csv`; explicit step pin @ 836500 (epoch=7), seed=5. **Best PDB FID and AFDB FID in the layer block** (single-step-pin caveat). |
| REPA L4 (sm_lite) | 750K | — | — | 50 | 1108.93 | 0.730 | 1.962 | 4.395 | 1072.67 | 0.332 | 1.551 | 4.435 | 7.204 | 0.000 | **18.032** | — | — | ⚠️ source: `n512_sm_lite`; ckpt of `proteina_60m_repa_v2`, seed=42; novelty = 0.730. scRMSD 18.03 < baseline 19.51 (still no designable structures). |
| REPA L4 last-EMA (full_eval) | 420K | — | — | — | 656.96 | 1.075 | 1.570 | 3.445 | 659.72 | 0.570 | 1.144 | 3.465 | 15.780 | — | — | — | — | ⚠️ source: `n512_full_eval/inference_fid_60m_repa.csv`; `last-EMA.ckpt` @ 420K (epoch=3), seed=5. |
| REPA L4 (full_eval, 840K) | 840K | — | — | — | 580.06 | 0.896 | 1.362 | 2.934 | 569.77 | 0.427 | 0.879 | 2.881 | 1.823 | — | — | — | — | ⚠️ source: `n512_full_eval/inference_fid_60m_repa_840k.csv`; `last-EMA.ckpt` @ 840K (epoch=7), seed=5. |
| REPA L9 (sm_lite) | 750K | — | — | — | 964.44 | 0.444 | **1.343** | 4.213 | 935.74 | **0.277** | **1.200** | 4.525 | 10.197 | — | — | — | — | ⚠️ source: `n512_sm_lite`; ckpt of `proteina_60m_repa_layer9_v2`, seed=42; designability/scRMSD not run for this row; novelty = 0.605. |
| REPA L9 last-EMA (full_eval) | — | — | — | — | 879.18 | **0.094** | 1.152 | 3.951 | 887.14 | 0.222 | 1.098 | 4.147 | 14.664 | — | — | — | — | ⚠️ source: `n512_full_eval/inference_fid_60m_repa_layer9.csv`; `last-EMA.ckpt`, seed=5; **global_step / epoch missing in source CSV** (header columns shifted) — exact step unknown; treat as a single anchor, not a step-matched comparator. |
| REPA L9 (full_eval, 840K) | 847K | — | — | — | 614.24 | 0.237 | 0.988 | 2.791 | 676.97 | 0.328 | 1.053 | 3.234 | **28.590** | — | — | — | — | ⚠️ source: `n512_full_eval/inference_fid_60m_repa_layer9_840k.csv`; explicit step pin @ 847K (epoch=7), seed=5. **Best fS_T in the layer block.** |
| **Encoder ablation — (REPA L4 with 6 target encoders)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (no n=512 encoder variants trained) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | only CA-GearNet was run at n=512; ESM2 / PW-Structure / PW-Torsional / ProteinMPNN / GearNet-random / MC-GearNet-Edge are n=256-only. |
| **Step-matched reference** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (no sister random-init n=512 run) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | the n=256 step-matched anchor (`repa_l9_256_ep17` ↔ `gearnet_random_ep17`) has no n=512 counterpart. |
| **Dataset ablation — (PDB vs AFDB-Swissprot)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (no n=512 AFDB run) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | AFDB baseline + REPA L4 only exist at n=256. |
| **λ ablation — (REPA L4, varying λ)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (no n=512 λ variants) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | λ ∈ {1.0, 2.0} variants exist only at n=128 / n=256. |
| **Averaging ablation — (per_residue vs per_sample)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (no n=512 per_sample variants) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | per_sample runs exist only at n=256 (L0/L4/L9). |
| **Batch size + LR ablation — (bs ∈ {24,80} × lr ∈ {1×,3×} × ±REPA)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (no n=512 bs/lr variants) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | bs/lr block is n=128-only. |

**Notes column legend.** ⚠️ marks every n=512 row because **no row replicates the paper protocol** (1125 PDBs × 9 lengths). The other glyphs (✅ clean bs / 🔁 mid-run bs change / 🚫 wrong bs throughout) used in the n=128/n=256 tables are not applied here — bs/sample-count metadata for the n=512 runs was not captured in the lite/full_eval result rows.

**bs / samples / des N columns.** Left blank for every row. The lite/full_eval result rows do not log effective batch size; the convergence trace at `results/lite/n512_convergence_lite/lite_convergence_all.csv` records `batch_size=6` (baseline) and `batch_size=4` (REPA L0/L4/L9), which matches yaml-declared bs but has not been wandb-verified per [feedback_bs_audit_method.md](../../../../../home/sr2173/.claude/projects/-home-sr2173-git-molecular-repa/memory/feedback_bs_audit_method.md). des N is 50 for `sm_lite` rows where designability ran (baseline + L4), blank elsewhere. Diversity-pairwise-TM is blank everywhere — neither n=512 harness emits it.

**Source-file map (n=512).**
- `results/lite/n512_sm_lite/sweep_results.jsonl` — 4 rows, seed=42, single ckpt per run.
- `results/full_eval/n512_full_eval/inference_fid_60m_baseline.csv` — baseline @ 742K.
- `results/full_eval/n512_full_eval/inference_fid_60m_repa.csv` — L4 last-EMA @ 420K.
- `results/full_eval/n512_full_eval/inference_fid_60m_repa_840k.csv` — L4 @ 840K.
- `results/full_eval/n512_full_eval/inference_fid_60m_repa_layer0.csv` — L0 last-EMA @ 420K.
- `results/full_eval/n512_full_eval/inference_fid_60m_repa_layer0_840k.csv` — L0 @ 836500.
- `results/full_eval/n512_full_eval/inference_fid_60m_repa_layer9.csv` — L9 last-EMA @ unknown step (CSV header shift).
- `results/full_eval/n512_full_eval/inference_fid_60m_repa_layer9_840k.csv` — L9 @ 847K.
- `results/lite/n512_convergence_lite/lite_convergence_all.csv` — 48-row convergence trace (Baseline, L0, L4, L9 across step ∈ {10K…840K}); not folded into rows above.

## Pending rows / how to fill the empty blocks

To turn this into a real n=512 paper table (matching n=128/n=256), each of the empty blocks would need:
- **Encoder block** — 6 new training runs at n=512 (ESM2, PW-Structure, PW-Torsional, ProteinMPNN, GearNet-random, MC-GearNet-Edge) with paper-protocol eval. Currently zero exist on disk.
- **Step-matched reference** — pair `repa_l9_512` with a `gearnet_random_512` of equal step (no random-init 512 run trained).
- **Dataset block** — `baseline_512_afdb` + `repa_l4_512_afdb` (no AFDB 512 configs registered).
- **λ block** — λ=1.0 and λ=2.0 variants (only λ=0.5 default exists).
- **Averaging block** — L0/L4/L9 per_sample (only per_residue exists at 512).
- **bs/lr block** — bs ∈ {24,80} × lr ∈ {1×,3×}; would mirror the n=128 grid (8 cells).
- **Paper-protocol eval** — even for the rows we already have, replicate them under the 1125-PDB × 9-length protocol so the numbers can be cross-compared with the n=128/n=256 tables. The current `sm_lite` and `full_eval` numbers will not be apples-to-apples once the paper-protocol n=512 sweep exists.
