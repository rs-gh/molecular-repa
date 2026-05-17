# Proteina Ablation Checkpoints

Inventory of the runs grouped into the ablations we are tracking for the final stretch of the project. One row per wandb run; rows ordered with the **baseline first** within each ablation block. The header row that opens each block matches the wandb group label in the screenshots from 2026-05-16.

Conventions:

- **WandB run** is the live run ID under [`sr2173-university-of-cambridge/proteina-repa`](https://wandb.ai/sr2173-university-of-cambridge/proteina-repa). The `MMDD-HHMM-` date prefix in the wandb UI is the launch timestamp — the underlying `run_name_` (and the checkpoint store dir) drops it.
- **Store dir** is under `/rds/user/sr2173/hpc-work/proteina/store/<dir>/checkpoints/`.
- **Step cadence** lists every periodic step (`chk_epoch=…_step=…[-EMA].ckpt`) we have on disk. All runs in these ablations save every **100k optimizer steps**, so the column is shown as `100k–Xk @100k` where `X` is the highest step.
- **Last-EMA step** is the `global_step` stored inside `last-EMA.ckpt` (Lightning's "save_last" EMA snapshot, refreshed every checkpoint event). It is independent of the periodic-save cadence: it may sit *between* two periodic ticks (most common), *above* the highest periodic step (run kept training past the last 100k save), or — marked with **†** — *below* the highest periodic step. The † case appears when a wandb rename forked the run mid-training: the new session overwrote `last-EMA.ckpt` while the older periodic ckpts from before the fork were retained. For those rows, treat the periodic ckpts as the source of truth and the † last-EMA value as "current live session offset since the fork." See the `_part1` + `_part2` averaging block for the canonical example.
- **N** is `lmdb_max_num_residues` (128 / 256). **DS** is dataset (PDB / AFDB-SwissProt). **GPUs** is per-job device count.
- **bs** is per-GPU batch size; **eff bs** = bs × GPUs × grad-accum. `†` on a bs cell flags grad-accum > 1 (footnote in that block lists the value). Audited 2026-05-16 from each run's `data_config.json` (`batch_size`) + `exp_config.json` (`ngpus_per_node_`, `accumulate_grad_batches`).
- **Enc** is the REPA target encoder (`—` for baselines): `gn` = CA-GearNet (default), `esm` = ESM-2 base (t12), `esm-t30` = ESM-2 t30 medium, `mpnn` = ProteinMPNN, `pw-s/pw-t` = pairwise-GearNet structure / torsional, `rand-gn` = randomly-initialised CA-GearNet. **L** is the transformer layer the REPA head is tapped from. **Avg** is REPA averaging (`per_residue` / `per_sample`).
- **λ**, **wd**, **lr** are shown only when they differ from defaults (λ=0.5, wd=0, lr=1×base).

Inventory generated 2026-05-16; n=128 blocks refreshed 2026-05-17 (several bs80 runs trained well past the 2026-05-16 snapshot, and two new MPNN-L9 runs landed on disk: `proteina_60m_repa_mpnn_l9_128_per_residue_bs80_2gpu` on PDB and `proteina_60m_repa_mpnn_l9_128_afdb_per_residue_bs80_2gpu` on AFDB). Periodic-step list comes from `ls /rds/.../store/*/checkpoints/`; `Last-EMA step` is read from each `last-EMA.ckpt`'s `global_step` field via `torch.load(..., mmap=True, weights_only=False)`. Refresh snippet at the bottom of this file.

**Common cross-model comparison steps** (intersection of all non-crashed runs in this doc): **100k, 200k**. Most ablations also share 300k–400k; AFDB and the long-running PDB-256 runs go past 1M. Each ablation block notes its own common-step window.

---

## Ablation: n=128 PDB bs ablation

Common cross-model steps: **100k, 200k**. bs12 caps at 300k; the bs24 REPA-L4 leg reaches 400k; `baseline_128_bs80` now reaches 700k and `baseline_128` reaches 1000k.

| WandB run | Store dir | N | DS | GPUs | bs | eff bs | Enc | L | Avg | Step cadence | Last-EMA step |
|---|---|---|---|---:|---:|---:|---|---|---|---|---:|
| `proteina_60m_baseline_128` *(bs_24 in wandb label)* | `proteina_60m_baseline_128` | 128 | PDB | 1 | 24 | 24 | — | — | — | 100k–1000k @100k | 1,004,500 |
| `proteina_60m_baseline_128_bs80` | `proteina_60m_baseline_128_bs80` | 128 | PDB | 2 | 40 | 80 | — | — | — | 100k–700k @100k | 714,000 |
| `proteina_60m_repa_l4_128_per_residue_bs12` | `proteina_60m_repa_l4_128_per_residue_bs12` | 128 | PDB | 1 | 12 | 12 | gn | 4 | per_residue | 100k–300k @100k | 315,000 |
| `proteina_60m_repa_l4_128_per_residue_bs24` | `proteina_60m_repa_l4_128_per_residue_bs24` | 128 | PDB | 1 | 24 | 24 | gn | 4 | per_residue | 100k–400k @100k | 472,500 |
| `proteina_60m_repa_l4_128_per_residue_bs80` | `proteina_60m_repa_l4_128_per_residue_bs80` | 128 | PDB | 2 | 40 | 80 | gn | 4 | per_residue | 100k–500k @100k | 514,500 |

---

## Ablation: n=128 PDB L4 REPA weight-decay + λ + lr ablation

Common cross-model steps: **100k, 200k**. (lr3x only reached 100k periodic; last-EMA at 161k.)

| WandB run | Store dir | N | DS | GPUs | bs | eff bs | Enc | L | Avg | λ | wd | lr | Step cadence | Last-EMA step |
|---|---|---|---|---:|---:|---:|---|---|---|---:|---:|---|---|---:|
| `proteina_60m_baseline_128_bs80` | `proteina_60m_baseline_128_bs80` | 128 | PDB | 2 | 40 | 80 | — | — | — | — | 0 | 1× | 100k–700k @100k | 714,000 |
| `proteina_60m_repa_l4_128_per_residue_bs80` *(λ=0.5 reference)* | `proteina_60m_repa_l4_128_per_residue_bs80` | 128 | PDB | 2 | 40 | 80 | gn | 4 | per_residue | 0.5 | 0 | 1× | 100k–500k @100k | 514,500 |
| `proteina_60m_repa_l4_128_per_residue_bs80_lambda025` | `proteina_60m_repa_l4_128_per_residue_bs80_lambda025` | 128 | PDB | 1 | 80 | 80 | gn | 4 | per_residue | 0.25 | 0 | 1× | 100k–200k @100k | 234,500 |
| `proteina_60m_repa_l4_128_per_residue_bs80_lambda1` | `proteina_60m_repa_l4_128_per_residue_bs80_lambda1` | 128 | PDB | 1 | 80 | 80 | gn | 4 | per_residue | 1.0 | 0 | 1× | 100k–200k @100k | 245,000 |
| `proteina_60m_repa_l4_128_per_residue_bs80_lambda2` | `proteina_60m_repa_l4_128_per_residue_bs80_lambda2` | 128 | PDB | 1 | 80 | 80 | gn | 4 | per_residue | 2.0 | 0 | 1× | 100k–200k @100k | 259,000 |
| `proteina_60m_repa_l4_128_per_residue_bs80_wd1e-2` | `proteina_60m_repa_l4_128_per_residue_bs80_wd1e-2` | 128 | PDB | 1 | 80 | 80 | gn | 4 | per_residue | 0.5 | 1e-2 | 1× | 100k–200k @100k | 245,000 |
| `proteina_60m_repa_l4_128_per_residue_bs80_lr3x` | `proteina_60m_repa_l4_128_per_residue_bs80_lr3x` | 128 | PDB | 1 | 80 | 80 | gn | 4 | per_residue | 0.5 | 0 | 3× | 100k only | 161,000 |

---

## Ablation: n=128 PDB REPA averaging ablation (all crashed early)

All 13 runs in this group show `Crashed` in wandb (pre-bs80 sweep era). Many ESM variants never produced a periodic checkpoint — their store dirs hold only config JSON. The L4 baselines reused here are repeated from earlier blocks.

Common cross-model steps (over runs with ckpts): **100k, 200k**.

| WandB run | Store dir | N | DS | GPUs | bs | eff bs | Enc | L | Avg | Step cadence | Last-EMA step |
|---|---|---|---|---:|---:|---:|---|---|---|---|---:|
| `proteina_60m_baseline_128_bs80` | `proteina_60m_baseline_128_bs80` | 128 | PDB | 2 | 40 | 80 | — | — | — | 100k–700k @100k | 714,000 |
| `proteina_60m_repa_l0_128_per_residue_bs80` | `proteina_60m_repa_l0_128_per_residue_bs80` | 128 | PDB | 1 | 80 | 80 | gn | 0 | per_residue | 100k–200k @100k | 231,000 |
| `proteina_60m_repa_l4_128_per_residue_bs80` | `proteina_60m_repa_l4_128_per_residue_bs80` | 128 | PDB | 2 | 40 | 80 | gn | 4 | per_residue | 100k–500k @100k | 514,500 |
| `proteina_60m_repa_l9_128_per_residue_bs80` | `proteina_60m_repa_l9_128_per_residue_bs80` | 128 | PDB | 2 | 40 | 80 | gn | 9 | per_residue | 100k–400k @100k | 483,000 |
| `proteina_60m_repa_l0_128_per_sample` | `proteina_60m_repa_l0_128_per_sample` | 128 | PDB | 1 | 80 | 80 | gn | 0 | per_sample | 100k–200k @100k | 217,000 |
| `proteina_60m_repa_l4_128_per_sample` | `proteina_60m_repa_l4_128_per_sample` | 128 | PDB | 1 | 80 | 80 | gn | 4 | per_sample | 100k–200k @100k | 217,000 |
| `proteina_60m_repa_l9_128_per_sample` | `proteina_60m_repa_l9_128_per_sample` | 128 | PDB | 1 | 80 | 80 | gn | 9 | per_sample | 100k–200k @100k | 217,000 |
| `proteina_60m_repa_esm_l0_128_per_residue` | *(no store dir)* | 128 | PDB | ? | ? | ? | esm | 0 | per_residue | — | — |
| `proteina_60m_repa_esm_l4_128_per_residue` | *(no store dir)* | 128 | PDB | ? | ? | ? | esm | 4 | per_residue | — | — |
| `proteina_60m_repa_esm_l9_128_per_residue` | *(no store dir)* | 128 | PDB | ? | ? | ? | esm | 9 | per_residue | — | — |
| `proteina_60m_repa_esm_l0_128_per_sample` | *(no store dir)* | 128 | PDB | ? | ? | ? | esm | 0 | per_sample | — | — |
| `proteina_60m_repa_esm_l4_128_per_sample` | *(no store dir)* | 128 | PDB | ? | ? | ? | esm | 4 | per_sample | — | — |
| `proteina_60m_repa_esm_l9_128_per_sample` | *(no store dir)* | 128 | PDB | ? | ? | ? | esm | 9 | per_sample | — | — |

---

## Ablation: n=128 PDB L4 REPA encoder ablation

Common cross-model steps (over runs with ckpts): **100k**. (pw-structure / pw-torsional only reached 100k periodic; ESM-t30 still has no periodic save.)

| WandB run | Store dir | N | DS | GPUs | bs | eff bs | Enc | L | Avg | Step cadence | Last-EMA step |
|---|---|---|---|---:|---:|---:|---|---|---|---|---:|
| `proteina_60m_baseline_128_bs80` | `proteina_60m_baseline_128_bs80` | 128 | PDB | 2 | 40 | 80 | — | 4 | — | 100k–700k @100k | 714,000 |
| `proteina_60m_repa_l4_128_per_residue_bs80` *(CA-GearNet, default)* | `proteina_60m_repa_l4_128_per_residue_bs80` | 128 | PDB | 2 | 40 | 80 | gn | 4 | per_residue | 100k–500k @100k | 514,500 |
| `proteina_60m_repa_l4_128_per_residue_random` *(random-init CA-GearNet)* | `proteina_60m_repa_l4_128_per_residue_random` | 128 | PDB | 2 | 40 | 80 | rand-gn | 4 | per_residue | 100k–400k @100k | 490,000 |
| `proteina_60m_repa_mpnn_l4_128_per_residue_bs80` | `proteina_60m_repa_mpnn_l4_128_per_residue_bs80` | 128 | PDB | 2 | 40 | 80 | mpnn | 4 | per_residue | 100k–200k @100k | 231,000 |
| `proteina_60m_repa_esm_l4_128_per_residue` | *(no store dir — config only)* | 128 | PDB | ? | ? | ? | esm | 4 | per_residue | — | — |
| `proteina_60m_repa_esm_l4_t30_128_per_residue` | `proteina_60m_repa_esm_l4_t30_128_per_residue` | 128 | PDB | 1 | 80 | 80 | esm-t30 | 4 | per_residue | — *(config only)* | — |
| `proteina_60m_repa_l4_128_per_residue_pw_structure` | `proteina_60m_repa_l4_128_per_residue_pw_structure` | 128 | PDB | 1 | 80 | 80 | pw-s | 4 | per_residue | 100k only | 98,000 |
| `proteina_60m_repa_l4_128_per_residue_pw_torsional` | `proteina_60m_repa_l4_128_per_residue_pw_torsional` | 128 | PDB | 1 | 80 | 80 | pw-t | 4 | per_residue | 100k only | 108,500 |

---

## Ablation: n=128 PDB L9 REPA encoder ablation

Common cross-model steps (over runs with ckpts): **100k, 200k**. (`mpnn_l9_bs80_2gpu` reaches 500k; CA-GearNet L9 reaches 400k.)

| WandB run | Store dir | N | DS | GPUs | bs | eff bs | Enc | L | Avg | Step cadence | Last-EMA step |
|---|---|---|---|---:|---:|---:|---|---|---|---|---:|
| `proteina_60m_baseline_128_bs80` | `proteina_60m_baseline_128_bs80` | 128 | PDB | 2 | 40 | 80 | — | 9 | — | 100k–700k @100k | 714,000 |
| `proteina_60m_repa_l9_128_per_residue_bs80` *(CA-GearNet, default)* | `proteina_60m_repa_l9_128_per_residue_bs80` | 128 | PDB | 2 | 40 | 80 | gn | 9 | per_residue | 100k–400k @100k | 483,000 |
| `proteina_60m_repa_mpnn_l9_128_per_residue_bs80_2gpu` | `proteina_60m_repa_mpnn_l9_128_per_residue_bs80_2gpu` | 128 | PDB | 2 | 40 | 80 | mpnn | 9 | per_residue | 100k–500k @100k | 518,000 |
| `proteina_60m_repa_esm_l9_128_per_residue` | *(no store dir — config only)* | 128 | PDB | ? | ? | ? | esm | 9 | per_residue | — | — |

---

## Ablation: n=128 PDB REPA bs80 layer ablation

Common cross-model steps: **100k, 200k**.

| WandB run | Store dir | N | DS | GPUs | bs | eff bs | Enc | L | Avg | Step cadence | Last-EMA step |
|---|---|---|---|---:|---:|---:|---|---|---|---|---:|
| `proteina_60m_baseline_128_bs80` | `proteina_60m_baseline_128_bs80` | 128 | PDB | 2 | 40 | 80 | — | — | — | 100k–700k @100k | 714,000 |
| `proteina_60m_repa_l0_128_per_residue_bs80` | `proteina_60m_repa_l0_128_per_residue_bs80` | 128 | PDB | 1 | 80 | 80 | gn | 0 | per_residue | 100k–200k @100k | 231,000 |
| `proteina_60m_repa_l4_128_per_residue_bs80` | `proteina_60m_repa_l4_128_per_residue_bs80` | 128 | PDB | 2 | 40 | 80 | gn | 4 | per_residue | 100k–500k @100k | 514,500 |
| `proteina_60m_repa_l9_128_per_residue_bs80` | `proteina_60m_repa_l9_128_per_residue_bs80` | 128 | PDB | 2 | 40 | 80 | gn | 9 | per_residue | 100k–400k @100k | 483,000 |

---

## Ablation: n=128 AFDB encoder ablation

Common cross-model steps: **100k → 400k @100k** (set by the newest run, `mpnn_l9_afdb`, which reaches 400k; `l4_afdb` GearNet reaches 600k, baseline and MPNN-L4 run past 1M).

| WandB run | Store dir | N | DS | GPUs | bs | eff bs | Enc | L | Avg | Step cadence | Last-EMA step |
|---|---|---|---|---:|---:|---:|---|---|---|---|---:|
| `proteina_60m_baseline_afdb_128_bs80_2gpu` | `proteina_60m_baseline_afdb_128_bs80_2gpu` | 128 | AFDB | 2 | 40 | 80 | — | — | — | 100k–1200k @100k | 1,200,500 |
| `proteina_60m_repa_l4_128_afdb_per_residue_bs80_2gpu` *(CA-GearNet)* | `proteina_60m_repa_l4_128_afdb_per_residue_bs80_2gpu` | 128 | AFDB | 2 | 40 | 80 | gn | 4 | per_residue | 100k–600k @100k | 630,000 |
| `proteina_60m_repa_mpnn_l4_128_afdb_per_residue_bs80_2gpu` | `proteina_60m_repa_mpnn_l4_128_afdb_per_residue_bs80_2gpu` | 128 | AFDB | 2 | 40 | 80 | mpnn | 4 | per_residue | 100k–1100k @100k | 1,158,500 |
| `proteina_60m_repa_mpnn_l9_128_afdb_per_residue_bs80_2gpu` | `proteina_60m_repa_mpnn_l9_128_afdb_per_residue_bs80_2gpu` | 128 | AFDB | 2 | 40 | 80 | mpnn | 9 | per_residue | 100k–400k @100k | 490,000 |

---

## Ablation: n=256 PDB REPA encoder + layer + bs ablation

Common cross-model steps: **100k, 200k, 300k** (limited by `bs80_2gpu` at 200k, `esm-t30` at 300k; `mpnn_l9` is too young for a periodic ckpt but is live at 31k).

| WandB run | Store dir | N | DS | GPUs | bs | eff bs | Enc | L | Avg | Step cadence | Last-EMA step |
|---|---|---|---|---:|---:|---:|---|---|---|---|---:|
| `proteina_60m_baseline_256_bs24_2gpu` | `proteina_60m_baseline_256_bs24_2gpu` | 256 | PDB | 2 | 12 | 24 | — | — | — | 100k–1500k @100k | 1,578,500 |
| `proteina_60m_repa_l4_256_per_residue_bs24_2gpu` *(CA-GearNet, bs24)* | `proteina_60m_repa_l4_256_per_residue_bs24_2gpu` | 256 | PDB | 2 | 12 | 24 | gn | 4 | per_residue | 100k–900k @100k | 955,500 |
| `proteina_60m_repa_l4_256_per_residue_bs80_2gpu` *(CA-GearNet, bs80)* | `proteina_60m_repa_l4_256_per_residue_bs80_2gpu` | 256 | PDB | 2 | 10† | 80 | gn | 4 | per_residue | 100k–200k @100k | 252,000 |
| `proteina_60m_repa_l4_256_per_residue_random_bs24_2gpu` | `proteina_60m_repa_l4_256_per_residue_random_bs24_2gpu` | 256 | PDB | 2 | 12 | 24 | rand-gn | 4 | per_residue | 100k–600k @100k | 693,000 |
| `proteina_60m_repa_l9_256_per_residue_bs24_2gpu` | `proteina_60m_repa_l9_256_per_residue_bs24_2gpu` | 256 | PDB | 2 | 12 | 24 | gn | 9 | per_residue | 100k–900k @100k | 966,000 |
| `proteina_60m_repa_mpnn_l4_256_per_residue` | `proteina_60m_repa_mpnn_l4_256_per_residue` | 256 | PDB | 2 | 12 | 24 | mpnn | 4 | per_residue | 100k–1600k @100k | 1,613,500 |
| `proteina_60m_repa_mpnn_l9_256_per_residue` | `proteina_60m_repa_mpnn_l9_256_per_residue` | 256 | PDB | 2 | 12 | 24 | mpnn | 9 | per_residue | — *(no periodic ckpt yet)* | 31,500 |
| `proteina_60m_repa_esm_l9_t30_256_per_residue` | `proteina_60m_repa_esm_l9_t30_256_per_residue` | 256 | PDB | 1 | 12 | 12 | esm-t30 | 9 | per_residue | 100k–300k @100k | 322,000 |

† `bs80_2gpu` runs use per-GPU bs=10 × 2 GPU × grad-accum=4 = eff bs=80.

---

## Ablation: n=256 PDB L4 GearNet REPA weight-decay + λ + bs ablation

Common cross-model steps: **100k, 200k, 300k** (limited by `bs80_2gpu` at 200k and `lambda025` at 300k).

| WandB run | Store dir | N | DS | GPUs | bs | eff bs | Enc | L | Avg | λ | wd | Step cadence | Last-EMA step |
|---|---|---|---|---:|---:|---:|---|---|---|---:|---:|---|---:|
| `proteina_60m_baseline_256_bs24_2gpu` | `proteina_60m_baseline_256_bs24_2gpu` | 256 | PDB | 2 | 12 | 24 | — | — | — | — | 0 | 100k–1500k @100k | 1,578,500 |
| `proteina_60m_repa_l4_256_per_residue_bs24_2gpu` *(λ=0.5 reference)* | `proteina_60m_repa_l4_256_per_residue_bs24_2gpu` | 256 | PDB | 2 | 12 | 24 | gn | 4 | per_residue | 0.5 | 0 | 100k–900k @100k | 955,500 |
| `proteina_60m_repa_l4_256_per_residue_bs80_2gpu` | `proteina_60m_repa_l4_256_per_residue_bs80_2gpu` | 256 | PDB | 2 | 10† | 80 | gn | 4 | per_residue | 0.5 | 0 | 100k–200k @100k | 252,000 |
| `proteina_60m_repa_l4_256_per_residue_lambda025` | `proteina_60m_repa_l4_256_per_residue_lambda025` | 256 | PDB | 1 | 24 | 24 | gn | 4 | per_residue | 0.25 | 0 | 100k–300k @100k | 343,000 |
| `proteina_60m_repa_l4_256_per_residue_lambda1` | `proteina_60m_repa_l4_256_per_residue_lambda1` | 256 | PDB | 1 | 24 | 24 | gn | 4 | per_residue | 1.0 | 0 | 100k–500k @100k | 514,500 |
| `proteina_60m_repa_l4_256_per_residue_lambda2` | `proteina_60m_repa_l4_256_per_residue_lambda2` | 256 | PDB | 1 | 24 | 24 | gn | 4 | per_residue | 2.0 | 0 | 100k–400k @100k | 437,500 |
| `proteina_60m_repa_l4_256_per_residue_random_bs24_2gpu` | `proteina_60m_repa_l4_256_per_residue_random_bs24_2gpu` | 256 | PDB | 2 | 12 | 24 | rand-gn | 4 | per_residue | 0.5 | 0 | 100k–600k @100k | 693,000 |

† `bs80_2gpu` row uses per-GPU bs=10 × 2 GPU × grad-accum=4 = eff bs=80.

---

## Ablation: n=256 AFDB REPA encoder + layer + bs ablation

Common cross-model steps: **100k, 200k, 300k** (limited by `mpnn_l4_bs80_2gpu` at 300k).

| WandB run | Store dir | N | DS | GPUs | bs | eff bs | Enc | L | Avg | Step cadence | Last-EMA step |
|---|---|---|---|---:|---:|---:|---|---|---|---|---:|
| `proteina_60m_baseline_afdb_swissprot_256` | `proteina_60m_baseline_afdb_swissprot_256` | 256 | AFDB | 2 | 12 | 24 | — | — | — | 100k–1600k @100k | 1,624,000 |
| `proteina_60m_repa_l4_256_afdb_per_residue` *(CA-GearNet)* | `proteina_60m_repa_l4_256_afdb_per_residue` | 256 | AFDB | 2 | 12 | 24 | gn | 4 | per_residue | 100k–1200k @100k | 1,200,500 |
| `proteina_60m_repa_l9_256_afdb_per_residue` | `proteina_60m_repa_l9_256_afdb_per_residue` | 256 | AFDB | 2 | 12 | 24 | gn | 9 | per_residue | 100k–400k @100k | 423,500 |
| `proteina_60m_repa_mpnn_l4_256_afdb_per_residue` | `proteina_60m_repa_mpnn_l4_256_afdb_per_residue` | 256 | AFDB | 2 | 12 | 24 | mpnn | 4 | per_residue | 100k–1100k @100k | 1,144,500 |
| `proteina_60m_repa_mpnn_l4_256_afdb_per_residue_bs80_2gpu` | `proteina_60m_repa_mpnn_l4_256_afdb_per_residue_bs80_2gpu` | 256 | AFDB | 2 | 10† | 80 | mpnn | 4 | per_residue | 100k–300k @100k | 385,000 |
| `proteina_60m_repa_mpnn_l9_256_afdb_per_residue` | `proteina_60m_repa_mpnn_l9_256_afdb_per_residue` | 256 | AFDB | 2 | 12 | 24 | mpnn | 9 | per_residue | 100k–1500k @100k | 1,543,500 |

† `bs80_2gpu` row uses per-GPU bs=10 × 2 GPU × grad-accum=4 = eff bs=80. The rest of the AFDB-256 cohort is 2-GPU × bs=12 (eff 24).

---

## Ablation: n=256 PDB averaging ablation

Pre-2026-04-17 launch dates carry `_part1`/`_part2` because the rename forked the wandb run mid-training (see [proteina_training_runs.md](proteina_training_runs.md#2026-04-17-rename-history)). Checkpoint continuity is preserved through the physical store-dir rename, so the `_part1` and `_part2` rows of the same logical model share one checkpoint directory. The **†** in the Last-EMA column flags rows where `last-EMA.ckpt` was overwritten by the post-rename session and therefore sits *below* the highest pre-rename periodic step — trust the periodic ckpts for those rows.

Common cross-model steps: **100k, 200k, 300k**.

| WandB run (part1 + part2) | Store dir | N | DS | GPUs | bs | eff bs | Enc | L | Avg | Step cadence | Last-EMA step |
|---|---|---|---|---:|---:|---:|---|---|---|---|---:|
| *(no baseline launched for this sweep — compare against `proteina_60m_baseline_256` / `_bs24_2gpu` above)* | — | 256 | PDB | — | — | — | — | — | — | — | — |
| `proteina_60m_repa_l0_256_per_residue_part1` + `_part2` | `proteina_60m_repa_l0_256_per_residue` | 256 | PDB | 1 | 24 | 24 | gn | 0 | per_residue | 100k–400k @100k | 70,000 † |
| `proteina_60m_repa_l4_256_per_residue_part2` | `proteina_60m_repa_l4_256_per_residue` | 256 | PDB | 1 | 24 | 24 | gn | 4 | per_residue | 100k–500k @100k | 122,500 † |
| `proteina_60m_repa_l9_256_per_residue_part1` + `_part2` | `proteina_60m_repa_l9_256_per_residue` | 256 | PDB | 1 | 12 | 12 | gn | 9 | per_residue | 100k–400k @100k | 59,500 † |
| `proteina_60m_repa_l0_256_per_sample` | `proteina_60m_repa_l0_256_per_sample` | 256 | PDB | 1 | 24 | 24 | gn | 0 | per_sample | 100k–300k @100k | 381,500 |
| `proteina_60m_repa_l4_256_per_sample_part1` + `_part2` | `proteina_60m_repa_l4_256_per_sample` | 256 | PDB | 1 | 24 | 24 | gn | 4 | per_sample | 100k–400k @100k | 56,000 † |
| `proteina_60m_repa_l9_256_per_sample` | `proteina_60m_repa_l9_256_per_sample` | 256 | PDB | 1 | 24 | 24 | gn | 9 | per_sample | 100k–300k @100k | 385,000 |

---

## How to refresh this table

Periodic-step list:

```bash
cd /rds/user/sr2173/hpc-work/proteina/store/
for d in proteina_60m_*; do
  if [ -d "$d/checkpoints" ]; then
    steps=$(ls "$d/checkpoints/" | grep -oE 'step=[0-9]+' | sort -u | sed 's/step=0*//')
    last=$(echo "$steps" | tail -1)
    printf "%-70s last_periodic=%-10s steps=%s\n" "$d" "${last:-—}" "$(echo $steps | tr ' ' ',')"
  fi
done
```

`Last-EMA step` (`global_step` inside `last-EMA.ckpt`):

```bash
cd /rds/user/sr2173/hpc-work/proteina/store/
source /home/sr2173/git/molecular-repa/.venv/bin/activate
python3 - <<'PY'
import torch, glob
for c in sorted(glob.glob("proteina_60m_*/checkpoints/last-EMA.ckpt")):
    try:
        d = torch.load(c, map_location="cpu", weights_only=False, mmap=True)
        print(f"{c.split('/')[0]:70s} step={d.get('global_step','?')} epoch={d.get('epoch','?')}")
    except Exception as e:
        print(f"{c.split('/')[0]:70s} ERR {type(e).__name__}")
PY
```

Then update the `Step cadence` column (compact `100k–Xk @100k` when uniform) and `Last-EMA step` column in each table. Mark `Last-EMA step` with **†** when it sits below the highest periodic step (rename-fork case).
