# Representation Quality Evaluation

Probing-style eval for proteina backbones: what structural information is
linearly decodable from frozen hidden states, across checkpoints, layers, and
runs.

Headline metrics:

- **Contact P@L/5** — precision at top-L/5 predicted long-range
  (|i−j|≥24) Cα–Cα contacts (<8 Å).
- **CATH fold accuracy** — linear probe accuracy on CATH hierarchy
  (T-level by default; C/A also probed in the paper sweeps).

Results and interpretation: [FINDINGS.md](FINDINGS.md).

---

## Protocols at a glance

Three experimental protocols share this codebase. They differ on **how the
probe is trained and evaluated**, not on what the probe head looks like.

| Bucket | Pipeline | Protocol | Driver | Outputs |
|---|---|---|---|---|
| **lite** | A | In-place 80/20 split of ~200 proteins from `val.lmdb`. Fast triage; many runs/timesteps; analytic baselines included. ~40 test proteins per row. | [scripts/lite/run_sweep.py](scripts/lite/run_sweep.py) | `results/lite/n{128,256,512,128_L4_bs80}_lite/` |
| **convergence** (subset of lite) | A | Same protocol as lite but at n=512 with **many checkpoints per run** (10k–840k steps) so probe metrics can be plotted vs training step. Single pair of runs (baseline, repa_l4). | [scripts/lite/run_sweep.py](scripts/lite/run_sweep.py) (invoked with explicit `--step` lists) + [scripts/lite/plot_convergence.py](scripts/lite/plot_convergence.py) | `results/lite/n512_convergence_lite/` |
| **paper** | B | REPA-paper protocol: probe trains on a large sample of `train.lmdb` (default 1K–5K proteins), evaluates on a fixed manifest from `val.lmdb` (~500–4999 proteins). Per-checkpoint feature cache on `/dev/shm`. Paper-quality numbers. | [scripts/paper/pretrain_probe_sweep.py](scripts/paper/pretrain_probe_sweep.py) | `results/paper/contact_max256/`, `results/paper/n{128,256}_paper_cath/cath/` |

The two pipelines emit **different JSONL schemas** — never concatenate their
output files. Pipeline A rows are nested (`contact: {linear, mlp}`, `cath:
{accuracy, ...}`); Pipeline B rows are flat with a `probe_kind` discriminator.

See the per-pipeline sections lower in this file for invocation details.

---

## Directory layout

```
representation/
├── README.md                       # this file (overview + invocation)
├── FINDINGS.md                     # narrative results, interpretation
├── sweep_config.yaml               # named parameter profiles (--config <name>)
│
├── lib/                            # backbone + probe infrastructure (no figures)
│   ├── checkpoints.py              # RUN_SCHEDULES, PRETRAINED_CHECKPOINTS
│   ├── data.py / extract.py        # LMDB loading, hidden-state extraction
│   ├── feature_cache.py            # /dev/shm fp16 cache (Pipeline B only)
│   ├── labels.py / manifest.py     # CATH labels, contact maps, reproducible manifests
│   ├── sources.py                  # run name → checkpoint path resolution
│   └── probes/
│       ├── cath.py / cath_pretrained.py        # CATH probe (in-place + pretrained-split)
│       ├── contact.py / contact_pretrained.py  # contact probe (both pipelines)
│
├── utils/                          # shared plot/table helpers (figure-layer only)
│   └── plot_helpers.py             # palette, sentinel-layer codes, RUN_ALIGNED_LAYER
│
├── scripts/                        # drivers + plot code, grouped by protocol
│   ├── lite/
│   │   ├── run_sweep.py            # Pipeline A driver (also used for convergence)
│   │   ├── plot_per_n.py           # 3-size grid (n=128/256/512) → figures/lite/{contact,cath}/
│   │   ├── plot_per_n_L4_bs80.py   # bs=80 ablation → figures/lite/n128_L4_bs80_lite/{contact,cath}/
│   │   └── plot_convergence.py     # multi-step trajectory → figures/lite/n512_convergence_lite/{contact,cath}/
│   ├── paper/
│   │   ├── pretrain_probe_sweep.py # Pipeline B driver (Phase 2)
│   │   ├── sample_size_probe.py    # Pipeline B Phase 1 — pick N_train (run once)
│   │   ├── build_cath_classifier.py# Build CATH-classifier pickle for gen-eval suite
│   │   ├── plot_contact_probe.py   # → figures/paper/contact_max256/contact/
│   │   └── plot_cath_results.py    # → figures/paper/n{128,256}_paper_cath/cath/
│   └── _legacy/                    # gitignored: cath.py / contact.py / run_all.py /
│                                   # patch_cath.py / utils.py (pre-pipeline-A/B shims)
│
├── results/                        # JSONL / CSV / MD outputs (one README.md per leaf)
│   ├── inputs/cath_classifier/     # CATH labels + GearNet baseline classifier (shared input)
│   ├── lite/
│   │   ├── n{128,256,512}_lite/                # Pipeline A single-step
│   │   ├── n128_L4_bs80_lite/                  # Pipeline A bs=80 ablation
│   │   └── n512_convergence_lite/              # Pipeline A multi-step trajectory
│   ├── paper/
│   │   ├── contact_max256/               # Pipeline B contact sweep (all sizes, max_size=256)
│   │   └── n256_paper_cath/cath/               # Pipeline B paper CATH n=256
│   ├── pretrained_probe_paper_n128/            # ⚠ pending rename to paper/n128_paper_cath/cath/
│   └── _archive/                   # gitignored: legacy/orphaned sweep outputs
│
└── figures/                        # PNG outputs only; lowest level always {contact,cath}/
    ├── lite/
    │   ├── {contact,cath}/                     # cross-size grid figures (plot_per_n.py)
    │   ├── n128_L4_bs80_lite/{contact,cath}/   # bs=80 ablation per-layer
    │   └── n512_convergence_lite/{contact,cath}/  # convergence trajectory
    └── paper/
        ├── contact_max256/contact/        # contact pretrained-probe layer curves
        └── n{128,256}_paper_cath/cath/          # paper-table CATH layer curves + ablation blocks
```

The deferred rename: `results/pretrained_probe_paper_n128/` is being written to
by a probe-fit job at the time of the 2026-05-11 refactor; it'll be moved to
`results/paper/n128_paper_cath/cath/` in a follow-up commit once the job
finishes. `scripts/paper/plot_cath_results.py` already routes both old and
new locations correctly via its `_results_dir()` helper.

**Per-leaf README**: every directory under `results/<bucket>/<sweep>/` carries
a short `README.md` naming the checkpoints, protocol, profile, and driver
command — direct answer to "what was run here?" at the data location.

---

## Pipeline A — In-place split sweep (lite / convergence)

**Script:** [scripts/lite/run_sweep.py](scripts/lite/run_sweep.py)
**SLURM wrapper:** [hpc-scripts/proteina/evaluation/representation/run_sweep.sh](../../../hpc-scripts/proteina/evaluation/representation/run_sweep.sh)

Loads ~200 proteins from val.lmdb, splits 80/20 train/test in-place, trains a
contact or CATH probe head, evaluates on the held-out 20% (~40 proteins after
the L≥50 filter). Repeats across all (run, step, layer) tuples.

**When to use:**
- Fast triage across many runs / training stages.
- CATH probe (no fold-aware train/test split needed at small N).
- Analytic baselines: `random_rank`, `distance_only`, `seq_onehot`,
  `untrained_proteina`, `random_gauss`.
- Multi-timestep (t=1.0, 0.75, 0.5) and multi-seed comparisons.

**Outputs** (under `results/lite/<config_name>/`, including the convergence
sweep at `results/lite/n512_convergence_lite/`):
- `sweep_results.jsonl` — one row per probe, append-resumed
- `sweep_results.csv` / `.md` — consolidated CSV + human-readable summary

**Resume:** results are appended to the JSONL as each probe completes.
Restarting the job skips already-completed (run, step, layer, t) tuples.
Delete the JSONL to force a full rerun.

**Quick start:**

```bash
# Standard lite sweep at n=128 (fast, ~2h on A100)
sbatch hpc-scripts/proteina/evaluation/representation/run_sweep.sh \
    --sweep --config n128

# Smoke test (one run, one step, no SLURM)
python evaluation/proteina/representation/scripts/lite/run_sweep.py \
    --config n128 --runs baseline_128 --steps 800000 --n_proteins 20

# Override one field
sbatch ... run_sweep.sh --sweep --config n128 --n_proteins 50

# Convergence sweep (no config profile — supply step list explicitly)
sbatch hpc-scripts/proteina/evaluation/representation/run_sweep.sh \
    --sweep --runs baseline,repa_l4 \
    --steps 10000,20000,40000,80000,150000,250000,350000,450000,550000,650000,740000,840000 \
    --max_size 512 --output_dir results/lite/n512_convergence_lite
```

---

## Pipeline B — Pretrained-probe sweep (paper)

**Driver:** [scripts/paper/pretrain_probe_sweep.py](scripts/paper/pretrain_probe_sweep.py)
**SLURM wrapper:** [hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh](../../../hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh)

REPA-paper protocol: probe trained on a large sample from train.lmdb (separate
from the eval set), evaluated on a fixed manifest from val.lmdb. Ensures the
probe head has seen enough data to saturate — giving a ceiling estimate of
contact decodability rather than a small-sample proxy.

Two phases.

### Phase 1 — Sample-size learning curve (run once)

**Script:** [scripts/paper/sample_size_probe.py](scripts/paper/sample_size_probe.py)

Sweeps N_train ∈ {500, 1K, 2K, 5K, 10K} on a single checkpoint+layer and plots
P@L/5 vs N_train. Pick the elbow as the canonical N_train, commit it to
`sweep_config.yaml`, then run Phase 2 with that value.

**Current canonical value:** N_train = 1000 (`pretrained_probe` profile)
or 5000 (`paper_n{128,256}_cath` profiles).

```bash
sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh --sample_size
```

Output: `results/paper/contact_max256/sample_size_curve.{csv,json,png}`

### Phase 2 — Full pretrained-probe sweep

Iterates all (run, step) in `PRETRAINED_CHECKPOINTS` × all 10 layers × multi-t.
Per-checkpoint feature extraction is cached on `/dev/shm` (fp16), probed
across all layers, then purged before moving to the next checkpoint.

**Data split:**
- Train set: n_train proteins from train.lmdb (via `train_v1` manifest)
- Eval set: n_eval proteins from val.lmdb (via `eval_v1` manifest)

**When to use:** final reported contact / CATH numbers for paper/comparisons.

**Outputs** (under `results/paper/<config_name>/`):
- `pretrained_sweep_results.jsonl` — one row per (run, step, layer, t, probe)
- `pretrained_sweep_results.csv` / `.json` — consolidated

**SLURM wrapper** stages both train.lmdb (~51 GB) and val.lmdb to `/dev/shm`
and purges both on exit. Do not run on login nodes.

```bash
# Full contact sweep at n=256
sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh \
    --config pretrained_probe

# Paper CATH sweep at n=128 (writes to the pre-refactor dir until follow-up rename)
sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh \
    --config paper_n128_cath

# Smoke test
sbatch hpc-scripts/.../run_pretrained_probe.sh \
    --config pretrained_probe --runs baseline --steps 400000 \
    --skip_pretrained_refs
```

---

## sweep_config.yaml profiles

| Profile | max_size | n_proteins / n_train / n_eval | output_dir | Pipeline |
|---|---|---|---|---|
| `n128` | 128 | n=200 | `results/lite/n128_lite` | A |
| `n256` | 256 | n=200 | `results/lite/n256_lite` | A |
| `n512` | 512 | n=200 | `results/lite/n512_lite` | A |
| `n128_L4_bs80` | 128 | n=200 | `results/lite/n128_L4_bs80_lite` | A |
| `pretrained_probe` | 256 | n_train=1000, n_eval=500 | `results/paper/contact_max256` | B |
| `paper_n128_cath` | 128 | n_train=5000, n_eval=1237 | `results/pretrained_probe_paper_n128` (rename pending) | B |
| `paper_n256_cath` | 256 | n_train=5000, n_eval=3190 | `results/paper/n256_paper_cath/cath` | B |

Pass `--config <name>` to load a profile. Any additional CLI flags override
individual fields and are logged, making deviations from canonical settings
explicit in the job output.

---

## Output schema

| Field | Pipeline A (`sweep_results.jsonl`) | Pipeline B (`pretrained_sweep_results.jsonl`) |
|---|---|---|
| `run, step, layer` | yes | yes |
| `t` | yes (multi-timestep) | yes (multi-timestep) |
| `p_at_L, p_at_L_2, p_at_L_5` | yes (nested under `contact:{linear,mlp}`) | yes (flat) |
| `cath_acc, cath_f1` | yes (nested under `cath:`) | no (cath rows tagged `probe_kind=cath` with flat `cath_accuracy`/`cath_macro_f1`) |
| `n_proteins_test` | yes (~40) | yes (~490–3190) |
| `manifest` | `v1` / `v2` | — |
| `train_manifest, eval_manifest` | — | `train_v1`, `eval_v1` |
| `probe_kind` | no | `contact` or `cath` |
| `seed` | yes | yes |

Plot scripts under `scripts/lite/`, `scripts/convergence/`, and
`scripts/paper/` each read their own results dir — they're not designed to be
swapped between schemas.
