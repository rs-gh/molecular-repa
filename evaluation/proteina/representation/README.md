# Representation Quality Evaluation

Measures what structural information is linearly decodable from frozen Proteina
backbone representations, across checkpoints, layers, and runs. Two metrics:

- **Contact P@L/5** — precision at top-L/5 predicted long-range (|i−j|≥24)
  Cα–Cα contacts (<8 Å). Headline metric.
- **CATH fold accuracy** — linear probe accuracy on the T-level CATH hierarchy.

Two probe protocols live side by side (see below). Results and interpretation
are in [FINDINGS.md](FINDINGS.md).

---

## Directory layout

```
representation/
├── lib/                          # shared library — import from scripts
│   ├── checkpoints.py            # RUN_SCHEDULES, PRETRAINED_CHECKPOINTS: canonical
│   │                             #   (run, step, ckpt_path) schedule for both pipelines
│   ├── data.py                   # LMDB loading, batch collation
│   ├── extract.py                # backbone feature extraction (all layers, one forward)
│   ├── feature_cache.py          # fp16 /dev/shm cache for pretrained pipeline
│   ├── labels.py                 # CATH labels, contact map construction
│   ├── manifest.py               # reproducible protein-sample manifests (JSON)
│   ├── sources.py                # run name → checkpoint path resolution
│   └── probes/
│       ├── cath.py               # CATH probe (linear + MLP); train + eval functions
│       ├── contact.py            # in-place contact probe; also exports _build_head
│       └── contact_pretrained.py # pretrained-split contact probe (Pipeline B)
│
├── scripts/
│   ├── run_sweep.py              # Pipeline A: in-place split sweep (contact + CATH)
│   ├── pretrain_probe_sweep.py   # Pipeline B, Phase 2: pretrained-probe sweep
│   ├── sample_size_probe.py      # Pipeline B, Phase 1: pick N_train elbow (run once)
│   ├── plot.py                   # figures from Pipeline A results
│   ├── plot_per_n.py             # per-protein-size figures from Pipeline A
│   ├── cath.py                   # legacy one-off CATH diagnostic (not part of sweep)
│   ├── contact.py                # legacy one-off contact diagnostic
│   └── utils.py                  # shared plot/table helpers
│
├── sweep_config.yaml             # named parameter profiles (--config flag)
├── FINDINGS.md                   # narrative results and interpretation
├── figures/                      # generated PNGs (git-tracked)
└── results/                      # JSONL / CSV / MD outputs (gitignored)
    ├── n128_val/                 # Pipeline A output at max_size=128
    ├── n256_val/                 # Pipeline A output at max_size=256
    ├── n512_val/                 # Pipeline A output at max_size=512
    └── pretrained_probe/         # Pipeline B output
        └── probe_heads/          # saved head state_dicts (--save_heads)
```

---

## Pipeline A — In-place split sweep

**Script:** `scripts/run_sweep.py`
**SLURM wrapper:** `hpc-scripts/proteina/evaluation/representation/run_probes.sh`

Loads a small batch (~200 proteins) from val.lmdb, splits 80/20
train/test in-place, trains a contact or CATH probe head, and evaluates on the
held-out 20% (~40 proteins after the L≥50 filter). Repeats across all
(run, step, layer) tuples in `RUN_SCHEDULES`.

**When to use:**
- Fast triage across many runs / training stages
- CATH probe (no fold-aware train/test split needed at small N)
- Analytic baselines: `random_rank`, `distance_only`, `seq_onehot`,
  `untrained_proteina`, `random_gauss`
- Multi-timestep (t=1.0, 0.75, 0.5) and multi-seed comparisons

**Outputs** (under `results/<config_name>/`):
- `sweep_results.jsonl` — one row per probe, append-resumed
- `sweep_results.csv` — consolidated from JSONL
- `sweep_results.md` — human-readable summary table

**Resume:** results are appended to the JSONL as each probe completes.
Restarting the job skips already-completed (run, step, layer, t) tuples.
Delete the JSONL to force a full rerun.

**Quick start:**
```bash
# Standard sweep at n=128 (fast, ~2h on A100)
sbatch hpc-scripts/proteina/evaluation/representation/run_probes.sh \
    --sweep --config n128

# Smoke test (one run, one step, no SLURM)
python evaluation/proteina/representation/scripts/run_sweep.py \
    --config n128 --runs baseline --steps 400000 --n_proteins 20

# Override one field
sbatch ... run_probes.sh --sweep --config n128 --n_proteins 50
```

---

## Pipeline B — Pretrained-probe sweep

**Script:** `scripts/pretrain_probe_sweep.py`
**SLURM wrapper:** `hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh`

REPA-paper protocol: probe trained on a large sample from train.lmdb (separate
from the eval set), evaluated on a fixed manifest from val.lmdb. Ensures the
probe head has seen enough data to saturate — giving a ceiling estimate of
contact decodability rather than a small-sample proxy.

This pipeline runs in two phases:

### Phase 1 — Sample-size learning curve (run once)

**Script:** `scripts/sample_size_probe.py`

Sweeps N_train ∈ {500, 1K, 2K, 5K, 10K} on a single checkpoint+layer and plots
P@L/5 vs N_train. Pick the elbow as the canonical N_train, commit it to
`sweep_config.yaml`, then run Phase 2 with that value.

**Current canonical value:** N_train = 5000 (2026-04-24).
Elbow: 500→0.868, 1K→0.891, 2K→0.899, 5K→0.903, 10K→0.904. The 5K→10K
gain is +0.001 at double the backbone-forward cost.

```bash
sbatch hpc-scripts/.../run_pretrained_probe.sh --sample_size
```

Output: `results/pretrained_probe/sample_size_curve.{csv,json,png}`

### Phase 2 — Full pretrained-probe sweep

Iterates all (run, step) in `PRETRAINED_CHECKPOINTS` × all 10 layers × t=1.0.
Per-checkpoint feature extraction is cached on `/dev/shm` (fp16), probed
across all layers, then purged before moving to the next checkpoint.

**Data split:**
- Train set: n_train=5000 proteins from train.lmdb (via `train_v1` manifest)
- Eval set: n_eval=500 proteins from val.lmdb (via `eval_v1` manifest)
  - val.lmdb has **4,999 proteins total**; the eval manifest samples 500
  - After the L≥50 filter: ~490 proteins reach the scorer
  - Bump `n_eval` to 4999 in `sweep_config.yaml` for paper-quality eval
    (adds ~40 min to the job, ~10× more stable metrics)

**When to use:** final reported contact numbers for paper/comparisons.
Not suitable for CATH (no fold-aware split) or analytic baselines.

**Outputs** (under `results/pretrained_probe/`):
- `pretrained_sweep_results.jsonl` — one row per probe, append-resumed
- `pretrained_sweep_results.csv` / `.json` — consolidated
- `probe_heads/<head_type>/` — saved head bundles (`--save_heads` flag)

**SLURM wrapper** stages both train.lmdb (~51 GB) and val.lmdb to `/dev/shm`
and purges both on exit. Do not run on login nodes.

```bash
# Full sweep
sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh \
    --config pretrained_probe

# Smoke test (one run, one step)
sbatch hpc-scripts/.../run_pretrained_probe.sh \
    --config pretrained_probe --runs baseline --steps 400000 \
    --skip_pretrained_refs

# With saved heads
sbatch hpc-scripts/.../run_pretrained_probe.sh \
    --config pretrained_probe --save_heads
```

---

## sweep_config.yaml profiles

| Profile | max_size | n_proteins | output_dir | Pipeline |
|---|---|---|---|---|
| `n128` | 128 | 200 | results/n128_val | A |
| `n256` | 256 | 200 | results/n256_val | A |
| `n512` | 512 | 200 | results/n512_val | A |
| `pretrained_probe` | 256 | n_train=5000, n_eval=500 | results/pretrained_probe | B |

Pass `--config <name>` to load a profile. Any additional CLI flags override
individual fields and are logged, making deviations from canonical settings
explicit in the job output.

**Note:** `pretrained_probe` uses `max_size=256`, which truncates inputs for
the `*_512_sm` runs (trained at n=512). For paper-quality numbers on those
runs, add a separate sweep with `--runs repa_l0_512_sm,... --max_size 512`.

---

## Output schema

The two pipelines emit **different JSONL schemas** — do not concatenate their
output files.

| Field | Pipeline A (`sweep_results.jsonl`) | Pipeline B (`pretrained_sweep_results.jsonl`) |
|---|---|---|
| `run, step, layer` | yes | yes |
| `t` | yes (multi-timestep) | yes (always 1.0) |
| `p_at_L, p_at_L_2, p_at_L_5` | yes | yes |
| `cath_acc, cath_f1` | yes | no |
| `n_proteins_test` | yes (~40) | yes (~490) |
| `manifest` | `v1` / `v2` | — |
| `train_manifest, eval_manifest` | — | `train_v1`, `eval_v1` |
| `seed` | yes | yes |

Plot scripts (`plot.py`, `plot_per_n.py`) read Pipeline A CSV only.
Pipeline B has its own consolidation step inside `pretrain_probe_sweep.py`.
