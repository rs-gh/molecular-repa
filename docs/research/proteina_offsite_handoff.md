# Proteina Offsite Handoff

How to continue proteina generation / representation evals on another cluster when this machine is unavailable (e.g. CSD3 maintenance windows).

The bundle is built once on the source cluster and mirrored to a private HF model repo, then pulled on the destination cluster. Code lives on GitHub.

## Bundle contents

`hf://rscam/proteina-repa-convergence` — 158 files, 90.75 GB.

Curated subset of `/rds/user/sr2173/hpc-work/proteina/store/<run>/checkpoints/` covering every (run, step) pair consumed by the n=128 and n=256 convergence plots, plus each run's `data_config_*.json` / `exp_config_*.json`. Only the `-EMA.ckpt` variants are shipped — the non-EMA twins are pure resume-state and excluded (see [reference_proteina_ckpt_ema_dupes](../../memory/reference_proteina_ckpt_ema_dupes.md)).

| Group | Runs | EMA ckpts |
|---|---:|---:|
| n=128 AFDB convergence (incl. new MPNN-L9) | 4 | 17 |
| n=128 PDB bs80 cohort | 7 | 27 |
| n=256 PDB convergence | 6 | 33 |
| n=256 AFDB convergence | 5 | 32 |
| **Total** | **22** | **109** + 2 misc |

Run × step inventory: see [proteina_ablation_checkpoints.md](proteina_ablation_checkpoints.md). Source-of-truth manifest lives at `/rds/user/sr2173/hpc-work/proteina_offsite_pkg/manifest.jsonl` on the source cluster, with sha256 per file.

## Building the bundle (source cluster)

```bash
# 1. Stage to /rds (resume-safe, sha256 per file, ~4 min on Lustre)
source .venv/bin/activate
python3 playground/proteina/package_offsite/package.py \
    --dst /rds/user/sr2173/hpc-work/proteina_offsite_pkg

# 2. Verify
python3 playground/proteina/package_offsite/package.py \
    --dst /rds/user/sr2173/hpc-work/proteina_offsite_pkg --verify

# 3. Push to HF (runs ~15 min on icelake CPU partition)
sbatch hpc-scripts/proteina/data_prep/upload_offsite_pkg_hf.sh
```

The packager is incremental: re-running skips files whose dst sha matches the manifest. The HF Slurm job uses `hf upload-large-folder`, which is itself resumable (state under `$STAGING/.cache/huggingface/`). To extend coverage (e.g. add a new run/step), edit the `RUNS` dict in [package.py](../../playground/proteina/package_offsite/package.py) and re-run both steps.

Prereq, one-time: `hf auth login` on the source cluster with a write token. The Slurm job reads `~/.cache/huggingface/token` and exports it as `HF_TOKEN` (the script overrides `HF_HOME` to `/rds`, which would otherwise redirect the token lookup).

## Pulling on the destination cluster

```bash
git clone git@github.com:rs-gh/molecular-repa.git
cd molecular-repa
# (set up .venv with uv as on the source cluster — see CLAUDE.md / project docs)
source .venv/bin/activate

hf auth login   # paste a token with read access to rscam/proteina-repa-convergence

hf download rscam/proteina-repa-convergence \
    --repo-type=model \
    --local-dir /path/on/destination/proteina_offsite_pkg
```

`hf download` is resumable, skips files whose etag matches. Useful flags:

- `--include 'proteina_60m_baseline_256*'` — pull just one run for fast iteration.
- `--max-workers 8` — parallel downloads.

Layout on disk after download mirrors `/rds/.../store/`:

```
proteina_offsite_pkg/
├── proteina_60m_baseline_128_bs80/checkpoints/
│   ├── chk_epoch=…_step=000000100000-EMA.ckpt
│   ├── …
│   ├── data_config_proteina_60m_baseline_128_bs80.json
│   └── exp_config_proteina_60m_baseline_128_bs80.json
├── proteina_60m_baseline_256_bs24_2gpu/checkpoints/…
└── …  (22 run dirs)
```

## Wiring existing eval scripts to the downloaded path

Inference / representation configs reference `/rds/user/sr2173/hpc-work/proteina/store/<run>/checkpoints/<ckpt>`. Three ways to redirect on the destination cluster, from least to most invasive:

1. **Sed-replace in configs.** `find <configs-dir> -name '*.yaml' -exec sed -i 's|/rds/user/sr2173/hpc-work/proteina/store|/path/to/proteina_offsite_pkg|g' {} +`. Explicit, easy to revert.
2. **Symlink tree at the original path** (if you have write access to that path on the destination): `ln -s /path/to/proteina_offsite_pkg /rds/.../proteina/store`. Zero code change.
3. **Per-run symlinks** into a `store/` parent: `for d in /path/to/proteina_offsite_pkg/proteina_60m_*; do ln -s "$d" ~/proteina_store/$(basename "$d"); done`, then point the configs at `~/proteina_store`.

Path resolution code: see [evaluation/proteina/lib/checkpoints.py](../../evaluation/proteina/lib/checkpoints.py).

## What is NOT in the bundle

Generation/representation evals also depend on:

- **Foldseek DBs** (`/rds/user/sr2173/hpc-work/proteina/foldseek_dbs/{pdb,afdb_swissprot}/`, 6.8 GB) — required for novelty metrics.
- **PDB val.lmdb** — required for representation probes.
- **Inference / training YAMLs** — these are in the git repo, so come for free.

Extend the bundle by adding these to a separate HF dataset repo if needed, or rsync them point-to-point — see chat log from 2026-05-17 for the rationale on splitting them out.

The `src/proteina` git submodule is also pulled by `git submodule update --init --recursive` after clone — needed because the training entrypoint lives there.
