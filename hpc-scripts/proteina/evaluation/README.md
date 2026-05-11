# evaluation

SLURM wrappers for proteina evaluation. Scripts are split by eval type, mirroring `evaluation/proteina/`:

```
hpc-scripts/proteina/evaluation/
  generation/       ← FID / generative-quality evals  →  evaluation/proteina/generation/
  representation/   ← representation-quality probes   →  evaluation/proteina/representation/
```

Both suites share a checkpoint registry at `evaluation/proteina/lib/checkpoints.py`
(step schedules, sample-matching arithmetic, run → Hydra config mapping).

---

## generation/

Heavy lifting in `evaluation/proteina/generation/scripts/`.
Checkpoint schedules and sample-matching arithmetic documented in
`evaluation/proteina/lib/checkpoints.py`.
Sweep profiles (canonical parameter sets) in `evaluation/proteina/generation/sweep_config.yaml`.

| File | Role |
|---|---|
| `run_sweep.sh` | SLURM array: lite FID sweep (~300 samples, ~35 min/ckpt) across a schedule of checkpoints. Pass `--config <profile>`. |
| `eval_fid.sh` | Full FID on a single checkpoint (~6,125 samples, ~12h). Pass `<inference_config> [config_subdir] [extra...]`. |

```bash
# Print task table before submitting (confirms --array range)
python evaluation/proteina/generation/scripts/run_sweep.py --config n512_convergence --dry_run

# n=512 convergence curve (11 tasks for baseline, 12 for repa variants)
sbatch --array=0-10 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n512_convergence

# Sample-matched single points (4 tasks each)
sbatch --array=0-3 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n128
sbatch --array=0-3 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n256
sbatch --array=0-3 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n512_sm

# Backfill a single run within a profile
sbatch --array=0-11 hpc-scripts/proteina/evaluation/generation/run_sweep.sh \
  --config n512_convergence --runs repa_l4

# Ad-hoc single checkpoint (no --array needed)
sbatch hpc-scripts/proteina/evaluation/generation/run_sweep.sh \
  --ckpt_path /rds/...ckpt --ckpt_label myrun_100k \
  --config_name inference/inference_fid_60m_baseline_lite

# Override seed
EVAL_SEED=123 sbatch --array=0-3 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n128

# One-shot full FID on a single checkpoint
sbatch hpc-scripts/proteina/evaluation/generation/eval_fid.sh inference_fid_60m_baseline inference
```

Results land under `evaluation/proteina/generation/results/<profile>/sweep_results.{jsonl,csv,md}`.

---

## representation/

Heavy lifting in `evaluation/proteina/representation/scripts/`, grouped by
experimental protocol (`lite/` = in-place split sweep, `convergence/` = n=512
multi-step trajectory, `paper/` = pretrained-probe paper-table sweeps).
Canonical probe parameters in `evaluation/proteina/representation/sweep_config.yaml`.
Always pass `--config <name>` for production sweeps.

| File | Role |
|---|---|
| `run_sweep.sh` | Pipeline A: in-place split sweep — contact P@L/5 + CATH fold probes across checkpoints, layers, and timesteps. |
| `run_pretrained_probe.sh` | Pipeline B: pretrained-probe sweep (large train.lmdb sample, fixed val.lmdb eval manifest). |
| `run_extract_only.sh` / `run_probe_only.sh` | Pipeline B split-mode (GPU extract → CPU probe fit). |
| `build_cath_classifier.sh` | One-off: build the CATH-classifier pickle consumed by the generation eval suite. |

```bash
# Pipeline A: full sweep for a given protein-length regime
sbatch hpc-scripts/proteina/evaluation/representation/run_sweep.sh --sweep --config n128
sbatch hpc-scripts/proteina/evaluation/representation/run_sweep.sh --sweep --config n256
sbatch hpc-scripts/proteina/evaluation/representation/run_sweep.sh --sweep --config n512

# Subset of runs only
sbatch hpc-scripts/proteina/evaluation/representation/run_sweep.sh --sweep --config n128 \
  --runs esm_repa_l0_128,esm_repa_l4_128

# Override a single field (e.g. quick smoketest)
sbatch hpc-scripts/proteina/evaluation/representation/run_sweep.sh --sweep --config n128 --n_proteins 20

# Pipeline B: pretrained-probe sweep
sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh --config pretrained_probe
```

Results land under:
- `evaluation/proteina/representation/results/lite/n{128,256,512}_lite/` (Pipeline A)
- `evaluation/proteina/representation/results/lite/n512_convergence_lite/` (n=512 multi-step)
- `evaluation/proteina/representation/results/paper/contact_max256/` (Pipeline B contact sweep)
- `evaluation/proteina/representation/results/paper/n{128,256}_paper_cath/cath/` (Pipeline B paper-table CATH sweeps)
