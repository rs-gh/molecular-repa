# Variance + sampler-regime sweep (n128 / n256)

Two-axis extension of the standard sample-matched generation sweep, designed to
answer:

1. **How noisy are the reported metrics?** N=3 replicate runs per checkpoint at
   the current SDE sampler (`sc_scale_noise=0.45`), report mean ± sample sd.
2. **Does sampler choice move the metrics?** ODE (`vf`) and SDE with
   `sc_scale_noise ∈ {0.0, 1.0}`, narrowed to 2 checkpoints per profile
   (baseline + one REPA pick) to keep cost bounded.

Per-task wall times (from real recent logs):
- n128 sample-matched lite: ~17–45 min/task (median ~20 min)
- n256 sample-matched lite: ~30–98 min/task (median ~60 min)

## Total budget

| Bucket | Tasks | Median wall | GPU·h |
|---|---|---|---|
| n128 Axis A (variance) | 12 | 20 min | ~4 |
| n128 Axis B (sampler) | 18 | 20 min | ~6 |
| n256 Axis A (variance) | 12 | 60 min | ~12 |
| n256 Axis B (sampler) | 18 | 60 min | ~18 |
| **Total** | **60** | | **~40** |

## Replicate seeds

`42 + 1000·rep_idx` → `{42, 1042, 2042}` for N=3. The 1000-step gap avoids
collision with ProteinMPNN's per-length seed derivation
(`seed + nres`, [evaluate.py:_subsample](../../scripts/evaluate.py)) for any
plausible protein length.

## Data safety

- `output_suffix` extends with `__{sampler_tag}__rep{N}` so new
  `eval_output/.../` dirs **never overwrite** existing single-rep ones.
- `sweep_results.jsonl` is append-only. Done-set key is
  `(run, step, sampler_tag, rep_idx)` so a rerun of one replicate doesn't
  re-run the others, and legacy rows (no sampler_tag/rep_idx) remain valid.
- Reruns of the same (run, step, sampler_tag, rep_idx) get last-write-wins
  dedup at consolidation time (matching run_sweep.consolidate).

## Launch

Pick the REPA checkpoint(s) for Axis B first. Defaults below use
`repa_l4_{n}` (layer-4 REPA, strongest in current pareto); override with
`--runs <baseline>,<your_pick>`.

### Smoke (Recommended before fan-out)

Validate plumbing with one intr-qos task before submitting the full matrix:

```bash
# ODE, rep 0, one n128 baseline ckpt. ~20 min wall, intr queue starts in <1 min.
EVAL_SEED=42 sbatch --qos=intr --time=00:45:00 --array=0 \
  hpc-scripts/proteina/evaluation/generation/run_sweep.sh \
  --config n128 --runs baseline_128 \
  --sampling_mode vf --rep_idx 0 \
  --output_dir evaluation/proteina/generation/results/variance/n128
```

After the task finishes, check:

```bash
# New eval_output dir with the expected suffix.
ls eval_output/inference_lite_inference_fid_60m_baseline_128_lite_sweep_baseline_128_step_800000__ode__rep0/

# New JSONL row with sampler_tag + rep_idx.
tail -1 evaluation/proteina/generation/results/variance/n128/sweep_results.jsonl
```

### Full launch

```bash
# Axis A — variance reps on current sde_n0.45 (3 array jobs × 4 tasks = 12)
hpc-scripts/proteina/evaluation/generation/submit_variance_sweep.sh \
  --profile n128 --axis A --reps 3 --dry_run
# then drop --dry_run

# Axis B — sampler ablation on 2 ckpts (9 array jobs × 2 tasks = 18)
hpc-scripts/proteina/evaluation/generation/submit_variance_sweep.sh \
  --profile n128 --axis B --reps 3 \
  --runs baseline_128,repa_l4_128 --dry_run

# Same for n256
hpc-scripts/proteina/evaluation/generation/submit_variance_sweep.sh \
  --profile n256 --axis A --reps 3
hpc-scripts/proteina/evaluation/generation/submit_variance_sweep.sh \
  --profile n256 --axis B --reps 3 \
  --runs baseline_256,repa_l4_256
```

## Aggregate

After all replicate jobs finish:

```bash
python evaluation/proteina/joint/scripts/aggregate_variance.py \
  --jsonl evaluation/proteina/generation/results/variance/n128/sweep_results.jsonl
python evaluation/proteina/joint/scripts/aggregate_variance.py \
  --jsonl evaluation/proteina/generation/results/variance/n256/sweep_results.jsonl
```

Outputs `sweep_results_agg.{csv,jsonl,md}` alongside the input, with
`<metric>_mean` and `<metric>_sd` (sample sd, ddof=1) columns plus `n_reps`
and the seed list per group.

## Open items (not yet built)

- Representation eval probe-seed replicates (`--rep_idx` plumbing for
  `evaluation/proteina/representation/scripts/lite/run_sweep.py`). Sampler
  doesn't flow into rep eval, so only the rep axis applies. Cheap (~2 GPU·h
  total). Deferred to a follow-up.
- Pareto / convergence plot scripts reading `sweep_results_agg.csv` for
  error-bar rendering. Currently reads the raw single-rep CSVs.
- `sc_scale_score` is documented in the inference config but [the code path](../../../src/proteina/proteinfoundation/flow_matching/r3n_fm.py#L329-L334)
  ignores it — only `sc_scale_noise` is wired. Worth a separate fix.
