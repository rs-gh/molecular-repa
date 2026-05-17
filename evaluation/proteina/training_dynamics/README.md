# Proteina training dynamics

Per-ablation loss curves for every proteina run tracked in
[`docs/research/proteina_ablation_checkpoints.md`](../../../docs/research/proteina_ablation_checkpoints.md).

## Layout

- `scripts/runs.py` — wandb run registry. Mirrors the ablation doc 1:1
  (11 ablation blocks, ~50 unique runs). Baseline-first within each block.
- `scripts/fetch_histories.py` — pulls per-run history from wandb and caches
  one pickled DataFrame per run under `cache/<run_id>.pkl`.
- `scripts/plot_training_dynamics.py` — emits one figure per
  (ablation × metric × x-axis) under `figures/`.

## What's plotted

Three independent metrics:

| Metric key (plot id) | wandb key                  | Includes baselines? |
|---|---|---|
| `trans` (FM)         | `train/trans_loss_epoch`   | yes (apples-to-apples vs REPA) |
| `repa`               | `train/repa/loss_epoch`    | no (baselines don't have it)   |
| `total`              | `train/loss_epoch`         | yes (= trans + aux + λ·repa)   |

X-axis is either `trainer/global_step` (default) or
`scaling/nsamples_processed` (`--x nsamples` — batch-size-fair view).

## Refresh

```bash
source .venv/bin/activate
# 1. pull/refresh wandb caches (~5-10 min cold, seconds when cached)
python evaluation/proteina/training_dynamics/scripts/fetch_histories.py [--refresh] [--only <run_id> ...]
# 2. re-render all 66 figures (33 × 2 x-axes)
python evaluation/proteina/training_dynamics/scripts/plot_training_dynamics.py --smooth 5
python evaluation/proteina/training_dynamics/scripts/plot_training_dynamics.py --smooth 5 --x nsamples
```

## Notes & gotchas

- **wandb sampling.** We use `run.history(samples=4000)`, which is server-side
  downsampled. ~250-step resolution on a 1M-step run is plenty for smoothed
  convergence views; bump `SAMPLES` if you need fidelity.
- **Per-key fetch, then merge.** `history(keys=[k1,k2,...])` returns the
  *intersection* of rows where every key has a value. Epoch-cadence losses
  (`*_epoch`) and step-cadence counters (`trainer/global_step`,
  `scaling/nsamples_processed`) live on disjoint rows, so intersecting drops
  most of the curve. We pull each key in its own call and merge on `_step`.
- **Forward-fill of step counters.** The merged epoch-loss rows initially
  carry NaN for `trainer/global_step` / `scaling/nsamples_processed`; we
  ffill on `_step` order so every loss point gets a usable x value.
- **Averaging-block forks.** The 2026-04-17 rename forked four logical
  models into pre-rename (`_perres` / `_persamp` / bare) and post-rename
  (`_per_residue` / `_per_sample`) wandb runs. Both ids are listed in
  `runs.ABLATIONS["n256_pdb_averaging"]` and labelled `(part1: pre-rename)`
  / `(part2: post-rename)` so the fork is visible in the plots.
- **ESM-t30 runs.** Both ESM-t30 entries have ≤14 epoch points (`esm_l4_t30_128`
  has 2; `esm_l9_t30_256` has 14) — they emit a curve but it's barely visible.
  Reflects how little training they did.
- **Smoothing.** `--smooth N` is a centred rolling mean over N points. The
  defaults in committed figures used `--smooth 5`.
