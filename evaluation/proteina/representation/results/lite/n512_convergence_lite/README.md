# `lite/n512_convergence_lite/` — Pipeline A, n=512 multi-step trajectory

Same probe protocol as the other `lite/` sweeps, but with **11–12 training
checkpoints per run** so probe metrics can be plotted vs training step.
Single pair of runs only — the question this answers is "when does
representation quality saturate?", not "how do runs compare?".

## Protocol

- **Pipeline**: A (in-place 80/20 split of ~200 proteins from `val.lmdb`)
- **Config profile**: none — driven by an explicit `--steps` list
- **Probes**: contact (P@L/5) and CATH (T-level)
- **max_size**: 512
- **n_proteins**: 200
- **timestep**: 1.0 only (clean input)

## Checkpoints

| Run | Steps probed |
|---|---|
| `baseline` | 10k, 20k, 40k, 80k, 150k, 250k, 350k, 450k, 550k, 650k, 740k (11 ckpts; bs=6) |
| `repa_l4` | 10k, 20k, 40k, 80k, 150k, 250k, 350k, 450k, 550k, 650k, 750k, 840k (12 ckpts; bs=4) |
| `gearnet` (frozen) | 1 row |
| `pretrained_dfs_60m` (NVIDIA NGC 12-layer reference) | 12 layers × 1 row |

Note: this is the n=512 v2 ablation, **not** the `_sm` runs in `n512_lite/`.
Run names lack a size suffix (`baseline`, `repa_l4`) because they predate the
suffix convention.

## Driver

```bash
sbatch hpc-scripts/proteina/evaluation/representation/run_sweep.sh \
    --sweep --runs baseline,repa_l4 \
    --steps 10000,20000,40000,80000,150000,250000,350000,450000,550000,650000,740000,840000 \
    --max_size 512 --output_dir results/lite/n512_convergence_lite
```

## Outputs

- `sweep_results.{jsonl,csv,md,json}` — Pipeline A schema.
- Plotted into `figures/lite/n512_convergence_lite/{contact,cath}/` by
  [scripts/lite/plot_convergence.py](../../../scripts/lite/plot_convergence.py):
  - `<probe>/fig_layerwise_*.png` — per-layer curves at each run's final step.
  - `<probe>/fig_step_progression.png` — probe metric vs training samples
    processed (x-axis is `step × batch_size` to make bs=6 baseline / bs=4
    REPA comparable).

## Consumers

[evaluation/proteina/joint/scripts/pareto.py](../../../../joint/scripts/pareto.py)
reads this CSV to build the joint probe × FID Pareto plots (Fig 3c analogue).
