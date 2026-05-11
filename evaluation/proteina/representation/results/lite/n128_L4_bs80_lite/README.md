# `lite/n128_L4_bs80_lite/` — Pipeline A, n=128 L4 bs=80 ablation

Layer-4-only ablation of the bs=80 family at n=128, with two-step samples
(100k / 200k) for each variant so you can see early- vs late-training behavior.

## Protocol

- **Pipeline**: A (in-place 80/20 split of ~200 proteins from `val.lmdb`)
- **Config profile**: `n128_L4_bs80` in [../../../sweep_config.yaml](../../../sweep_config.yaml)
- **Probes**: contact (P@L/5) and CATH (T-level)
- **max_size**: 128
- **n_proteins**: 200
- **timesteps**: 1.0, 0.75, 0.5

## Checkpoints

4 student runs, all bs=80, all REPA at L4 (or baseline) — focuses on what
varies at the L4 alignment point:

| Run | Step(s) probed | Notes |
|---|---|---|
| `baseline_128_bs80` | 100k, 200k | bs=80 baseline (cf. bs=24 elsewhere) |
| `repa_l4_128_bs80` | 100k, 200k | REPA at L4, CA-GearNet target |
| `repa_l4_128_bs80_lr3x` | 100k, 161k | 3× learning rate |
| `repa_l4_128_random` | 100k, 200k | random-init target (control) |

Plus analytic/encoder baselines: `gearnet`, `distance_only`, `random_gauss`,
`random_rank`, `seq_onehot`, `untrained_proteina`.

Renamed 2026-05-06 from `n128_bs80_sweep` to make the layer-4-only scope explicit.

## Driver

```bash
sbatch hpc-scripts/proteina/evaluation/representation/run_sweep.sh \
    --sweep --config n128_L4_bs80 \
    --runs baseline_128_bs80,repa_l4_128_bs80,repa_l4_128_bs80_lr3x,repa_l4_128_random
```

## Outputs

- `sweep_results.{jsonl,csv,md,json}` — Pipeline A schema.
- Plotted into `figures/lite/n128_L4_bs80_lite/{contact,cath}/` by
  [scripts/lite/plot_per_n_L4_bs80.py](../../../scripts/lite/plot_per_n_L4_bs80.py).
