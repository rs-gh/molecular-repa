# `lite/n256_lite/` — Pipeline A, single-step probes at max_size=256

## Protocol

- **Pipeline**: A (in-place 80/20 split of ~200 proteins from `val.lmdb`)
- **Config profile**: `n256` in [../../../sweep_config.yaml](../../../sweep_config.yaml)
- **Probes**: contact (P@L/5) **and** CATH (T-level)
- **max_size**: 256
- **n_proteins**: 200 (→ ~40 test)
- **timesteps**: 1.0, 0.75, 0.5

## Checkpoints

10 runs:

| Family | Runs | Step probed |
|---|---|---|
| In-house 60M baseline | `baseline_256` | 400k |
| In-house REPA (gearnet) | `repa_l0_256`, `repa_l4_256`, `repa_l9_256` | 400k each |
| Frozen encoder | `gearnet` | n/a |
| Analytic baselines | `distance_only`, `random_gauss`, `random_rank`, `seq_onehot`, `untrained_proteina` | n/a |

## Driver

```bash
sbatch hpc-scripts/proteina/evaluation/representation/run_sweep.sh \
    --sweep --config n256
```

## Outputs

- `sweep_results.{jsonl,csv,md,json}` — Pipeline A schema.
- Plotted into `figures/lite/{contact,cath}/` (joint cross-size grid).
