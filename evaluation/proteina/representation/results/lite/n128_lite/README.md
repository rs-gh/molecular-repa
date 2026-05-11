# `lite/n128_lite/` — Pipeline A, single-step probes at max_size=128

## Protocol

- **Pipeline**: A (in-place 80/20 split of ~200 proteins from `val.lmdb`)
- **Config profile**: `n128` in [../../../sweep_config.yaml](../../../sweep_config.yaml)
- **Probes**: contact (P@L/5) **and** CATH (T-level), both per row
- **max_size**: 128
- **n_proteins**: 200 (→ ~40 test after L≥50 filter and 80/20 split)
- **timesteps**: 1.0, 0.75, 0.5

## Checkpoints covered (snapshot)

13 runs total (see JSONL `run` column for the canonical list):

| Family | Runs | Step probed |
|---|---|---|
| In-house 60M baseline | `baseline_128` | 800k |
| In-house REPA (gearnet target) | `repa_l0_128`, `repa_l4_128`, `repa_l9_128` | 400k each |
| In-house REPA (ESM target) | `esm_repa_l0_128`, `esm_repa_l4_128`, `esm_repa_l9_128` | 87.5k / 248.5k / 266k |
| Frozen encoder | `gearnet` | n/a |
| Analytic baselines | `distance_only`, `random_gauss`, `random_rank`, `seq_onehot`, `untrained_proteina` | n/a |

Layer index is 0–9 (10-layer trunk).

## Driver

```bash
sbatch hpc-scripts/proteina/evaluation/representation/run_sweep.sh \
    --sweep --config n128
```

Local smoke (no SLURM):
```bash
python evaluation/proteina/representation/scripts/lite/run_sweep.py \
    --config n128 --runs baseline_128 --steps 800000 --n_proteins 20
```

## Outputs

- `sweep_results.{jsonl,csv,md,json}` — Pipeline A schema (rows with nested
  `contact:{linear,mlp}` and `cath:{accuracy,...}`).
- Plotted into `figures/lite/{contact,cath}/` by
  [scripts/lite/plot_per_n.py](../../../scripts/lite/plot_per_n.py)
  (cross-size grid across `n128_lite` / `n256_lite` / `n512_lite`).
