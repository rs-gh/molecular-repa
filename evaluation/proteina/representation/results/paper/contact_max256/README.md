# `paper/contact_max256/` — Pipeline B contact sweep (all training sizes, max_size=256)

Pretrained-probe (REPA-paper-style) contact sweep covering every entry in
`RUN_SCHEDULES`. Renamed 2026-05-11 from `n256_paper_contact/` because the
"256" refers to the **probe input cap**, not the training-size of the model.
Training-size n=128, n=256, and n=512_sm runs are all probed here.

## Protocol

- **Pipeline**: B (pretrained probe — train on `train.lmdb`, eval on `val.lmdb`)
- **Config profile**: `pretrained_probe` in [../../../sweep_config.yaml](../../../sweep_config.yaml)
- **Probes**: contact (P@L, P@L/2, P@L/5); CATH rows present if `probes` flag
  was set to `"contact,cath"` (current default)
- **max_size**: 256 — proteins >256 residues are truncated. **Caveat**: the
  `*_512_sm` runs (trained at n=512) are *being probed below their training
  receptive field*. Probe results are structurally valid but do not match the
  training-time context. For paper-quality 512-context numbers, run a separate
  sweep with `--runs baseline_512_sm,repa_l{0,4,9}_512_sm --max_size 512`.
- **n_train**: 1000 (elbow from sample-size curve, see `sample_size_curve.png`)
- **n_eval**: 500 proteins from `val.lmdb`
- **head_type**: MLP
- **timesteps**: 1.0, 0.75, 0.5, 0.0

## Checkpoints

17 runs total (training-size bucket inferred from suffix):

| n bucket | Runs |
|---|---|
| `n128` | `baseline_128`, `repa_l0_128`, `repa_l4_128`, `repa_l9_128`, `esm_repa_l0_128`, `esm_repa_l4_128`, `esm_repa_l9_128` |
| `n256` | `baseline_256`, `repa_l0_256`, `repa_l4_256`, `repa_l9_256` |
| `n512` (truncated to 256) | `baseline_512_sm`, `repa_l0_512_sm`, `repa_l4_512_sm`, `repa_l9_512_sm`, plus the older `baseline` (no suffix) |
| Reference | `pretrained_dfs_60m` (NVIDIA NGC 12-layer ckpt) |

Each run is probed at its last step × all trunk layers × 4 timesteps.

## Driver

```bash
sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh \
    --config pretrained_probe
```

Phase-1 sample-size selection (one-off; produces `sample_size_curve.{csv,json,png}` here):

```bash
sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh --sample_size
```

## Outputs

- `pretrained_sweep_results.{jsonl,csv,json}` — Pipeline B flat schema with
  `probe_kind` discriminator, `train_manifest=train_v1`, `eval_manifest=eval_v1`.
- `batch_manifest_*.json` — frozen protein-sample manifests used for train/eval.
- `sample_size_curve.{csv,json,png}` — Phase 1 learning-curve output.
- Plotted into `figures/paper/contact_max256/contact/` by
  [scripts/paper/plot_contact_probe.py](../../../scripts/paper/plot_contact_probe.py):
  1×3 grid (n=128 / n=256 / n=512) of layer curves for P@L, P@L/2, P@L/5.
