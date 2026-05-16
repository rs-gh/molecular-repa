# `pretrained_probe_paper_n128/` — Pipeline B paper-table CATH sweep at n=128

> **Pending rename**: this directory will move to
> `paper/n128_paper_cath/cath/` once the in-flight probe job (29193420)
> finishes. The deferred rename is tracked in
> [../../README.md](../../README.md) and handled in
> [scripts/paper/plot_cath_results.py](../../scripts/paper/plot_cath_results.py)'s
> `_results_dir()`.

Paper-quality CATH probe results, one row per n=128 paper-table entry on the
generation side.

## Protocol

- **Pipeline**: B (pretrained probe)
- **Config profile**: `paper_n128_cath` in [../../sweep_config.yaml](../../sweep_config.yaml)
- **Probe**: CATH only; C, A, T levels
- **max_size**: 128
- **n_train**: 5000
- **n_eval**: 1237 (all proteins in val.lmdb ≤128 residues)
- **head_type**: linear
- **timestep**: 1.0 only

## Checkpoints

23 runs configured in `sweep_config.yaml` `paper_n128_cath.runs`. Covers
baseline (bs=24, bs=80, lr3x variants), REPA L0/L4/L9 at bs=24 and bs=80,
encoder ablations (random, PW-Structure, PW-Torsional, MPNN, ESM),
lambda/wd ablations, per-residue variants, and `pretrained_dfs_60m` as the
12-layer reference.

## Driver

```bash
sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh \
    --config paper_n128_cath
```

## Outputs

- `pretrained_sweep_results.{jsonl,csv,json}` — Pipeline B schema with
  `probe_kind="cath"` rows.
- `batch_manifest_*.json` — frozen protein-sample manifests.
- Plotted into `figures/paper/n128_paper_cath/cath/` by
  [scripts/paper/plot_cath_results.py](../../scripts/paper/plot_cath_results.py)
  (`--sweep paper_n128_cath`).
