# `paper/n256_paper_cath/` — Pipeline B paper-table CATH sweep at n=256

Paper-quality CATH probe results, one row per generation-paper-table entry.
Companion to the n=256 paper table on the generation side.

The CATH probe data lives in the `cath/` subdir (matches the lowest-level
`{contact,cath}/` convention used throughout).

## Protocol

- **Pipeline**: B (pretrained probe — train on `train.lmdb`, eval on `val.lmdb`)
- **Config profile**: `paper_n256_cath` in [../../../../sweep_config.yaml](../../../../sweep_config.yaml)
- **Probe**: CATH only (`probes: "cath"`); C, A, T levels
- **max_size**: 256
- **n_train**: 5000
- **n_eval**: 3190 (all proteins in val.lmdb ≤256 residues)
- **head_type**: linear (paper convention)
- **timestep**: 1.0 only (clean input)

## Checkpoints

17 runs configured in `sweep_config.yaml` `paper_n256_cath.runs`, including
baseline, REPA L0/L4/L9 variants, AFDB-trained variants, per-sample vs
per-residue averaging, lambda ablations, encoder ablations (ESM, MPNN,
random), and `pretrained_dfs_60m` as the 12-layer reference.

Status (2026-05-11): 8 rows present out of expected ~510 (sweep started but
mostly unrun). See JSONL for live state.

## Driver

```bash
sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh \
    --config paper_n256_cath
```

## Outputs

- `cath/pretrained_sweep_results.{jsonl,csv,json}` — Pipeline B flat schema,
  rows tagged `probe_kind="cath"`, columns `cath_accuracy` / `cath_macro_f1`
  per (run, step, layer, cath_level).
- `cath/batch_manifest_*.json` — frozen protein-sample manifests.
- Plotted into `figures/paper/n256_paper_cath/cath/` by
  [scripts/paper/plot_cath_results.py](../../../scripts/paper/plot_cath_results.py)
  (`--sweep paper_n256_cath`): per-CATH-level layer curves + per-ablation-block
  figures.
