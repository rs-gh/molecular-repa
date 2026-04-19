# training

Slurm launchers for proteina training. Both scripts auto-resume from `last.ckpt` in the run's store dir and continue the matching WandB run.

## Scripts

| File | Role |
|---|---|
| `train_baseline.sh` | Launch the 60M baseline (no REPA, torch.compile enabled). Copies LMDB to local NVMe before training to avoid Lustre mmap thrashing, with a size-checked fallback to Lustre. |
| `train_repa.sh` | Launch REPA-aligned training. Takes `<config_name> <config_subdir>` positional args to pick a config (default: `training_repa` flat symlink). Same LMDB-copy logic as the baseline. |

## Usage

```bash
sbatch hpc-scripts/proteina/training/train_baseline.sh
sbatch hpc-scripts/proteina/training/train_repa.sh
sbatch hpc-scripts/proteina/training/train_repa.sh training_repa_l4_256_per_residue training/256/gearnet/per_residue
```

Before submitting:

- `DATA_PATH` env var points to your proteina data directory (must contain PDB LMDB + `metric_factory/model_weights/gearnet_ca.pth` for REPA).
- Log dir exists: `mkdir -p /rds/user/$USER/hpc-work/proteina/logs`.
- LMDB has been built (`data_prep/convert_to_lmdb.sh`) and the length index precomputed (`data_prep/build_length_index.sh`).
