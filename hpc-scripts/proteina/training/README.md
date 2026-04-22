# training

Slurm launchers for proteina training. All scripts auto-resume from `last.ckpt` in the run's store dir and continue the matching WandB run.

## Structure

```
training/
  pdb/
    train_baseline.sh   — 60M baseline on PDB (no REPA, torch.compile)
    train_repa.sh       — REPA-aligned training on PDB
  afdb/
    train_baseline.sh   — 60M baseline on AFDB Swiss-Prot (no REPA, torch.compile)
```

## Usage

```bash
# PDB
sbatch hpc-scripts/proteina/training/pdb/train_baseline.sh
sbatch hpc-scripts/proteina/training/pdb/train_repa.sh
sbatch hpc-scripts/proteina/training/pdb/train_repa.sh training_repa_l4_256_per_residue training/256/gearnet/per_residue

# AFDB
sbatch hpc-scripts/proteina/training/afdb/train_baseline.sh
```

Before submitting:

- `DATA_PATH` env var points to your proteina data directory (must contain the relevant LMDB + `metric_factory/model_weights/gearnet_ca.pth` for REPA).
- Log dir exists: `mkdir -p /rds/user/$USER/hpc-work/proteina/logs`.
- LMDB has been built (`data_prep/`) and the length index precomputed (`data_prep/build_length_index.sh`).
