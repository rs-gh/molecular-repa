# evaluation

FID evaluation pipelines for proteina checkpoints. The heavy lifting lives in `evaluation/proteina/generation/scripts/evaluate.py`; these Slurm wrappers stand up the environment and pass the right args.

## Scripts

| File | Role |
|---|---|
| `eval_fid.sh` | Full FID evaluation on one checkpoint (~6,125 samples). Takes `<inference_config_name> [config_subdir] [extra args...]`. |
| `eval_fid_lite_sweep.sh` | Slurm array job: lite FID (~300 samples, ~35 min each) across a list of intermediate checkpoints. Produces convergence curves. Use `--array=0-N` where N = number of checkpoints - 1. |
| `transfer_lite_eval.sh` | Bundle the files needed to run lite FID on a remote server (e.g., workstation with better ProteinMPNN/ESMFold availability) and generate an `eval_lite_remote.sh` to run there. |

## Usage

```bash
# One-shot full FID
sbatch hpc-scripts/proteina/evaluation/eval_fid.sh inference_fid_60m_baseline

# Convergence sweep across 11 checkpoints of baseline + REPA
sbatch --array=0-10 hpc-scripts/proteina/evaluation/eval_fid_lite_sweep.sh baseline
sbatch --array=0-11 hpc-scripts/proteina/evaluation/eval_fid_lite_sweep.sh repa

# Transfer a single checkpoint's lite bundle
bash hpc-scripts/proteina/evaluation/transfer_lite_eval.sh bundle baseline 250000
```

Collected results land under `evaluation/proteina/generation/results/` and are plotted by the scripts in `evaluation/proteina/generation/scripts/`.
