#!/bin/bash
#SBATCH -J hf-upload-pkg
#SBATCH -A computerlab-sl3-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/hf-upload-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/hf-upload-%j.err
#SBATCH -p icelake

# Push /rds/.../proteina_offsite_pkg to a private HF model repo.
# hf upload-large-folder is resumable across job restarts (state in .cache/huggingface/).
# 90.8 GB outbound from Cambridge — give it 4h to be safe; rerun if it hits the wall.

set -euo pipefail

cd /home/sr2173/git/molecular-repa
source .venv/bin/activate

# Keep HF resume state on /rds (survives login-node restarts; not subject to /home quota).
export HF_HOME=/rds/user/sr2173/hpc-work/proteina/hf_cache
mkdir -p "$HF_HOME"

# Token lives at ~/.cache/huggingface/token (from `hf auth login` on the login node).
# HF_HOME override above redirects token lookup, so pass it explicitly via HF_TOKEN.
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"

REPO=rscam/proteina-repa-convergence
SRC=/rds/user/sr2173/hpc-work/proteina_offsite_pkg

echo "[$(date -Iseconds)] starting hf upload-large-folder $REPO <- $SRC"
hf upload-large-folder "$REPO" "$SRC" --repo-type=model
echo "[$(date -Iseconds)] done"
