#!/bin/bash
#!
#! SLURM job script for building the CATH-classifier artifact consumed by
#! the generation evaluation suite (compute_cath_metrics in evaluate.py).
#!
#! Single-shot job: loads N_train proteins from train.lmdb, embeds them with
#! the same NoTrainCAGearNet checkpoint the FID metric uses, fits a linear
#! logistic-regression CATH head, validates on val.lmdb, and writes a pickle
#! bundle (head + train_meta + train_dist + sidecar JSON) under
#! evaluation/proteina/representation/results/cath_classifier/.
#!
#! Wall-time budget: ~30 min on intr GPU at default n_train=5000.
#!
#! Usage:
#!   #! Smoke (200 train / 100 eval, ~5 min):
#!   sbatch hpc-scripts/proteina/evaluation/representation/build_cath_classifier.sh \
#!       --n_train 200 --n_eval 100
#!
#!   #! Production (5000 / 500):
#!   sbatch hpc-scripts/proteina/evaluation/representation/build_cath_classifier.sh
#!
#!   #! Per-level breakdown (rerun once per level into a different output):
#!   for L in C A T H; do
#!     sbatch ... build_cath_classifier.sh --cath_level $L \
#!         --output evaluation/proteina/representation/results/cath_classifier/cath_gearnet_${L}.pkl
#!   done

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --job-name=cath-clf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --qos=intr
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/cath-clf-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/cath-clf-%j.err
#SBATCH -p ampere

set -e

EXTRA_ARGS="$@"

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
#! torch_scatter wheels need CXXABI_1.3.15 (gcc 13+); system libstdc++ only has 1.3.11.
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

#! Stage both LMDBs to local tmpfs. /dev/shm is 126 GB on ampere nodes, train.lmdb
#! is ~51 GB. Lustre mmap thrashes on compute nodes (feedback_lmdb_local_nvme.md).
LUSTRE_LMDB="/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb"
LOCAL_LMDB="/dev/shm/proteina_cath_clf_lmdb_${SLURM_JOB_ID:-local}"
mkdir -p "$LOCAL_LMDB"

echo "=== Copying LMDB sidecars + files to $LOCAL_LMDB ==="
for split in train val; do
    for suffix in ".lmdb" "_keys.pkl" "_lengths.npy"; do
        f="${split}${suffix}"
        if [ -f "$LUSTRE_LMDB/$f" ]; then
            sz=$(du -h "$LUSTRE_LMDB/$f" | cut -f1)
            echo "  copying $f ($sz)"
            cp "$LUSTRE_LMDB/$f" "$LOCAL_LMDB/"
        else
            echo "  WARNING: $LUSTRE_LMDB/$f not found"
        fi
    done
done
ls -la "$LOCAL_LMDB"

export PROJECT_ROOT="$REPO_DIR/src/proteina"
export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export PROBES_LMDB_PATH="$LOCAL_LMDB/val.lmdb"
export GEARNET_CA_CKPT="$DATA_PATH/metric_factory/model_weights/gearnet_ca.pth"
export OMP_NUM_THREADS=1

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) ==="
echo "=== Time: $(date) ==="

cd "$REPO_DIR"

#! Inject train/eval LMDB paths from /dev/shm if user didn't override.
if ! echo "$EXTRA_ARGS" | grep -q -- "--train_lmdb_path"; then
    EXTRA_ARGS="$EXTRA_ARGS --train_lmdb_path $LOCAL_LMDB/train.lmdb"
fi
if ! echo "$EXTRA_ARGS" | grep -q -- "--eval_lmdb_path"; then
    EXTRA_ARGS="$EXTRA_ARGS --eval_lmdb_path $LOCAL_LMDB/val.lmdb"
fi

echo "=== Args: $EXTRA_ARGS ==="

#! Same torch.load / pyg_compat shims as run_pretrained_probe.sh.
SCRIPT="evaluation/proteina/representation/scripts/build_cath_classifier.py"
python -u -c "
import os, sys
sys.path.insert(0, 'src/proteina/proteinfoundation')
sys.path.insert(0, 'src/proteina')

import proteinfoundation.repa.pyg_compat  # replaces torch_scatter/cluster with pure-torch

import torch
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    result = _orig_torch_load(*args, **kwargs)
    if isinstance(result, dict) and 'state_dict' in result:
        sd = result['state_dict']
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        result['state_dict'] = sd
    return result
torch.load = _patched_load

sys.argv = ['$SCRIPT'] + '$EXTRA_ARGS'.split()
import runpy
runpy.run_path('$SCRIPT', run_name='__main__')
"
EXIT=$?

rm -rf "$LOCAL_LMDB"
echo "=== DONE (exit $EXIT), $(date) ==="
exit $EXIT
