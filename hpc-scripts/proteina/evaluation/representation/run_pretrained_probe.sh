#!/bin/bash
#!
#! SLURM job script for Proteina PRETRAINED-PROBE sweep
#! Wilkes3 (AMD EPYC 7763, ConnectX-6, A100 80GB)
#!
#! Counterpart to run_sweep.sh, but for the pretrained-probe pipeline
#! (pretrain_probe_sweep.py). Key differences:
#!   - Stages BOTH train.lmdb (~51 GB) and val.lmdb (~1.1 GB) to /dev/shm
#!   - Writes to results/pretrained_probe/ (not overwriting the in-place sweep)
#!   - No gearnet/encoder baselines — probe-relevant sources only
#!
#! ALWAYS pass --config pretrained_probe so the canonical N_train / N_eval /
#! probe hyperparams are loaded from sweep_config.yaml. Override individual
#! fields via --n_train, --n_eval, etc.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh \
#!       --config pretrained_probe
#!   sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh \
#!       --config pretrained_probe --runs baseline --steps 400000    # smoke
#!   sbatch hpc-scripts/proteina/evaluation/representation/run_pretrained_probe.sh \
#!       --config pretrained_probe --n_train 5000                    # override
#!
#! Phase 1 (sample-size learning curve) uses a separate entrypoint:
#!   sbatch ... run_pretrained_probe.sh --sample_size

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --job-name=repa-pretrained
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/pretrained-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/pretrained-%j.err
#SBATCH -p ampere

set -e

#! Parse mode flags and drop them from the args passed to Python.
SAMPLE_SIZE=0
PY_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --sample_size) SAMPLE_SIZE=1 ;;
        *)             PY_ARGS+=("$arg") ;;
    esac
done
EXTRA_ARGS="${PY_ARGS[@]}"

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
#! torch_scatter wheels need CXXABI_1.3.15 (gcc 13+); system libstdc++ only has 1.3.11.
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

#! Stage both LMDBs to local tmpfs. /dev/shm is 126 GB on ampere nodes —
#! train.lmdb is ~51 GB so it fits with headroom. Lustre mmap thrashes on
#! compute nodes (feedback_lmdb_local_nvme.md).
LUSTRE_LMDB="/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb"
LOCAL_LMDB="/dev/shm/proteina_pretrained_lmdb_${SLURM_JOB_ID:-local}"
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
#! PROBES_LMDB_PATH is used by checkpoints.py as the default eval LMDB; our
#! scripts also derive the train LMDB path by swapping the stem.
export PROBES_LMDB_PATH="$LOCAL_LMDB/val.lmdb"
export OMP_NUM_THREADS=1

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) ==="
echo "=== Time: $(date) ==="

cd "$REPO_DIR"

if [ "$SAMPLE_SIZE" -eq 1 ]; then
    PROBE_SCRIPT="evaluation/proteina/representation/scripts/sample_size_probe.py"
    echo "=== Mode: sample-size learning curve (Phase 1) ==="
else
    PROBE_SCRIPT="evaluation/proteina/representation/scripts/pretrain_probe_sweep.py"
    echo "=== Mode: pretrained-probe sweep (Phase 2) ==="
    # Per-checkpoint feature cache lives on /dev/shm (tmpfs, 126 GB, job-local)
    # NOT on /home (50 GB quota — blew through it on 2026-04-24). Cache is
    # ephemeral (purged after each checkpoint in-script; dir nuked on job exit
    # below), so tmpfs is ideal. Users can still override with --cache_dir.
    PROBE_CACHE_DIR="/dev/shm/proteina_pretrained_cache_${SLURM_JOB_ID:-local}"
    mkdir -p "$PROBE_CACHE_DIR"
    # Only inject --cache_dir if the user didn't pass one already.
    if ! echo "$EXTRA_ARGS" | grep -q -- "--cache_dir"; then
        EXTRA_ARGS="$EXTRA_ARGS --cache_dir $PROBE_CACHE_DIR"
    fi
    echo "  cache_dir: $PROBE_CACHE_DIR"
fi

#! Same torch.load / pyg_compat shims as run_sweep.sh.
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

sys.argv = ['$PROBE_SCRIPT'] + '$EXTRA_ARGS'.split()
import runpy
runpy.run_path('$PROBE_SCRIPT', run_name='__main__')
"
EXIT=$?

rm -rf "$LOCAL_LMDB"
# Nuke the probe feature cache too (defensive — the sweep script purges after
# each checkpoint, but an OOM or preempt can leave stragglers on tmpfs).
if [ -n "${PROBE_CACHE_DIR:-}" ]; then
    rm -rf "$PROBE_CACHE_DIR"
fi
echo "=== DONE (exit $EXIT), $(date) ==="
exit $EXIT
