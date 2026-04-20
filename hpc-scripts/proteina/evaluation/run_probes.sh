#!/bin/bash
#!
#! SLURM job script for Proteina representation-quality probes
#! Wilkes3 (AMD EPYC 7763, ConnectX-6, A100 80GB)
#!
#! Modes:
#!   (default)   run_all.py    → one probe per run (last.ckpt), fast sanity
#!   --sweep     run_sweep.py  → FID-style log-spaced step schedule (11-12 points × 4 runs + gearnet)
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/evaluation/run_probes.sh                        # last.ckpt only
#!   sbatch hpc-scripts/proteina/evaluation/run_probes.sh --sweep                # full sweep
#!   sbatch hpc-scripts/proteina/evaluation/run_probes.sh --sweep --n_proteins 300
#!

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --job-name=repa-probes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/probes-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/probes-%j.err
#SBATCH -p ampere

set -e

#! Parse mode flags and drop them from the args passed to Python.
SWEEP=0
CATH_PATCH=0
PY_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --sweep)       SWEEP=1 ;;
        --cath_patch)  CATH_PATCH=1 ;;
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

#! Lustre mmap is flaky on login nodes and thrashes on compute nodes — always
#! copy the LMDB and length index to the compute node's local NVMe before running.
#! (Matches the pattern in feedback_lmdb_local_nvme.md.)
LUSTRE_LMDB="/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb"
#! Use /dev/shm (tmpfs backed by RAM, ~125 GB on ampere nodes) — train.lmdb
#! may be large and /tmp on some nodes is only 16 GB.
LOCAL_LMDB="/dev/shm/proteina_probe_lmdb_${SLURM_JOB_ID:-local}"
mkdir -p "$LOCAL_LMDB"

echo "=== Copying LMDB to local NVMe at $LOCAL_LMDB ==="
#! Use train.lmdb: it's the ~95% majority of the PDB data. val/test splits are
#! tiny (<1k proteins), and a random sample of train is a fine held-out set for
#! probe training/evaluation (probes are independent models).
SPLIT="${PROBE_SPLIT:-train}"
for f in "${SPLIT}.lmdb" "${SPLIT}_keys.pkl" "${SPLIT}_lengths.npy"; do
    if [ -f "$LUSTRE_LMDB/$f" ]; then
        echo "  copying $f ($(du -h "$LUSTRE_LMDB/$f" | cut -f1))"
        cp "$LUSTRE_LMDB/$f" "$LOCAL_LMDB/"
    else
        echo "  WARNING: $LUSTRE_LMDB/$f not found"
    fi
done
ls -la "$LOCAL_LMDB"

export PROJECT_ROOT="$REPO_DIR/src/proteina"
export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export GEARNET_CKPT_PATH="$DATA_PATH/metric_factory/model_weights/gearnet_ca.pth"
export PROBES_LMDB_PATH="$LOCAL_LMDB/${SPLIT}.lmdb"
export OMP_NUM_THREADS=1

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) ==="
echo "=== Time: $(date) ==="

cd "$REPO_DIR"

#! Install the same pyg_compat + torch.load shims as eval_fid_lite_sweep.sh so
#! OmegaConf / torch.compile / REPA-only key filtering works identically.
if [ "$CATH_PATCH" -eq 1 ]; then
    PROBE_SCRIPT="evaluation/proteina/representation/scripts/patch_cath.py"
    echo "=== Mode: CATH patch (second pass, re-runs CATH only) ==="
elif [ "$SWEEP" -eq 1 ]; then
    PROBE_SCRIPT="evaluation/proteina/representation/scripts/run_sweep.py"
    echo "=== Mode: sweep (FID-schedule) ==="
else
    PROBE_SCRIPT="evaluation/proteina/representation/scripts/run_all.py"
    echo "=== Mode: last.ckpt single-point ==="
fi

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
        # Keep repa_loss keys for ProteinaREPA loading (unlike FID, we *want* the full REPA module).
        result['state_dict'] = sd
    return result
torch.load = _patched_load

sys.argv = ['$PROBE_SCRIPT'] + '$EXTRA_ARGS'.split()
import runpy
runpy.run_path('$PROBE_SCRIPT', run_name='__main__')
"
EXIT=$?

rm -rf "$LOCAL_LMDB"
echo "=== DONE (exit $EXIT), $(date) ==="
exit $EXIT
