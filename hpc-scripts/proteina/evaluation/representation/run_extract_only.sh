#!/bin/bash
#!
#! SLURM job script for the GPU EXTRACTION phase of the pretrained-probe
#! sweep. Pair with run_probe_only.sh (CPU node) to split the pipeline so the
#! GPU portion fits inside intr's 1h wall and avoids the gpu1 queue wait.
#!
#! Flow:
#!   sbatch --qos=intr run_extract_only.sh --config paper_n128_cath
#!     → writes fp16 features to /rds/.../features/<sweep>/
#!     → does NOT fit probes (no probe_kind=cath rows in the JSONL yet)
#!   sbatch run_probe_only.sh --config paper_n128_cath
#!     → reads features back, runs sklearn fits on CPUs, writes JSONL/CSV
#!
#! Why split: probe-fit is single-threaded sklearn (~5 min/ckpt CPU work),
#! while extraction is ~1 min/ckpt GPU work. Keeping them in one job holds
#! an A100 idle for ~80% of the run. Splitting lets us use intr (priority
#! 100k vs gpu1's 5k) for the short GPU phase and any CPU partition for the
#! probe-fit tail.

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --job-name=repa-extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/extract-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/extract-%j.err
#SBATCH -p ampere
#SBATCH --qos=intr

set -e

PY_ARGS=("$@")
EXTRA_ARGS="${PY_ARGS[@]}"

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

#! Stage LMDBs to /dev/shm for fast random access during extraction
#! (Lustre mmap thrashes per feedback_lmdb_local_nvme.md). Eval LMDB needed
#! for the manifests/labels even though we only read it once.
LUSTRE_LMDB="/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb"
LOCAL_LMDB="/dev/shm/proteina_extract_lmdb_${SLURM_JOB_ID:-local}"
mkdir -p "$LOCAL_LMDB"

echo "=== Staging LMDBs to $LOCAL_LMDB ==="
for split in train val; do
    for suffix in ".lmdb" "_keys.pkl" "_lengths.npy"; do
        f="${split}${suffix}"
        if [ -f "$LUSTRE_LMDB/$f" ]; then
            sz=$(du -h "$LUSTRE_LMDB/$f" | cut -f1)
            echo "  copying $f ($sz)"
            cp "$LUSTRE_LMDB/$f" "$LOCAL_LMDB/"
        fi
    done
done

export PROJECT_ROOT="$REPO_DIR/src/proteina"
export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export PROBES_LMDB_PATH="$LOCAL_LMDB/val.lmdb"
#! ProteinMPNN-target REPA runs (repa_mpnn_*) resolve $PROTEINMPNN_WEIGHTS_DIR
#! at checkpoint load time. Without it the ckpt load raises and the row is
#! skipped — costs us one cell in the encoder-ablation block per run.
export PROTEINMPNN_WEIGHTS_DIR="/rds/user/sr2173/hpc-work/proteina/ProteinMPNN"
export OMP_NUM_THREADS=1

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) ==="
echo "=== Time: $(date) ==="

cd "$REPO_DIR"

PROBE_SCRIPT="evaluation/proteina/representation/scripts/paper/pretrain_probe_sweep.py"
echo "=== Mode: EXTRACT-ONLY (GPU phase) ==="

#! Cache MUST be on a persistent filesystem so the downstream probe-fit job
#! can read it. /dev/shm is node-local and dies on job exit. Use /rds with
#! a sweep-name-derived subdir so different sweeps don't collide.
#!
#! The cache subdir name is derived from --config if present, else falls
#! back to a generic name. The user can always override with --cache_dir.
SWEEP_TAG=""
for a in "${PY_ARGS[@]}"; do
    case "$prev" in
        --config) SWEEP_TAG="$a" ;;
    esac
    prev="$a"
done
if [ -z "$SWEEP_TAG" ]; then
    SWEEP_TAG="adhoc_${SLURM_JOB_ID:-local}"
fi

PERSISTENT_CACHE_DIR="/rds/user/sr2173/hpc-work/proteina/representation_features/${SWEEP_TAG}"
mkdir -p "$PERSISTENT_CACHE_DIR"

if ! echo "$EXTRA_ARGS" | grep -q -- "--cache_dir"; then
    EXTRA_ARGS="$EXTRA_ARGS --cache_dir $PERSISTENT_CACHE_DIR"
fi
if ! echo "$EXTRA_ARGS" | grep -q -- "--extract_only"; then
    EXTRA_ARGS="$EXTRA_ARGS --extract_only"
fi
echo "  cache_dir: $PERSISTENT_CACHE_DIR"
echo "  args:      $EXTRA_ARGS"

python -u -c "
import os, sys
sys.path.insert(0, 'src/proteina/proteinfoundation')
sys.path.insert(0, 'src/proteina')

import proteinfoundation.repa.pyg_compat

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
#! Do NOT purge $PERSISTENT_CACHE_DIR - the follow-up CPU probe-fit job
#! needs to read it. Purge happens after run_probe_only.sh consumes it.
echo "=== DONE (exit $EXIT), $(date) ==="
echo "=== Features cached at: $PERSISTENT_CACHE_DIR ==="
echo "=== Next: sbatch run_probe_only.sh ${EXTRA_ARGS/--extract_only/--probe_from_cache} ==="
exit $EXIT
