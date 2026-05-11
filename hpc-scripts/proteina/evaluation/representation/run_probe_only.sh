#!/bin/bash
#!
#! SLURM job script for the CPU PROBE-FIT phase of the pretrained-probe
#! sweep. Pair with run_extract_only.sh (GPU node) — features must already
#! be in /rds/.../representation_features/<sweep>/ before this runs.
#!
#! Flow (after run_extract_only.sh):
#!   sbatch run_probe_only.sh --config paper_n128_cath
#!     → reads features from /rds, fits sklearn probes on icelake CPUs
#!     → writes JSONL/CSV under results/pretrained_probe_paper_n128/
#!
#! Why CPU partition: probe fitting is sklearn LogisticRegression, no GPU
#! benefit. icelake has 76 cores/node and the computerlab-sl2-cpu account
#! gets fast queue access. Frees A100s for actual training work.

#SBATCH -A COMPUTERLAB-SL2-CPU
#SBATCH --job-name=repa-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/probe-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/probe-%j.err
#SBATCH -p icelake

set -e

PY_ARGS=("$@")
EXTRA_ARGS="${PY_ARGS[@]}"

. /etc/profile.d/modules.sh
module purge
#! On icelake/cclake the rhel8 base module is `default-icl` (no /base suffix)
#! and the spack tree only has gcc-runtime/13.3.0 (vs 14.3.0 on ampere).
#! gcc 13 added CXXABI_1.3.15 which is what the torch wheels need, so 13.3.0
#! is sufficient. The `/gcc/2sn7kkm3` hash suffix is required because the
#! module isn't a plain alias.
module load rhel8/default-icl
module use /usr/local/software/spack/csd3/spack-modules/cclake-2024-06-01/linux-rocky8-cascadelake
module load gcc-runtime/13.3.0/gcc/2sn7kkm3

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export PROJECT_ROOT="$REPO_DIR/src/proteina"
export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
#! Read LMDBs directly from Lustre — probe-fit reads each LMDB once for
#! labels/masks, no mmap-thrash risk (the issue is repeated random reads
#! during training, not one-shot scans).
export PROBES_LMDB_PATH="/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb/val.lmdb"
#! ProteinMPNN-target REPA runs need the weights dir env var even though we
#! never call MPNN in probe-fit mode — the Hydra resolver runs at config
#! merge time inside iter_jobs. Set it defensively to avoid surprise crashes.
export PROTEINMPNN_WEIGHTS_DIR="/rds/user/sr2173/hpc-work/proteina/ProteinMPNN"
#! CRITICAL: pin BLAS/OMP threads to 1 (matches the GPU wrapper). sklearn
#! LogisticRegression uses lbfgs which is single-threaded anyway; the cost
#! is in the BLAS gemm calls. The numpy openblas wheel is compiled with
#! MAX_THREADS=2 — oversubscribing with OMP_NUM_THREADS=8 caused 28× slower
#! probe fits than the GPU-job sidecar CPU (142s vs 5s per fit, observed
#! on job 29168206 before fix).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== CPUS: $(nproc) ==="
echo "=== Time: $(date) ==="

cd "$REPO_DIR"

PROBE_SCRIPT="evaluation/proteina/representation/scripts/paper/pretrain_probe_sweep.py"
echo "=== Mode: PROBE-FROM-CACHE (CPU phase) ==="

#! Cache must be the same persistent location as run_extract_only.sh wrote to.
SWEEP_TAG=""
prev=""
for a in "${PY_ARGS[@]}"; do
    case "$prev" in
        --config) SWEEP_TAG="$a" ;;
    esac
    prev="$a"
done
if [ -z "$SWEEP_TAG" ]; then
    SWEEP_TAG="adhoc"
fi

PERSISTENT_CACHE_DIR="/rds/user/sr2173/hpc-work/proteina/representation_features/${SWEEP_TAG}"

if [ ! -d "$PERSISTENT_CACHE_DIR" ]; then
    echo "ERROR: feature cache not found at $PERSISTENT_CACHE_DIR"
    echo "       Run run_extract_only.sh --config $SWEEP_TAG first."
    exit 2
fi

if ! echo "$EXTRA_ARGS" | grep -q -- "--cache_dir"; then
    EXTRA_ARGS="$EXTRA_ARGS --cache_dir $PERSISTENT_CACHE_DIR"
fi
if ! echo "$EXTRA_ARGS" | grep -q -- "--probe_from_cache"; then
    EXTRA_ARGS="$EXTRA_ARGS --probe_from_cache"
fi
echo "  cache_dir: $PERSISTENT_CACHE_DIR"
echo "  args:      $EXTRA_ARGS"
echo "=== Cache size: $(du -sh $PERSISTENT_CACHE_DIR | cut -f1) ==="

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

echo "=== DONE (exit $EXIT), $(date) ==="
#! Features stay on /rds — user purges manually with:
#!   rm -rf $PERSISTENT_CACHE_DIR
#! once they're confident no more probe-fit re-runs are needed.
echo "=== To purge features: rm -rf $PERSISTENT_CACHE_DIR ==="
exit $EXIT
