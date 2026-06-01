#!/bin/bash
#!
#! SLURM array launcher: one task per checkpoint, in-memory mode.
#!
#! Counterpart to run_pretrained_probe.sh. Differences:
#!   - sbatch --array=0-N runs N+1 tasks in parallel, each handling ONE
#!     checkpoint (taken from the profile's `runs:` list)
#!   - Each task uses --in_memory (no per-checkpoint disk feature cache)
#!     and --jsonl_shard_tag <run> (per-task JSONL shard, no append contention)
#!   - consolidate() at the end of the last task (or via a separate
#!     --consolidate_only invocation) globs all shards into the canonical
#!     CSV/JSON.
#!
#! Why in-memory: at one task per checkpoint, the disk cache (~5-12 GB / ckpt
#! at n=128, ~12-24 GB at n=256) is wasted — features get used once then
#! dropped. RAM holds them transiently (~12-24 GB peak per task, ≤2 layers
#! resident) which fits comfortably on either A100 (80 GB) or a 64 GB CPU
#! node. Aggregate disk savings: ~600 GB across the n=128 + n=256 sweeps.
#!
#! Why one task per checkpoint: model load (~30-60s) + forward (~1-2 min on
#! GPU, ~30-60 min on CPU) is the per-task bulk; the 30 probe fits run on
#! in-memory tensors without re-extracting. Finer fan-out (per-layer or
#! per-probe-kind) would re-pay the model-load + forward each time. Coarser
#! fan-out (multiple ckpts per task) gives up parallelism across the array.
#!
#! Usage:
#!   sbatch --array=0-29%30 hpc-scripts/proteina/evaluation/representation/run_pretrained_probe_array.sh \
#!       --config paper_n128_struct
#!   sbatch --array=0-27%20 hpc-scripts/proteina/evaluation/representation/run_pretrained_probe_array.sh \
#!       --config paper_n256_struct
#!
#! After all tasks complete, consolidate the shards:
#!   python evaluation/proteina/representation/scripts/paper/pretrain_probe_sweep.py \
#!       --config paper_n128_struct --consolidate_only

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --job-name=repa-array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/repa-array-%A_%a.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/repa-array-%A_%a.err
#SBATCH -p ampere
#SBATCH --qos=intr

set -e

#! Parse: this launcher needs --config <profile> to look up the runs list.
#! --lmdb_dataset {pdb,afdb} (default pdb): which source LMDB to read.
PROFILE=""
LMDB_DATASET="pdb"
PY_ARGS=()
i=0
ALL_ARGS=("$@")
while [ $i -lt ${#ALL_ARGS[@]} ]; do
    case "${ALL_ARGS[$i]}" in
        --config) PROFILE="${ALL_ARGS[$((i + 1))]}"; PY_ARGS+=("--config" "$PROFILE"); i=$((i + 2)) ;;
        --lmdb_dataset) LMDB_DATASET="${ALL_ARGS[$((i + 1))]}"; i=$((i + 2)) ;;
        *)        PY_ARGS+=("${ALL_ARGS[$i]}"); i=$((i + 1)) ;;
    esac
done

if [ -z "$PROFILE" ]; then
    echo "ERROR: --config <profile> is required (e.g. paper_n128_struct)" >&2
    exit 2
fi
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: this script must be launched as a SLURM array (sbatch --array=...)" >&2
    exit 2
fi

REPO_DIR="/home/sr2173/git/molecular-repa"

#! Resolve the runs list from sweep_config.yaml at submit time, then index.
RUN_NAME=$(python - <<EOF
import sys, yaml
cfg = yaml.safe_load(open("$REPO_DIR/evaluation/proteina/representation/sweep_config.yaml"))
runs = cfg["$PROFILE"]["runs"].split(",")
idx = int("$SLURM_ARRAY_TASK_ID")
if idx >= len(runs):
    print(f"ERROR: array task id {idx} out of range; profile has {len(runs)} runs", file=sys.stderr)
    sys.exit(2)
print(runs[idx])
EOF
)
if [ -z "$RUN_NAME" ]; then
    echo "ERROR: failed to resolve run name for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
echo "=== Array task $SLURM_ARRAY_TASK_ID -> run '$RUN_NAME' (profile $PROFILE) ==="

#! ── Module / env setup (same as run_pretrained_probe.sh) ──
. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

#! Skip /dev/shm staging entirely — read both LMDBs direct from Lustre.
#! Earlier rounds tried to stage to /dev/shm but accumulated leftover dirs
#! from TLE'd / FAILED tasks (the trap-less rm runs only on success), so
#! /dev/shm filled across all gpu nodes and even 1 GB val.lmdb copies
#! ENOSPC'd. The probe sweep reads each LMDB once at startup (~5000
#! protein lookups, no mmap-heavy training-time access pattern) so direct
#! Lustre is fast enough. Avoids any shm contention.
case "$LMDB_DATASET" in
    pdb)  LUSTRE_LMDB="/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb" ;;
    afdb) LUSTRE_LMDB="/rds/user/sr2173/hpc-work/proteina/data/afdb_swissprot/lmdb" ;;
    *)    echo "ERROR: --lmdb_dataset must be 'pdb' or 'afdb' (got: $LMDB_DATASET)" >&2; exit 2 ;;
esac
echo "=== LMDB dataset: $LMDB_DATASET (source: $LUSTRE_LMDB) ==="
echo "=== Reading both LMDBs direct from Lustre (no /dev/shm staging) ==="
ls -lh "$LUSTRE_LMDB/train.lmdb" "$LUSTRE_LMDB/val.lmdb" 2>&1 | head -2

export PROJECT_ROOT="$REPO_DIR/src/proteina"
export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export PROBES_LMDB_PATH="$LUSTRE_LMDB/val.lmdb"
TRAIN_LMDB_OVERRIDE="$LUSTRE_LMDB/train.lmdb"
#! Stub LOCAL_LMDB so the cleanup at the end is a no-op.
LOCAL_LMDB=""
export OMP_NUM_THREADS=1
#! ProteinMPNN-trained REPA checkpoints re-instantiate their encoder at load
#! time and need this env var to find CA-only weights, else InterpolationError.
#! Affects: repa_mpnn_l4_*, repa_mpnn_l9_*, repa_mpnn_l4_afdb_*.
export PROTEINMPNN_WEIGHTS_DIR="/rds/user/sr2173/hpc-work/proteina/ProteinMPNN"

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) ==="
echo "=== Time: $(date) ==="

cd "$REPO_DIR"

#! Resolve the profile's probe_seed so multi-seed runs write to DISJOINT shards.
#! Two seeds of the same checkpoint share a run name; without a seed-scoped tag
#! they would append to one shard file concurrently and corrupt it.
PROBE_SEED=$(python - <<EOF
import yaml
cfg = yaml.safe_load(open("$REPO_DIR/evaluation/proteina/representation/sweep_config.yaml"))
print(int(cfg["$PROFILE"].get("probe_seed", 42)))
EOF
)
[ -z "$PROBE_SEED" ] && PROBE_SEED=42
#! Sanitise the run name to a filesystem-safe, seed-scoped shard tag.
SHARD_TAG=$(printf '%s_s%s' "$RUN_NAME" "$PROBE_SEED" | tr -c 'A-Za-z0-9_-' '_')

#! Per-task python call: --in_memory + --runs single + --jsonl_shard_tag.
#! --baselines "" overrides the profile's baselines list — baselines are
#! checkpoint-agnostic so running them per-task duplicates work 30× across
#! the array (each baseline source emits up to 30 rows × 7 sources ≈ 70
#! rows × ~65s = 76 min of redundant compute). Run them once in a separate
#! --baselines_only job after the array completes (see launcher docstring).
PROBE_SCRIPT="evaluation/proteina/representation/scripts/paper/pretrain_probe_sweep.py"
EXTRA_ARGS="${PY_ARGS[@]} --runs $RUN_NAME --in_memory --jsonl_shard_tag $SHARD_TAG --baselines \"\" --train_lmdb_path $TRAIN_LMDB_OVERRIDE"
echo "  python call: $PROBE_SCRIPT $EXTRA_ARGS"

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

import shlex
sys.argv = ['$PROBE_SCRIPT'] + shlex.split('$EXTRA_ARGS')
import runpy
runpy.run_path('$PROBE_SCRIPT', run_name='__main__')
"
EXIT=$?

[ -n "$LOCAL_LMDB" ] && rm -rf "$LOCAL_LMDB"
echo "=== DONE task $SLURM_ARRAY_TASK_ID (exit $EXIT), $(date) ==="
exit $EXIT
