#!/bin/bash
#!
#! SLURM single-task launcher: --baselines_only sweep for one profile.
#!
#! The array launcher (run_pretrained_probe_array.sh) runs ONE checkpoint per
#! task with --baselines "" so baseline rows DON'T duplicate across the array.
#! This script complements it: a single GPU task that iterates ALL baselines
#! for a given profile (untrained_proteina, random_gauss, seq_onehot,
#! trained_noise, knn_dist, local_frame, seqsep_pair). Baselines are
#! checkpoint-agnostic so this runs once per profile.
#!
#! Wall time per profile (approximate, GPU):
#!   n=128: ~50 min  (distance ~60s/layer × 30 baseline rows ≈ 30 min,
#!                    + untrained_proteina forward, + small baselines)
#!   n=256: ~110 min (distance ~175s/layer × 30 baseline rows ≈ 90 min,
#!                    + forward + small baselines)
#!
#! Usage:
#!   sbatch --time=01:00:00 hpc-scripts/proteina/evaluation/representation/run_pretrained_probe_baselines.sh \
#!       --config paper_n128_struct
#!   sbatch --time=02:00:00 hpc-scripts/proteina/evaluation/representation/run_pretrained_probe_baselines.sh \
#!       --config paper_n256_struct

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --job-name=repa-baselines
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/repa-baselines-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/repa-baselines-%j.err
#SBATCH -p ampere

set -e

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

REPO_DIR="/home/sr2173/git/molecular-repa"

#! Resolve the profile's baselines + trained_noise reference run from yaml.
read -r BASELINES TRAINED_NOISE_RUN <<< $(python - <<EOF
import yaml
cfg = yaml.safe_load(open("$REPO_DIR/evaluation/proteina/representation/sweep_config.yaml"))
p = cfg["$PROFILE"]
print(p.get("baselines", ""), p.get("baseline_trained_noise_run", ""))
EOF
)

if [ -z "$BASELINES" ]; then
    echo "ERROR: no baselines configured in profile $PROFILE" >&2
    exit 2
fi
echo "=== Baselines for $PROFILE: $BASELINES (trained_noise_run=$TRAINED_NOISE_RUN) ==="

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

#! Direct Lustre access (same fix as array launcher; no /dev/shm).
case "$LMDB_DATASET" in
    pdb)  LUSTRE_LMDB="/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb" ;;
    afdb) LUSTRE_LMDB="/rds/user/sr2173/hpc-work/proteina/data/afdb_swissprot/lmdb" ;;
    *)    echo "ERROR: --lmdb_dataset must be 'pdb' or 'afdb' (got: $LMDB_DATASET)" >&2; exit 2 ;;
esac
echo "=== LMDB dataset: $LMDB_DATASET (source: $LUSTRE_LMDB) ==="
export PROJECT_ROOT="$REPO_DIR/src/proteina"
export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export PROBES_LMDB_PATH="$LUSTRE_LMDB/val.lmdb"
export OMP_NUM_THREADS=1
export PROTEINMPNN_WEIGHTS_DIR="/rds/user/sr2173/hpc-work/proteina/ProteinMPNN"

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) ==="
echo "=== Time: $(date) ==="

cd "$REPO_DIR"

PROBE_SCRIPT="evaluation/proteina/representation/scripts/paper/pretrain_probe_sweep.py"
EXTRA_ARGS=("${PY_ARGS[@]}" --baselines_only --in_memory \
    --baselines "$BASELINES" \
    --jsonl_shard_tag baselines \
    --train_lmdb_path "$LUSTRE_LMDB/train.lmdb")
if [ -n "$TRAINED_NOISE_RUN" ]; then
    EXTRA_ARGS+=(--baseline_trained_noise_run "$TRAINED_NOISE_RUN")
fi
echo "  python call: $PROBE_SCRIPT ${EXTRA_ARGS[@]}"

#! Use the same shim as the array launcher so module loading + state-dict
#! patching are identical.
EXTRA_ARGS_STR="${EXTRA_ARGS[@]}"
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

import shlex
sys.argv = ['$PROBE_SCRIPT'] + shlex.split('$EXTRA_ARGS_STR')
import runpy
runpy.run_path('$PROBE_SCRIPT', run_name='__main__')
"
EXIT=$?

echo "=== DONE baselines/$PROFILE (exit $EXIT), $(date) ==="
exit $EXIT
