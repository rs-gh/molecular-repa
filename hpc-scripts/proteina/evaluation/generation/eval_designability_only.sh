#!/bin/bash
#!
#! Designability-only backfill for existing generation-sweep output directories.
#!
#! Runs evaluate.py with --skip_generation --skip_fid on an existing eval_output/
#! dir, reads the PDBs already in samples_fid/, runs ProteinMPNN -> ESMFold ->
#! scRMSD/TM/pLDDT on a random subset, and merges the resulting _res_* columns
#! into the existing results_*.csv (evaluate.py:584 handles the merge).
#!
#! Array-driven: one task per (config_name, output_suffix) pair. Edit the
#! TASKS array below to control which runs get backfilled.
#!
#! Usage:
#!   # Dry-list tasks
#!   bash hpc-scripts/proteina/evaluation/generation/eval_designability_only.sh --list
#!
#!   # Backfill all 12 sample-matched checkpoints (N=100 each)
#!   sbatch --array=0-11 hpc-scripts/proteina/evaluation/generation/eval_designability_only.sh
#!
#!   # Single task (headline rerun at N=500)
#!   DESIG_N=500 sbatch --array=5-5 hpc-scripts/proteina/evaluation/generation/eval_designability_only.sh
#!
#!   # Submit under a different account (see "FairShare" note below)
#!   sbatch --array=0-11 -A computerlab-sl2-gpu hpc-scripts/.../eval_designability_only.sh
#!
#! Measured wall times (A100, N=100, 800 ESMFold forwards per task):
#!   n=128:    ~30-40 min per task  (seq len 128)
#!   n=256:    ~70 min per task     (seq len 256; hit 1h wall on first run)
#!   n=512_sm: ~70-90 min estimate  (pending; ESMFold ~linear in seq length)
#! Wall limit is 2h to give headroom for n=512_sm; n=128-only batches could
#! drop to 1h for better queue priority.
#!
#! FairShare: LIO-CHARM-SL2-GPU has been heavily oversubscribed by the group
#! (fairshare factor ≈ 0.0 as of 2026-04-24). Submitting with
#!   -A computerlab-sl2-gpu
#! uses a less-loaded account (fairshare ≈ 0.07) and starts sooner.
#! Check with: `sshare -U -u $USER -o account,fairshare,rawusage`
#!
#SBATCH -J desig-only
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/desig-only-%A_%a.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/desig-only-%A_%a.err
#SBATCH -p ampere

set -e

#! --- Task table: (config_name, output_suffix) pairs --- #
#! These reproduce the 12 eval_output/inference_inference_fid_60m_*_sweep_*_step_* dirs
#! from the n128 / n256 / n512_sm sample-matched sweeps.
#!
#! Status (2026-04-29):
#!   tasks 0-7 (n=128, n=256): done at DESIG_N=100
#!   tasks 8, 10 (n=512 baseline, repa_l4): done at DESIG_N=50 (~2h30m wall)
#!   TODO tasks 9, 11 (n=512 repa_l0, repa_l9): submit with DESIG_N=50, 3h wall:
#!     DESIG_N=50 sbatch --array=9,11 --time=03:00:00 hpc-scripts/proteina/evaluation/generation/eval_designability_only.sh
#!   At n=512, DESIG_N=100 does not fit in 4h wall (ESMFold ~3 min/protein).

TASKS=(
    "inference/inference_fid_60m_baseline_128_lite|sweep_baseline_128_step_800000"
    "inference/inference_fid_60m_repa_l0_128_lite|sweep_repa_l0_128_step_400000"
    "inference/inference_fid_60m_repa_l4_128_lite|sweep_repa_l4_128_step_400000"
    "inference/inference_fid_60m_repa_l9_128_lite|sweep_repa_l9_128_step_400000"
    "inference/inference_fid_60m_baseline_256_lite|sweep_baseline_256_step_400000"
    "inference/inference_fid_60m_repa_l0_256_lite|sweep_repa_l0_256_step_400000"
    "inference/inference_fid_60m_repa_l4_256_lite|sweep_repa_l4_256_step_400000"
    "inference/inference_fid_60m_repa_l9_256_lite|sweep_repa_l9_256_step_400000"
    "inference/inference_fid_60m_baseline_512_sm_lite|sweep_baseline_512_sm_step_500000"
    "inference/inference_fid_60m_repa_l0_512_sm_lite|sweep_repa_l0_512_sm_step_750000"
    "inference/inference_fid_60m_repa_l4_512_sm_lite|sweep_repa_l4_512_sm_step_750000"
    "inference/inference_fid_60m_repa_l9_512_sm_lite|sweep_repa_l9_512_sm_step_750000"
)

DESIG_N="${DESIG_N:-100}"
EVAL_SEED="${EVAL_SEED:-42}"

if [ "${1:-}" = "--list" ]; then
    echo "Task list (${#TASKS[@]} tasks, DESIG_N=$DESIG_N):"
    for i in "${!TASKS[@]}"; do
        IFS='|' read -r cfg suf <<< "${TASKS[$i]}"
        slug="${cfg//\//_}"
        dir="eval_output/${slug}_${suf}"
        exists="MISSING"; [ -d "$dir/samples_fid" ] && exists="OK ($(ls $dir/samples_fid 2>/dev/null | wc -l) pdbs)"
        printf "  %2d  %-55s  %-35s  [%s]\n" "$i" "$cfg" "$suf" "$exists"
    done
    exit 0
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [ -z "$TASK_ID" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set. Submit via: sbatch --array=0-$((${#TASKS[@]}-1)) $0"
    exit 2
fi
if [ "$TASK_ID" -ge "${#TASKS[@]}" ]; then
    echo "ERROR: task_id=$TASK_ID out of range (0..$((${#TASKS[@]}-1)))"
    exit 2
fi

IFS='|' read -r CONFIG_NAME OUTPUT_SUFFIX <<< "${TASKS[$TASK_ID]}"

#! --- Environment (mirrors eval_fid.sh / smoke_designability.sh) --- #

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

#! ProteinMPNN CA weights (same stage as eval_fid.sh)
CA_WEIGHTS_ROOT="/rds/user/sr2173/hpc-work/proteina/ProteinMPNN"
mkdir -p "$CA_WEIGHTS_ROOT/ca_model_weights"
for v in v_48_002 v_48_010 v_48_020; do
    f="$CA_WEIGHTS_ROOT/ca_model_weights/${v}.pt"
    if [ ! -f "$f" ]; then
        echo "Downloading ProteinMPNN CA weights: ${v}.pt"
        wget -q -O "$f" "https://github.com/dauparas/ProteinMPNN/raw/main/ca_model_weights/${v}.pt"
    fi
done
export PROTEINMPNN_DIR="$REPO_DIR/src/proteina/ProteinMPNN"
export PROTEINMPNN_WEIGHTS_DIR="$CA_WEIGHTS_ROOT"

#! HF cache + TMPDIR on RDS (ESMFold needs ~3GB; xet writes to TMPDIR)
export CACHE_DIR="/rds/user/sr2173/hpc-work/proteina/hf_cache"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_HUB_DISABLE_XET=1
export TMPDIR="/rds/user/sr2173/hpc-work/proteina/tmp"
mkdir -p "$CACHE_DIR/hub" "$TMPDIR"

rm -rf /tmp/torchinductor_${USER} 2>/dev/null

echo "=== Designability-only backfill ==="
echo "=== TASK_ID: $TASK_ID ==="
echo "=== CONFIG_NAME: $CONFIG_NAME ==="
echo "=== OUTPUT_SUFFIX: $OUTPUT_SUFFIX ==="
echo "=== DESIG_N: $DESIG_N ==="
echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) ==="
echo "=== Time: $(date) ==="
echo ""

cd "$REPO_DIR"

#! Run evaluate.py with --skip_generation --skip_fid; merges into existing CSV
python -u -c "
import os, sys
sys.path.insert(0, 'src/proteina/proteinfoundation')
sys.path.insert(0, 'src/proteina')
sys.path.insert(0, 'evaluation/proteina/generation/scripts')

import proteinfoundation.repa.pyg_compat  # patches torch_scatter/cluster

import torch
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

sys.argv = [
    'evaluate.py',
    '--config_name', '$CONFIG_NAME',
    '--output_suffix', '$OUTPUT_SUFFIX',
    '--skip_generation',
    '--skip_fid',
    '--designability_subset', '$DESIG_N',
    '--diversity_subset_per_bin', '0',
    '--seed', '$EVAL_SEED',
]
import evaluate
evaluate.main()
"
EXIT=$?

echo ""
echo "=== BACKFILL COMPLETE (exit code: $EXIT) ==="
echo "=== Time: $(date) ==="

exit $EXIT
