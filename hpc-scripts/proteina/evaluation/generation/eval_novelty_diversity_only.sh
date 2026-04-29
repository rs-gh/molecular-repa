#!/bin/bash
#!
#! Novelty + diversity backfill for existing generation-sweep output directories.
#! CPU-only (TM-score via biotite, no GPU needed).
#!
#! Runs evaluate.py with --skip_generation --skip_fid --designability_subset 0
#! on an existing eval_output/ dir, reads the PDBs already in samples_fid/, and
#! merges _res_novelty_* and _res_diversity_* columns into the existing
#! results_*.csv. Then call --consolidate_only to rebuild sweep_results.jsonl.
#!
#! Wall-time estimates (icelake CPU, per task):
#!   diversity (200-240 PDBs, 1 length bin): O(N^2/2) ~20-30K TM-aligns, ~20-30s
#!   novelty  (200-240 PDBs vs 4441 centroids, with early-exit): ~5-15 min
#!   Total: well within 1h wall limit.
#!
#! Usage:
#!   # Dry-list tasks
#!   bash hpc-scripts/proteina/evaluation/generation/eval_novelty_diversity_only.sh --list
#!
#!   # Backfill all 12 sample-matched checkpoints
#!   sbatch --array=0-11 hpc-scripts/proteina/evaluation/generation/eval_novelty_diversity_only.sh
#!
#!   # Single task
#!   sbatch --array=2-2 hpc-scripts/proteina/evaluation/generation/eval_novelty_diversity_only.sh

#SBATCH -J nov-div-only
#SBATCH -A computerlab-sl2-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/nov-div-%A_%a.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/nov-div-%A_%a.err
#SBATCH -p icelake

set -e

#! --- Task table: mirrors eval_designability_only.sh --- #

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

CENTROID_PATH="/rds/user/sr2173/hpc-work/proteina/data/centroids_pdb.pt"
DIVERSITY_N="${DIVERSITY_N:-200}"
EVAL_SEED="${EVAL_SEED:-42}"

if [ "${1:-}" = "--list" ]; then
    echo "Task list (${#TASKS[@]} tasks, DIVERSITY_N=$DIVERSITY_N, CENTROID=$CENTROID_PATH):"
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

#! --- Environment --- #

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== Novelty + diversity backfill (CPU-only) ==="
echo "=== TASK_ID: $TASK_ID ==="
echo "=== CONFIG_NAME: $CONFIG_NAME ==="
echo "=== OUTPUT_SUFFIX: $OUTPUT_SUFFIX ==="
echo "=== DIVERSITY_N: $DIVERSITY_N, CENTROID_PATH: $CENTROID_PATH ==="
echo "=== NODE: $(hostname), Time: $(date) ==="
echo ""

cd "$REPO_DIR"

python -u -c "
import os, sys
sys.path.insert(0, 'src/proteina/proteinfoundation')
sys.path.insert(0, 'src/proteina')
sys.path.insert(0, 'evaluation/proteina/generation/scripts')

import proteinfoundation.repa.pyg_compat

import torch
_orig = torch.load
def _p(*a, **kw): kw['weights_only'] = False; return _orig(*a, **kw)
torch.load = _p

sys.argv = [
    'evaluate.py',
    '--config_name', '$CONFIG_NAME',
    '--output_suffix', '$OUTPUT_SUFFIX',
    '--skip_generation',
    '--skip_fid',
    '--designability_subset', '0',
    '--diversity_subset_per_bin', '$DIVERSITY_N',
    '--centroid_path', '$CENTROID_PATH',
    '--seed', '$EVAL_SEED',
]
import evaluate
evaluate.main()
"
EXIT=$?

echo ""
echo "=== BACKFILL COMPLETE (exit code: $EXIT), $(date) ==="
exit $EXIT
