#!/bin/bash
#! SLURM job: CKNNA k-sensitivity sweep for the proteina alignment study.
#!
#! Motivation: the report reads the alignment *pattern* not its magnitude
#! because absolute CKNNA is ~0.02-0.05 (an order of magnitude below image-REPA).
#! CKNNA's k=10 mutual-kNN restriction scores *local* neighbourhood agreement;
#! as k -> N it approaches global CKA. This sweep tests whether the small
#! magnitude is an artifact of the small neighbourhood, or robust to k.
#!
#! Cheap by design: build_batch + extract_features are idempotent and skip on
#! the existing feature cache (results/{model,encoder}_features/*.pt), so only
#! the CKNNA aggregation re-runs, once per k. Each k writes a k-tagged sidecar
#! (cknna_matrix_per_{residue,protein}_k{K}.jsonl); k=10 keeps canonical names.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/evaluation/alignment/run_cknna_ksweep.sh
#!   # override the k grid:
#!   sbatch --export=ALL,CKNNA_K_GRID="50 100 500" run_cknna_ksweep.sh
#!
#! Estimated wall: ~60 min per k (both modes, N=10k residues / 3k proteins,
#! 50 bootstraps). Default grid {50,100,250,500} -> ~4 h. Budgeting 5 h.

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --job-name=cknna-ksweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/cknna-ksweep-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/cknna-ksweep-%j.err
#SBATCH -p ampere

set -e

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base
#! torch_scatter wheels need CXXABI_1.3.15 (gcc 13+); system libstdc++ only has 1.3.11.
module use /usr/local/software/spack/csd3/spack-modules/a100-2025-06-01/linux-rocky8-zen3
module load gcc-runtime/14.3.0

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export PROJECT_ROOT="$REPO_DIR/src/proteina"
export PYTHONUNBUFFERED=1
export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export GEARNET_CKPT_PATH="$DATA_PATH/metric_factory/model_weights/gearnet_ca.pth"
export PROTEINMPNN_WEIGHTS_DIR="/rds/user/sr2173/hpc-work/proteina/ProteinMPNN"
export HF_HOME="/rds/user/sr2173/hpc-work/.cache/huggingface"
mkdir -p "$HF_HOME"

cd "$REPO_DIR"

#! See run_cknna.sh for why the shim launcher is required (torch_scatter ABI
#! + weights_only + _orig_mod. prefix strip).
run_with_shim() {
    local script="$1"
    python -u -c "
import sys
sys.path.insert(0, 'src/proteina/proteinfoundation')
sys.path.insert(0, 'src/proteina')
import proteinfoundation.repa.pyg_compat  # patches sys.modules
import torch
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    result = _orig_torch_load(*args, **kwargs)
    if isinstance(result, dict) and 'state_dict' in result:
        sd = result['state_dict']
        result['state_dict'] = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    return result
torch.load = _patched_load
import runpy
runpy.run_path('$script', run_name='__main__')
"
}

#! Idempotent — both skip on the existing cache, but kept here so the job is
#! self-contained if the cache is ever cleared.
echo "=== build_batch (idempotent) ==="
run_with_shim evaluation/proteina/alignment/scripts/build_batch.py

echo "=== extract_features (idempotent) ==="
run_with_shim evaluation/proteina/alignment/scripts/extract_features.py

CKNNA_K_GRID="${CKNNA_K_GRID:-50 100 250 500}"
echo "=== CKNNA k-sweep over: ${CKNNA_K_GRID} ==="
for K in ${CKNNA_K_GRID}; do
    echo "----- k=${K} -----"
    CKNNA_K="${K}" run_with_shim evaluation/proteina/alignment/scripts/run_cknna.py
done

echo "=== done; sidecar matrices in evaluation/proteina/alignment/results/ ==="
ls -la evaluation/proteina/alignment/results/cknna_matrix_*_k*.jsonl
