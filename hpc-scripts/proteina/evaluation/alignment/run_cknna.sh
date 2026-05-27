#!/bin/bash
#! SLURM job script for the proteina CKNNA representation-alignment study.
#! End-to-end at N=10,000 proteins: builds frozen batch (idempotent), extracts
#! per-protein + per-residue features for 5 model rows + 3 encoder columns,
#! computes both CKNNA matrices, and plots.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/evaluation/alignment/run_cknna.sh
#!
#! Estimated wall: ~2-2.5 hours (5 proteina forwards ~50 min + ESM2-150M ~5 min
#! + GearNet/MPNN ~15 min + 300 CKNNA cells ~60 min + plots). Budgeting 4 h.

#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --job-name=cknna-matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/cknna-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/cknna-%j.err
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
#! Proteina configs reference ${oc.env:DATA_PATH} (and friends) via OmegaConf
#! interpolation at checkpoint-load time. Match the env the training runs used.
export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export GEARNET_CKPT_PATH="$DATA_PATH/metric_factory/model_weights/gearnet_ca.pth"
export PROTEINMPNN_WEIGHTS_DIR="/rds/user/sr2173/hpc-work/proteina/ProteinMPNN"
#! HuggingFace cache to /rds so the 650M ESM2 download survives node-local cleanup.
export HF_HOME="/rds/user/sr2173/hpc-work/.cache/huggingface"
mkdir -p "$HF_HOME"

cd "$REPO_DIR"

#! torch_scatter / torch_cluster / torch_sparse have chronic ABI mismatches
#! against this cluster's torch wheel. proteinfoundation.repa.pyg_compat
#! installs pure-torch shims into sys.modules so any subsequent `from
#! torch_scatter import ...` resolves to the shim, not the broken wheel.
#! Must be imported BEFORE anything that pulls in proteina internals — wrap
#! the entry points with a tiny launcher.
run_with_shim() {
    local script="$1"
    python -u -c "
import sys
sys.path.insert(0, 'src/proteina/proteinfoundation')
sys.path.insert(0, 'src/proteina')
import proteinfoundation.repa.pyg_compat  # patches sys.modules

# torch.load defaults to weights_only=True in newer torch; checkpoints contain
# OmegaConf DictConfig and need full pickle. Also strip torch.compile's
# '_orig_mod.' prefix so Proteina.load_from_checkpoint picks up the weights
# (otherwise it silently loads zeros — see project memory).
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

echo "=== build_batch ==="
run_with_shim evaluation/proteina/alignment/scripts/build_batch.py

echo "=== extract_features ==="
run_with_shim evaluation/proteina/alignment/scripts/extract_features.py

echo "=== run_cknna ==="
run_with_shim evaluation/proteina/alignment/scripts/run_cknna.py

echo "=== plot_cknna ==="
run_with_shim evaluation/proteina/alignment/scripts/plot_cknna.py

echo "=== done ==="
