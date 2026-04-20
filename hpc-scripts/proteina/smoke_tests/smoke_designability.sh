#!/bin/bash
#!
#! Smoke test: verify designability metric pipeline works end-to-end.
#! Runs batch_designability on 2 generated PDBs (ProteinMPNN -> ESMFold ->
#! scRMSD/TM/pLDDT) and hard-fails on NaN or empty results.
#!
#! Usage: sbatch hpc-scripts/proteina/smoke_tests/smoke_designability.sh
#!

#SBATCH -J smoke-desig
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=intr
#SBATCH --time=00:20:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-desig-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-desig-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

###############################################################
### ProteinMPNN CA weights (same as eval_fid.sh)            ###
###############################################################

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

###############################################################
### HF cache + TMPDIR on RDS (ESMFold needs ~3GB)           ###
###############################################################

# load_esmfold() at designability.py:149-166 reads CACHE_DIR. Compute-node /tmp
# is too small for ESMFold's staged download; xet_get writes to TMPDIR and
# fails with ENOSPC even with HF_HOME set, so disable xet and route TMPDIR to RDS.
export CACHE_DIR="/rds/user/sr2173/hpc-work/proteina/hf_cache"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_HUB_DISABLE_XET=1
export TMPDIR="/rds/user/sr2173/hpc-work/proteina/tmp"
mkdir -p "$CACHE_DIR/hub" "$TMPDIR"

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== PROTEINMPNN_DIR: $PROTEINMPNN_DIR ==="
echo "=== PROTEINMPNN_WEIGHTS_DIR: $PROTEINMPNN_WEIGHTS_DIR ==="
echo "=== HF_HOME: $HF_HOME ==="
echo "=== TMPDIR: $TMPDIR ==="
echo "=== Smoke test: designability pipeline ==="
echo "=== Time: $(date) ==="
echo ""

rm -rf /tmp/torchinductor_${USER} 2>/dev/null
cd "$REPO_DIR"

PDB_DIR="$REPO_DIR/eval_output/inference_fid_60m_baseline/samples_fid"
PDB_1="$PDB_DIR/0_fid.pdb"
PDB_2="$PDB_DIR/1_fid.pdb"
for p in "$PDB_1" "$PDB_2"; do
    if [ ! -f "$p" ]; then
        echo "ERROR: required smoke-test PDB not found: $p"
        exit 2
    fi
done

python -u -c "
import math, os, sys
sys.path.insert(0, 'src/proteina/proteinfoundation')
sys.path.insert(0, 'src/proteina')

import proteinfoundation.repa.pyg_compat  # patches broken torch_scatter/cluster

import torch
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

from proteinfoundation.metrics.designability import batch_designability

pdbs = ['$PDB_1', '$PDB_2']
r = batch_designability(pdbs, tmp_root='/tmp/smoke_desig_${SLURM_JOB_ID}')

print('RESULT:', {k: v for k, v in r.items() if not isinstance(v, list)})

# Hard assertions — batch_designability silently swallows per-PDB exceptions,
# so NaN / empty lists indicate a broken ProteinMPNN or ESMFold setup.
assert len(r['scRMSD_list']) == 2, \
    f'Expected 2 scRMSDs, got {len(r[\"scRMSD_list\"])}: {r[\"scRMSD_list\"]}'
assert not math.isnan(r['scRMSD_mean']), \
    f'scRMSD_mean is NaN (ProteinMPNN or ESMFold failed silently)'
assert not math.isnan(r['tm_score_mean']), 'tm_score_mean is NaN'
assert not math.isnan(r['plddt_mean']), 'plddt_mean is NaN'
assert 0.0 <= r['designability_rate'] <= 1.0, \
    f'designability_rate out of [0,1]: {r[\"designability_rate\"]}'
assert 0.0 <= r['plddt_mean'] <= 100.0, \
    f'plddt_mean out of [0,100]: {r[\"plddt_mean\"]}'
print('SMOKE PASS: designability pipeline OK')
"
EXIT_CODE=$?

echo ""
echo "=== SMOKE TEST COMPLETE (exit code: $EXIT_CODE) ==="
echo "=== Time: $(date) ==="

exit $EXIT_CODE
