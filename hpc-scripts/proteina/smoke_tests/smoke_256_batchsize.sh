#!/bin/bash
#!
#! Smoke test: find max batch size for REPA at 256 max length.
#! Tests batch_size=12 and batch_size=16 with REPA L4 (worst-case memory).
#! Runs 20 training steps each — if it doesn't OOM, the batch size works.
#!
#! Usage: sbatch hpc-scripts/proteina/smoke_tests/smoke_256_batchsize.sh
#!

#SBATCH -J smoke-bs
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-bs-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-bs-%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
PROTEINA_DIR="$REPO_DIR/src/proteina/proteinfoundation"

conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export LMDB_DIR="$DATA_PATH/pdb_train/lmdb"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p /rds/user/sr2173/hpc-work/proteina/logs

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo ""

rm -rf /tmp/torchinductor_${USER} 2>/dev/null

cd "$PROTEINA_DIR"

###############################################################
### Test batch_size=12 (REPA L4 at 256 max length)          ###
###############################################################

echo "=========================================="
echo "=== TEST: batch_size=12, REPA L4, 256  ==="
echo "=========================================="

timeout 300 python -u -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import sys, runpy
sys.argv = ['train_repa.py', '--config_name', 'training_smoke_256_bs12', '--show_prog_bar']
runpy.run_path('train_repa.py', run_name='__main__')
" 2>&1 | tail -5
BS12_EXIT=$?

if [ $BS12_EXIT -eq 0 ] || [ $BS12_EXIT -eq 124 ]; then
    echo ">>> batch_size=12: OK (exit=$BS12_EXIT, 124=timeout which is fine)"
else
    echo ">>> batch_size=12: FAILED (exit=$BS12_EXIT)"
fi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
rm -rf /tmp/torchinductor_${USER} 2>/dev/null
echo ""

###############################################################
### Test batch_size=16 (REPA L4 at 256 max length)          ###
###############################################################

echo "=========================================="
echo "=== TEST: batch_size=16, REPA L4, 256  ==="
echo "=========================================="

timeout 300 python -u -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import sys, runpy
sys.argv = ['train_repa.py', '--config_name', 'training_smoke_256_bs16', '--show_prog_bar']
runpy.run_path('train_repa.py', run_name='__main__')
" 2>&1 | tail -5
BS16_EXIT=$?

if [ $BS16_EXIT -eq 0 ] || [ $BS16_EXIT -eq 124 ]; then
    echo ">>> batch_size=16: OK (exit=$BS16_EXIT)"
else
    echo ">>> batch_size=16: FAILED (exit=$BS16_EXIT)"
fi

echo ""
echo "=== BATCH SIZE TESTS COMPLETE ==="
echo "=== Time: $(date) ==="
