#!/bin/bash
#! Smoke test — spawn start method + num_workers=1,2,4
#SBATCH -J prot-spawn
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-spawn-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-spawn-%j.err
#SBATCH -p ampere

module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

cd "$REPO_DIR/src/proteina/proteinfoundation"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"

# Suppress nested threading in worker subprocesses
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo "=== /dev/shm ===" && df -h /dev/shm
echo ""

# We use a wrapper script that sets spawn and overrides num_workers via Hydra,
# then runs training for a fixed number of steps.
#
# Test progression: if num_workers=1 segfaults, we stop. If it passes, try 2, then 4.

for NW in 1 2 4; do
    echo "========================================"
    echo "=== TEST: num_workers=${NW}, start_method=spawn ==="
    echo "=== Start: $(date) ==="
    echo "========================================"

    python -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os, sys, time
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Add parent dir to path so graphein_utils is importable (same as train_repa.py)
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

# Now run the actual training
import proteinfoundation.repa.pyg_compat  # noqa: F401

import argparse, hydra, lightning as L, torch
from omegaconf import OmegaConf
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.ema_utils.ema_callback import EMA
from proteinfoundation.utils.seed_callback import SeedCallback
from proteinfoundation.utils.training_analysis_utils import LogEpochTimeCallback

# Load configs
config_path = '../configs/experiment_config'
with hydra.initialize(config_path, version_base=hydra.__version__):
    cfg_exp = hydra.compose(config_name='smoke_test')
    cfg_exp.hardware.ngpus_per_node_ = 1
    cfg_exp.hardware.nnodes_ = 1
    cfg_exp.run_name_ = 'spawn_nw${NW}_test'

dataset_config_path = '../configs/datasets_config/pdb'
with hydra.initialize(dataset_config_path, version_base=hydra.__version__):
    cfg_data = hydra.compose(config_name='pdb_smoke_test')
    cfg_data.datamodule.num_workers = ${NW}

print(f'num_workers={cfg_data.datamodule.num_workers}, start_method={mp.get_start_method()}')

torch.set_float32_matmul_precision('medium')
L.seed_everything(42)

datamodule = hydra.utils.instantiate(cfg_data.datamodule)
model = Proteina(cfg_exp, store_dir='./store/spawn_nw${NW}_test')

callbacks = [SeedCallback(), EMA(**cfg_exp.ema), LogEpochTimeCallback()]

trainer = L.Trainer(
    max_epochs=3,
    accelerator='gpu',
    devices=1,
    num_nodes=1,
    callbacks=callbacks,
    logger=False,
    log_every_n_steps=1,
    enable_progress_bar=True,
    check_val_every_n_epoch=None,
    val_check_interval=9999,  # skip validation
    strategy='auto',
    precision='bf16-mixed',
    gradient_clip_algorithm='norm',
    gradient_clip_val=1.0,
)

start = time.time()
trainer.fit(model, datamodule)
elapsed = time.time() - start
print(f'RESULT: num_workers=${NW}, spawn, 3 epochs in {elapsed:.1f}s')
"
    EXIT_CODE=$?

    echo "=== End: $(date), exit_code=${EXIT_CODE} ==="
    echo ""

    if [ $EXIT_CODE -ne 0 ]; then
        echo "!!! num_workers=${NW} FAILED (exit code ${EXIT_CODE}) — stopping escalation"
        break
    fi
done

echo "=== SPAWN WORKERS TEST COMPLETE ==="
echo "=== Time: $(date) ==="
