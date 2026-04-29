#!/bin/bash
#! A/B test: SDPA vs manual attention x compile vs no-compile, at max_size=512
#! Full 2x2 matrix measuring throughput and peak memory.
#!
#! Time budget (~90 min):
#!   - 4 configs x ~6-12 min each + compile warmup + data loading
#SBATCH -J prot-sdpa-ab
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/sdpa-ab-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/sdpa-ab-%j.err
#SBATCH -p ampere

module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

cd "$REPO_DIR/src/proteina/proteinfoundation"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo ""

# 2x2 matrix: (manual, sdpa) x (no-compile, compile)
for ATTN in "manual" "sdpa"; do
    for COMPILE in "no-compile" "compile"; do
        echo "========================================"
        echo "=== TEST: ${ATTN} attention, ${COMPILE}, max_size=512 ==="
        echo "=== Start: $(date) ==="
        echo "========================================"

        if [ "$ATTN" = "sdpa" ]; then
            USE_SDPA="True"
        else
            USE_SDPA="False"
        fi

        if [ "$COMPILE" = "compile" ]; then
            DO_COMPILE="True"
        else
            DO_COMPILE="False"
        fi

        python -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os, sys, time
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import proteinfoundation.repa.pyg_compat  # noqa: F401

import hydra, lightning as L, torch
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.ema_utils.ema_callback import EMA
from proteinfoundation.utils.seed_callback import SeedCallback
from proteinfoundation.utils.training_analysis_utils import LogEpochTimeCallback

use_sdpa = ${USE_SDPA}
do_compile = ${DO_COMPILE}

# Load configs
config_path = '../configs/experiment_config'
with hydra.initialize(config_path, version_base=hydra.__version__):
    cfg_exp = hydra.compose(config_name='smoke_test')
    cfg_exp.hardware.ngpus_per_node_ = 1
    cfg_exp.hardware.nnodes_ = 1
    cfg_exp.run_name_ = '${ATTN}_${COMPILE}_512_test'
    cfg_exp.model.nn.use_sdpa = use_sdpa

dataset_config_path = '../configs/datasets_config/pdb/smoke_test'
with hydra.initialize(dataset_config_path, version_base=hydra.__version__):
    cfg_data = hydra.compose(config_name='pdb_smoke_test_512')

print(f'use_sdpa={use_sdpa}, compile={do_compile}, num_workers={cfg_data.datamodule.num_workers}')

torch.set_float32_matmul_precision('medium')
L.seed_everything(42)

datamodule = hydra.utils.instantiate(cfg_data.datamodule)
model = Proteina(cfg_exp, store_dir='./store/${ATTN}_${COMPILE}_512_test')

if do_compile:
    print('Compiling model.nn with torch.compile (mode=default)')
    model.nn = torch.compile(model.nn)

callbacks = [SeedCallback(), EMA(**cfg_exp.ema), LogEpochTimeCallback()]

trainer = L.Trainer(
    max_epochs=5,
    accelerator='gpu',
    devices=1,
    num_nodes=1,
    callbacks=callbacks,
    logger=False,
    log_every_n_steps=1,
    enable_progress_bar=True,
    check_val_every_n_epoch=None,
    val_check_interval=9999,
    strategy='auto',
    precision='bf16-mixed',
    gradient_clip_algorithm='norm',
    gradient_clip_val=1.0,
)

# Memory tracking
torch.cuda.reset_peak_memory_stats()

start = time.time()
trainer.fit(model, datamodule)
elapsed = time.time() - start

peak_mem = torch.cuda.max_memory_allocated() / 1e9
print(f'')
print(f'RESULT: ${ATTN}, ${COMPILE}, 512, 5 epochs in {elapsed:.1f}s ({elapsed/5:.1f}s/epoch)')
print(f'MEMORY: peak={peak_mem:.2f} GB')
"
        EXIT_CODE=$?
        echo "=== End: $(date), exit_code=${EXIT_CODE} ==="
        echo ""

        if [ $EXIT_CODE -ne 0 ]; then
            echo "!!! ${ATTN}+${COMPILE} FAILED - skipping remaining tests"
            break 2
        fi
    done
done

echo "=== SDPA A/B TEST COMPLETE ==="
echo "=== Time: $(date) ==="
