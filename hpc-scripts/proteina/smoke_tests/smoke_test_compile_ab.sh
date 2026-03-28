#!/bin/bash
#! A/B test: torch.compile vs no-compile at max_size=512
#! Runs both back-to-back on the same node, 5 epochs each.
#! First ~1 epoch is warmup (compile overhead); compare epochs 2-5.
#!
#! Time budget (~90 min):
#!   - Data selection + processing for 512 cutoff: ~20 min (new proteins to download/process)
#!   - No-compile (5 epochs at ~40s/epoch): ~4 min
#!   - Compile (warmup ~5-10 min, then 4 epochs at ~30-40s): ~13 min
#!   - Buffer for spawn overhead, GPU init, etc.
#SBATCH -J prot-compile-ab
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/compile-ab-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/compile-ab-%j.err
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

# Background GPU utilization monitor
GPU_LOG="/rds/user/sr2173/hpc-work/proteina/logs/gpu-util-compile-ab-${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
    --format=csv -l 2 > "$GPU_LOG" &
GPU_MON_PID=$!

for MODE in "no-compile" "compile"; do
    echo "========================================"
    echo "=== TEST: ${MODE}, max_size=512, num_workers=2, spawn ==="
    echo "=== Start: $(date) ==="
    echo "========================================"

    if [ "$MODE" = "compile" ]; then
        COMPILE_FLAG="True"
    else
        COMPILE_FLAG="False"
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

compile_mode = ${COMPILE_FLAG}

# Load configs
config_path = '../configs/experiment_config'
with hydra.initialize(config_path, version_base=hydra.__version__):
    cfg_exp = hydra.compose(config_name='training_ca_smoke_test')
    cfg_exp.hardware.ngpus_per_node_ = 1
    cfg_exp.hardware.nnodes_ = 1
    cfg_exp.run_name_ = '${MODE}_512_test'

dataset_config_path = '../configs/datasets_config/pdb'
with hydra.initialize(dataset_config_path, version_base=hydra.__version__):
    cfg_data = hydra.compose(config_name='pdb_smoke_test_512')

print(f'compile={compile_mode}, max_size=512, num_workers={cfg_data.datamodule.num_workers}, start_method={mp.get_start_method()}')

torch.set_float32_matmul_precision('medium')
L.seed_everything(42)

datamodule = hydra.utils.instantiate(cfg_data.datamodule)
model = Proteina(cfg_exp, store_dir='./store/${MODE}_512_test')

if compile_mode:
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

start = time.time()
trainer.fit(model, datamodule)
elapsed = time.time() - start
print(f'')
print(f'RESULT: ${MODE}, 512, 5 epochs in {elapsed:.1f}s ({elapsed/5:.1f}s/epoch)')
"
    EXIT_CODE=$?
    echo "=== End: $(date), exit_code=${EXIT_CODE} ==="
    echo ""

    if [ $EXIT_CODE -ne 0 ]; then
        echo "!!! ${MODE} FAILED — skipping remaining tests"
        break
    fi
done

# Stop GPU monitor
kill $GPU_MON_PID 2>/dev/null
wait $GPU_MON_PID 2>/dev/null

echo "=== GPU utilization & memory summary ==="
python -c "
import csv
with open('$GPU_LOG') as f:
    reader = csv.reader(f)
    header = next(reader)
    utils = []
    mem_used = []
    mem_total = None
    for row in reader:
        try:
            val = int(row[1].strip().replace(' %', ''))
            utils.append(val)
            mem_mb = int(row[2].strip().replace(' MiB', ''))
            mem_used.append(mem_mb)
            if mem_total is None:
                mem_total = int(row[3].strip().replace(' MiB', ''))
        except (ValueError, IndexError):
            pass
if utils:
    print(f'GPU utilization: min={min(utils)}%, max={max(utils)}%, mean={sum(utils)/len(utils):.1f}%, samples={len(utils)}')
if mem_used:
    peak = max(mem_used)
    print(f'GPU memory: peak={peak} MiB / {mem_total} MiB ({100*peak/mem_total:.1f}%)')
    print(f'GPU memory: mean={sum(mem_used)/len(mem_used):.0f} MiB, min={min(mem_used)} MiB')
    print(f'Headroom at peak: {mem_total - peak} MiB ({100*(mem_total - peak)/mem_total:.1f}%)')
if not utils and not mem_used:
    print('No GPU data collected')
"

echo ""
echo "=== COMPILE A/B TEST COMPLETE ==="
echo "=== Time: $(date) ==="
