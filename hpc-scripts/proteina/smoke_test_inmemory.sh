#!/bin/bash
#! Smoke test — in_memory=True to measure GPU-only throughput with real data shapes
#SBATCH -J prot-inmem
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-inmem-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-inmem-%j.err
#SBATCH -p ampere

module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

cd "$REPO_DIR/src/proteina/proteinfoundation"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"

# Clear stale checkpoints
rm -rf ./store/proteina_smoke_test_single/checkpoints/last.ckpt 2>/dev/null

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo ""

# Background GPU utilization logging
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 2 > /rds/user/sr2173/hpc-work/proteina/logs/gpu-util-inmem-${SLURM_JOB_ID}.csv &
GPU_MON_PID=$!

echo "=== IN-MEMORY: same real data, zero disk I/O per step ==="
echo "=== Start: $(date) ==="

# Use the same config but override in_memory via a tiny temp config
cat > /tmp/pdb_smoke_test_inmemory_${SLURM_JOB_ID}.yaml <<'YAMLEOF'
datamodule:
  _target_: "proteinfoundation.datasets.pdb_data.PDBLightningDataModule"
  data_dir: ${oc.env:DATA_PATH}/pdb_train/
  in_memory: True
  format: "cif"
  overwrite: False
  batch_padding: True
  sampling_mode: "random"
  transforms:
    - _target_: "proteinfoundation.datasets.transforms.GlobalRotationTransform"
    - _target_: "proteinfoundation.datasets.transforms.ChainBreakPerResidueTransform"
  batch_size: 4
  num_workers: 0
  pin_memory: True

  dataselector:
    _target_: "proteinfoundation.datasets.pdb_data.PDBDataSelector"
    data_dir: ${oc.env:DATA_PATH}/pdb_train/
    fraction: 0.001
    molecule_type: "protein"
    experiment_types: ["diffraction", "EM"]
    min_length: 50
    max_length: 256
    oligomeric_min: null
    oligomeric_max: null
    best_resolution: 0.0
    worst_resolution: 5.0
    has_ligands: []
    remove_ligands: []
    remove_non_standard_residues: True
    remove_pdb_unavailable: True
    exclude_ids: []

  datasplitter:
    _target_: "proteinfoundation.datasets.pdb_data.PDBDataSplitter"
    data_dir: ${oc.env:DATA_PATH}/pdb_train/
    train_val_test: [0.98, 0.019, 0.001]
    split_type: "random"
    split_sequence_similarity: 0.5
    overwrite_sequence_clusters: False
YAMLEOF

# Copy temp config into the datasets_config directory so Hydra can find it
cp /tmp/pdb_smoke_test_inmemory_${SLURM_JOB_ID}.yaml \
   "$REPO_DIR/src/proteina/configs/datasets_config/pdb/pdb_smoke_test_inmemory.yaml"

# Run training with in_memory dataset
python -c "
import os, sys, time
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import proteinfoundation.repa.pyg_compat  # noqa: F401

import hydra, lightning as L, torch
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.ema_utils.ema_callback import EMA
from proteinfoundation.utils.seed_callback import SeedCallback

config_path = '../configs/experiment_config'
with hydra.initialize(config_path, version_base=hydra.__version__):
    cfg_exp = hydra.compose(config_name='training_ca_smoke_test')
    cfg_exp.hardware.ngpus_per_node_ = 1
    cfg_exp.hardware.nnodes_ = 1
    cfg_exp.run_name_ = 'inmemory_baseline_test'

dataset_config_path = '../configs/datasets_config/pdb'
with hydra.initialize(dataset_config_path, version_base=hydra.__version__):
    cfg_data = hydra.compose(config_name='pdb_smoke_test_inmemory')

print(f'in_memory={cfg_data.datamodule.in_memory}, num_workers={cfg_data.datamodule.num_workers}')

torch.set_float32_matmul_precision('medium')
L.seed_everything(42)

datamodule = hydra.utils.instantiate(cfg_data.datamodule)
model = Proteina(cfg_exp, store_dir='./store/inmemory_baseline_test')

callbacks = [SeedCallback(), EMA(**cfg_exp.ema)]

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

print('Loading all data into RAM...')
start = time.time()
trainer.fit(model, datamodule)
elapsed = time.time() - start
print(f'')
print(f'RESULT: 5 epochs in {elapsed:.1f}s ({elapsed/5:.1f}s/epoch)')
print(f'Compare against diagnostic baseline (~10s/epoch from disk) to measure I/O cost.')
"

echo "=== End: $(date) ==="

# Stop GPU monitor and summarize
kill $GPU_MON_PID 2>/dev/null
wait $GPU_MON_PID 2>/dev/null

echo ""
echo "=== GPU utilization summary ==="
python -c "
import csv
with open('/rds/user/sr2173/hpc-work/proteina/logs/gpu-util-inmem-${SLURM_JOB_ID}.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    utils = []
    for row in reader:
        try:
            val = int(row[1].strip().replace(' %', ''))
            utils.append(val)
        except (ValueError, IndexError):
            pass
if utils:
    print(f'GPU utilization: min={min(utils)}%, max={max(utils)}%, mean={sum(utils)/len(utils):.1f}%, samples={len(utils)}')
    idle = sum(1 for u in utils if u == 0)
    print(f'Time at 0% (idle/loading): {idle}/{len(utils)} samples ({100*idle/len(utils):.1f}%)')
    high = sum(1 for u in utils if u >= 80)
    print(f'Time at 80%+: {high}/{len(utils)} samples ({100*high/len(utils):.1f}%)')
else:
    print('No GPU utilization data collected')
"

# Clean up temp config
rm -f "$REPO_DIR/src/proteina/configs/datasets_config/pdb/pdb_smoke_test_inmemory.yaml"

echo ""
echo "=== IN-MEMORY TEST COMPLETE ==="
echo "=== Time: $(date) ==="
