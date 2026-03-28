#!/bin/bash
#! Data preparation job — download and process PDB structures (CPU only, no GPU needed)
#! Run this BEFORE train_baseline.sh to avoid burning GPU hours on data download.
#!
#! At fraction=1.0, max_length=512: ~193k structures to download and process.
#! Estimated time: 12-24 hours depending on PDB server speed.
#! Once the CSV + processed .pt files exist, subsequent training jobs skip this step.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/prepare_data.sh

#SBATCH -J prot-data-prep
#SBATCH -A COMPUTERLAB-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/data-prep-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/data-prep-%j.err
#SBATCH -p icelake

module load rhel8/default

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

cd "$REPO_DIR/src/proteina/proteinfoundation"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="
echo "=== Data path: $DATA_PATH ==="
echo ""

python -c "
import os, sys, time
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import proteinfoundation.repa.pyg_compat  # noqa: F401

import hydra
from loguru import logger

# Load the training dataset config
config_path = '../configs/datasets_config/pdb'
with hydra.initialize(config_path, version_base=hydra.__version__):
    cfg_data = hydra.compose(config_name='pdb_train_compile')

print(f'Dataset config: max_length={cfg_data.datamodule.dataselector.max_length}, '
      f'fraction={cfg_data.datamodule.dataselector.fraction}')

# Instantiate datamodule and run prepare_data (download + process)
datamodule = hydra.utils.instantiate(cfg_data.datamodule)

start = time.time()
print(f'Starting data preparation at {time.strftime(\"%H:%M:%S\")}')
datamodule.prepare_data()
elapsed = time.time() - start
print(f'')
print(f'Data preparation complete in {elapsed/3600:.1f} hours')

# Count processed files
processed_dir = os.path.join(os.environ['DATA_PATH'], 'pdb_train', 'processed')
n_files = len([f for f in os.listdir(processed_dir) if f.endswith('.pt')])
print(f'Processed files: {n_files}')
"

echo ""
echo "=== DATA PREPARATION COMPLETE ==="
echo "=== Time: $(date) ==="
