#!/bin/bash
#! Download raw PDB structure files (CIF format) for proteina training.
#! Download only — processing to LMDB is done separately by convert_to_lmdb.sh.
#!
#! Resumable: skips files already in raw/. Retries on connection errors.
#! Requires GPU partition (ampere) because icelake nodes can't resolve PDB servers.
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/download_raw_pdb.sh

#SBATCH -J prot-pdb-dl
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/pdb-dl-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/pdb-dl-%j.err
#SBATCH -p ampere

module load rhel8/ampere/base

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

# Retry loop: the PDB server drops connections after ~10-30k downloads.
# Each retry picks up where the last left off (existing files are skipped).
MAX_RETRIES=10

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "========================================="
    echo "=== Download attempt $attempt / $MAX_RETRIES ==="
    echo "=== Start: $(date) ==="
    echo "========================================="

    python -u -c "
import os, sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import proteinfoundation.repa.pyg_compat  # noqa: F401

import hydra
import pandas as pd
from loguru import logger

data_dir = os.environ['DATA_PATH'] + '/pdb_train'

# Load config to get selector params
config_path = '../configs/datasets_config/pdb'
with hydra.initialize(config_path, version_base=hydra.__version__):
    cfg_data = hydra.compose(config_name='pdb_train_compile')

# Step 1: Create/load the dataset CSV (metadata + selection)
datamodule = hydra.utils.instantiate(cfg_data.datamodule)
file_identifier = datamodule._get_file_identifier(datamodule.dataselector)
csv_name = f'{file_identifier}.csv'
csv_path = datamodule.data_dir / csv_name

if csv_path.exists():
    logger.info(f'CSV exists: {csv_name}')
    df_data = pd.read_csv(csv_path)
else:
    logger.info(f'Creating dataset CSV...')
    df_data = datamodule.dataselector.create_dataset()
    logger.info(f'Dataset: {len(df_data)} entries. Saving CSV before download.')
    df_data.to_csv(csv_path, index=False)
    logger.info(f'CSV saved: {csv_name}')

# Step 2: Download raw structures (skips existing files)
n_raw_before = len(os.listdir(datamodule.raw_dir))
logger.info(f'Raw files before download: {n_raw_before}')
logger.info(f'Downloading structures...')
sys.stdout.flush()
sys.stderr.flush()

datamodule._download_structure_data(df_data['pdb'].tolist())

n_raw_after = len(os.listdir(datamodule.raw_dir))
logger.info(f'Raw files after download: {n_raw_after} (+{n_raw_after - n_raw_before} new)')
"
    EXIT_CODE=$?

    N_RAW=$(ls "$DATA_PATH/pdb_train/raw/" 2>/dev/null | wc -l)
    echo "=== Attempt $attempt finished: exit_code=$EXIT_CODE, raw_files=$N_RAW ==="
    echo ""

    if [ $EXIT_CODE -eq 0 ]; then
        echo "=== Download completed successfully ==="
        break
    fi

    echo "=== Download failed (likely connection error), retrying in 30s... ==="
    sleep 30
done

N_RAW=$(ls "$DATA_PATH/pdb_train/raw/" 2>/dev/null | wc -l)
echo ""
echo "=== DOWNLOAD COMPLETE ==="
echo "=== Total raw files: $N_RAW ==="
echo "=== Time: $(date) ==="
echo ""
echo "Next step: sbatch hpc-scripts/proteina/convert_to_lmdb.sh"
