#!/bin/bash
#! Download and process D_FS (Foldseek AFDB clusters) dataset.
#! Two stages:
#!   1. Download 588k PDB files from AlphaFold DB (~10-30 GB)
#!   2. Process raw PDB → .pt files (datamodule.prepare_data)
#!
#! Usage:
#!   sbatch hpc-scripts/proteina/prepare_data_dfs.sh

#SBATCH -J prot-dfs-prep
#SBATCH -A COMPUTERLAB-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/dfs-prep-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/dfs-prep-%j.err
#SBATCH -p icelake

module load rhel8/default

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"
export OMP_NUM_THREADS=1

DFS_DIR="$DATA_PATH/d_FS"
INDEX_FILE="$DFS_DIR/d_FS_index.txt"

echo "=== NODE: $(hostname) ==="
echo "=== Time: $(date) ==="
echo ""

###############################################################
### Stage 1: Download raw PDB files from AlphaFold DB       ###
###############################################################

if [ ! -f "$INDEX_FILE" ]; then
    echo "ERROR: d_FS_index.txt not found at $INDEX_FILE"
    echo "Download from: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/proteina_training_data_indices"
    echo "Place at: $INDEX_FILE"
    exit 1
fi

N_TOTAL=$(wc -l < "$INDEX_FILE")
N_EXISTING=$(ls "$DFS_DIR/raw/"*.pdb 2>/dev/null | wc -l)
echo "=== Index contains $N_TOTAL structures ==="
echo "=== Already downloaded: $N_EXISTING ==="
echo ""

if [ "$N_EXISTING" -lt "$N_TOTAL" ]; then
    echo "=== Stage 1: Downloading PDB files from AlphaFold DB ==="
    echo "=== Start: $(date) ==="
    bash "$REPO_DIR/src/proteina/script_utils/download_afdb_data.sh" "$INDEX_FILE" "$DFS_DIR"
    echo "=== Stage 1 complete: $(date) ==="
    N_DOWNLOADED=$(ls "$DFS_DIR/raw/"*.pdb 2>/dev/null | wc -l)
    echo "=== Downloaded: $N_DOWNLOADED / $N_TOTAL ==="
else
    echo "=== Stage 1: All $N_TOTAL files already downloaded, skipping ==="
fi

echo ""

###############################################################
### Stage 2: Process raw PDB → .pt files                    ###
###############################################################

echo "=== Stage 2: Processing structures ==="
echo "=== Start: $(date) ==="

cd "$REPO_DIR/src/proteina/proteinfoundation"

python -c "
import os, sys, time
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import proteinfoundation.repa.pyg_compat  # noqa: F401

import hydra

# Load the D_FS dataset config
config_path = '../configs/datasets_config/afdb'
with hydra.initialize(config_path, version_base=hydra.__version__):
    cfg_data = hydra.compose(config_name='d_FS')
    # Don't overwrite existing processed files (resume-safe)
    cfg_data.datamodule.overwrite = False

print(f'Data dir: {cfg_data.datamodule.data_dir}')

datamodule = hydra.utils.instantiate(cfg_data.datamodule)

start = time.time()
datamodule.prepare_data()
elapsed = time.time() - start
print(f'')
print(f'Processing complete in {elapsed/3600:.1f} hours')

# Count processed files
data_dir = os.path.expandvars(str(cfg_data.datamodule.data_dir))
processed_dir = os.path.join(data_dir, 'processed')
if os.path.exists(processed_dir):
    n_files = len([f for f in os.listdir(processed_dir) if f.endswith('.pt') or f.endswith('.pkl')])
    print(f'Processed files: {n_files}')
"

echo ""
echo "=== DATA PREPARATION COMPLETE ==="
echo "=== Time: $(date) ==="
