#!/bin/bash
#! Smoke test — fake dataset to measure GPU-only throughput ceiling
#SBATCH -J prot-fake
#SBATCH -A LIO-CHARM-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr2173@cam.ac.uk
#SBATCH --output=/rds/user/sr2173/hpc-work/proteina/logs/smoke-fake-%j.out
#SBATCH --error=/rds/user/sr2173/hpc-work/proteina/logs/smoke-fake-%j.err
#SBATCH -p ampere

module load rhel8/ampere/base

REPO_DIR="/home/sr2173/git/molecular-repa"
conda deactivate 2>/dev/null || true
source "$REPO_DIR/.venv/bin/activate"

cd "$REPO_DIR/src/proteina/proteinfoundation"

export DATA_PATH="/rds/user/sr2173/hpc-work/proteina/data"

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== Time: $(date) ==="
echo ""

# This script measures the theoretical maximum training speed by replacing
# the real PDB dataset with a fake one that returns pre-computed tensors
# from RAM — zero disk I/O, zero CIF parsing.
#
# Compare the output (seconds/epoch) against the diagnostic baseline to
# quantify the cost of num_workers=0.

python -c "
import os, sys, time
import proteinfoundation.repa.pyg_compat  # noqa: F401

import hydra
import lightning as L
import torch
from torch_geometric.data import Data, Dataset

from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.datasets.base_data import BaseLightningDataModule
from proteinfoundation.utils.ema_utils.ema_callback import EMA
from proteinfoundation.utils.seed_callback import SeedCallback
from proteinfoundation.utils.training_analysis_utils import LogEpochTimeCallback


class FakePDBDataset(Dataset):
    \"\"\"Returns pre-built Data objects from RAM. Zero disk I/O.\"\"\"

    def __init__(self, size=344, max_len=256, transform=None):
        super().__init__(transform=transform)
        self._size = size
        # Pre-build a single template and clone it each time
        # Matches the real PDB batch structure expected by Proteina
        self._template = Data(
            # coords: [n_residues, 37_atom_types, 3_xyz]
            coords=torch.randn(max_len, 37, 3),
            # coord_mask: [n_residues, 37] — which atoms exist
            coord_mask=torch.ones(max_len, 37, dtype=torch.bool),
            # seq: [n_residues] — amino acid indices (0-19)
            seq=torch.randint(0, 20, (max_len,)),
            # res_seq_pdb_idx: [n_residues] — residue indices
            res_seq_pdb_idx=torch.arange(max_len, dtype=torch.long),
            # chain_break_per_res: [n_residues] — binary chain break indicator
            chain_break_per_res=torch.zeros(max_len, dtype=torch.long),
            # Mark all residues as real (no padding)
            num_nodes=max_len,
        )

    def len(self):
        return self._size

    def get(self, idx):
        return self._template.clone()


class FakePDBDataModule(BaseLightningDataModule):
    \"\"\"Drop-in datamodule replacement with fake data.\"\"\"

    def __init__(self, size=344, max_len=256, batch_size=4, transforms=None, **kwargs):
        super().__init__(
            batch_padding=True,
            sampling_mode='random',
            transforms=transforms,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
        )
        self._size = size
        self._max_len = max_len

    def _get_dataset(self, split):
        if split == 'val':
            return FakePDBDataset(size=6, max_len=self._max_len, transform=self.transform)
        return FakePDBDataset(size=self._size, max_len=self._max_len, transform=self.transform)


# --- Load the same experiment config as baseline ---
config_path = '../configs/experiment_config'
with hydra.initialize(config_path, version_base=hydra.__version__):
    cfg_exp = hydra.compose(config_name='smoke_test')
    cfg_exp.hardware.ngpus_per_node_ = 1
    cfg_exp.hardware.nnodes_ = 1
    cfg_exp.run_name_ = 'fake_data_ceiling_test'

# Load transforms from the real config so the model sees the same input format
dataset_config_path = '../configs/datasets_config/pdb'
with hydra.initialize(dataset_config_path, version_base=hydra.__version__):
    cfg_data = hydra.compose(config_name='pdb_smoke_test')

transforms = None
if cfg_data.datamodule.get('transforms'):
    transforms = [hydra.utils.instantiate(t) for t in cfg_data.datamodule.transforms]

torch.set_float32_matmul_precision('medium')
L.seed_everything(42)

datamodule = FakePDBDataModule(
    size=344, max_len=256, batch_size=4, transforms=transforms,
)
model = Proteina(cfg_exp, store_dir='./store/fake_data_ceiling_test')

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
    val_check_interval=9999,  # skip validation for clean timing
    strategy='auto',
    precision='bf16-mixed',
    gradient_clip_algorithm='norm',
    gradient_clip_val=1.0,
)

print('=== FAKE DATA: Training 5 epochs with zero-cost data loading ===')
start = time.time()
trainer.fit(model, datamodule)
elapsed = time.time() - start
epochs = 5
print(f'')
print(f'RESULT: {epochs} epochs in {elapsed:.1f}s ({elapsed/epochs:.1f}s/epoch)')
print(f'This is the GPU-compute ceiling — compare against baseline to measure data loading cost.')
"

echo ""
echo "=== FAKE DATA TEST COMPLETE ==="
echo "=== Time: $(date) ==="
