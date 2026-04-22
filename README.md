# molecular-repa

A repository for molecular representation learning.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and requires Python 3.11+.

### Prerequisites

Install uv if you haven't already:
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Quick Start

Clone with submodules:
```bash
git clone --recurse-submodules <repo-url>
```

Or, if you already cloned without submodules:
```bash
git submodule update --init --recursive
```

Then run setup:
```bash
make setup
```

This will:
- Install tabasco and its dependencies (+ dev tools) using uv
- Set up pre-commit hooks with ruff for code linting and formatting

To also install the proteina dependencies (optional):
```bash
make setup-proteina
```

### Development

The project includes the following Make commands:

- `make setup` - Set up the base development environment (core deps + dev tools + pre-commit hooks)
- `make setup-proteina` - Install proteina dependencies and verify the install
- `make lint` - Run ruff linter
- `make format` - Format code with ruff
- `make test` - Run the test suite
- `make check` - Run linter and fix import sorting
- `make clean` - Remove virtual environment and cache files

### Pre-commit Hooks

Pre-commit hooks are automatically installed during setup. They will:
- Run ruff to check and fix code issues
- Format code automatically before each commit

To run pre-commit manually on all files:
```bash
uv run pre-commit run --all-files
```

## Training Tabasco

Tabasco is a flow matching model for molecular generation. Training is managed via Hydra configs.

### Running Training

```bash
uv run python hpc-scripts/tabasco/train_tabasco.py experiment=<experiment_name>
```

### Available Experiments

#### QM9 Dataset

| Experiment | Description |
|------------|-------------|
| `qm9/baseline` | Flow matching without REPA (128-dim, 16-layer, 100 epochs) |
| `qm9/local_baseline` | Smaller baseline for local testing (64-dim, 4-layer, 3 epochs) |
| `qm9/chemprop` | REPA with CheMeleon encoder (λ=0.5, tradeoff mode) |
| `qm9/local_chemprop` | Smaller REPA+CheMeleon variant for local testing |
| `qm9/repa` | REPA with DummyEncoder (control experiment without chemical guidance) |

#### GEOM Dataset

| Experiment | Description |
|------------|-------------|
| `geom/mild` | Baseline without REPA (128-dim, 16-layer) |
| `geom/hot` | Baseline without REPA (256-dim, 16-layer) |
| `geom/spicy` | Baseline without REPA (512-dim, 16-layer) |
| `geom/chemprop_additive` | REPA with CheMeleon encoder (λ=0.8, additive mode) |
| `geom/chemprop_tradeoff` | REPA with CheMeleon encoder (λ=0.8, tradeoff mode) |
| `geom/chemprop_cached` | REPA with pre-computed CheMeleon embeddings (fast lookup) |
| `geom/local_baseline` | Smaller baseline for local testing (64-dim, 4-layer, 3 epochs) |
| `geom/local_chemprop` | Smaller REPA+CheMeleon variant for local testing |

### Examples

```bash
# Quick local test (3 epochs, small model)
uv run python hpc-scripts/tabasco/train_tabasco.py experiment=qm9/local_baseline

# Full baseline training
uv run python hpc-scripts/tabasco/train_tabasco.py experiment=qm9/baseline

# Train with REPA loss (ChemProp encoder)
uv run python hpc-scripts/tabasco/train_tabasco.py experiment=qm9/chemprop
```

### Outputs

Training outputs are saved to `outputs/<date>/<time>/`:
- `checkpoints/` - Model checkpoints (top 3 + last)
- `.hydra/` - Config snapshots
- `train.log` - Training logs

### Resume Training

```bash
uv run python hpc-scripts/tabasco/train_tabasco.py experiment=qm9/baseline ckpt_path=/path/to/checkpoint.ckpt
```

### HPC Notes

**`torch.compile` requires `rhel8/ampere/base` on Wilkes3.** The default `rhel8/default-amp` module set ships gcc binaries compiled for a different CPU microarchitecture, causing SIGILL crashes in Triton's JIT. Loading `rhel8/ampere/base` instead resolves this. All SLURM scripts use `model.compile=true` with the correct module.

To redirect outputs to a high-capacity storage location (recommended — checkpoints can be large), create `src/tabasco/configs/local/default.yaml` on the cluster machine with:

```yaml
# @package _global_
hydra:
  run:
    dir: /path/to/your/storage/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

This file is gitignored and only affects the machine it's created on.

## Proteina

Proteina is a flow matching model for protein backbone (CA-trace) generation, included as a submodule. We extend it with REPA (Representation Alignment) to align the generative model's hidden states with frozen GearNet encoder representations.

### Setup

1. Complete the [Quick Start](#quick-start) steps above (`make setup`)
2. Install proteina dependencies:
   ```bash
   make setup-proteina
   ```
   This installs proteinfoundation and its dependencies (torch-geometric, einops, jax, transformers, etc.) and verifies the install.
3. If you need `mmseqs2` (sequence search):
   ```bash
   conda install -c bioconda mmseqs2
   ```

### PyG C Extension Compatibility

The PyG C extension packages (`torch-scatter`, `torch-sparse`, `torch-cluster`) have known compatibility issues on HPC clusters:

- **Pre-built wheels** from `data.pyg.org` require GLIBC 2.32+, but RHEL 8 clusters have GLIBC 2.28
- **Building from source** requires matching the exact torch C++ ABI, and older package versions (2.1.x) are incompatible with torch 2.9+

We work around this with a **compatibility shim** (`proteinfoundation/repa/pyg_compat.py`) that replaces `torch_scatter` with equivalent native PyTorch ops (`torch.scatter_reduce_`). The shim auto-detects whether the C extensions work and only patches if needed. This is imported automatically by `train_repa.py` and the test suite.

If you want to attempt installing the C extensions anyway (e.g., on a system with GLIBC 2.32+):
```bash
# Check your torch+cuda version
uv run python -c "import torch; print(torch.__version__)"  # e.g. 2.9.1+cu128
# Install pre-built wheels
uv pip install torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.9.1+cu128.html
```

### Training Proteina

Training uses `train_repa.py`, which supports both baseline and REPA modes via config:

```bash
# From src/proteina/proteinfoundation/
python train_repa.py --config_name training_ca_baseline   # baseline (no REPA)
python train_repa.py --config_name training_ca_repa       # with REPA alignment
```

The REPA config aligns transformer hidden states at layer 4 (of 10) with a frozen GearNet CA encoder using cosine similarity loss (λ=0.5, additive mode).

### HPC Training

SLURM scripts are provided for Wilkes3 (A100):

```bash
sbatch hpc-scripts/proteina/training/pdb/train_baseline.sh   # baseline 60M model (PDB)
sbatch hpc-scripts/proteina/training/pdb/train_repa.sh        # REPA-aligned 60M model (PDB)
sbatch hpc-scripts/proteina/training/afdb/train_baseline.sh   # baseline 60M model (AFDB Swiss-Prot)
```

Before submitting, ensure:
- `DATA_PATH` env var points to your data directory (containing PDB data and `metric_factory/model_weights/gearnet_ca.pth`)
- Log directory exists: `mkdir -p /rds/user/$USER/hpc-work/proteina/logs`

### Tests

```bash
PYTHONPATH=src/proteina:$PYTHONPATH uv run python -m pytest tests/proteina/ -v
```

Tests cover the REPA components (projector, loss, hidden state extraction, format conversion). The PyG compat shim is applied automatically, so all 11 tests pass on both login and compute nodes regardless of whether the C extensions work.

### REPA Architecture

The REPA integration adds these modules (all in `src/proteina/proteinfoundation/repa/`):

| Module | Purpose |
|--------|---------|
| `gearnet_encoder.py` | Frozen GearNet CA encoder returning per-residue features [b, n, 512] |
| `protein_transformer_repa.py` | Transformer subclass that captures hidden states at configurable layers |
| `repa_loss.py` | Cosine similarity REPA loss + trainable Projector MLP |
| `proteina_repa.py` | `ProteinaREPA` model subclass integrating REPA into the training loop |
| `pyg_compat.py` | Native PyTorch shim for `torch_scatter` (auto-applied if C extensions fail) |
