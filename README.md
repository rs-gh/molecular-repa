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
uv sync --group proteina
```

### Development

The project includes the following Make commands:

- `make setup` - Set up the development environment (tabasco + dev tools)
- `make setup-proteina` - Install proteina dependencies and verify the install
- `make lint` - Run ruff linter
- `make format` - Format code with ruff
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
uv run python scripts/tabasco/train_tabasco.py experiment=<experiment_name>
```

### Available Experiments

#### QM9 Dataset

| Experiment | Description |
|------------|-------------|
| `qm9/baseline` | Full flow matching model without REPA loss (100 epochs) |
| `qm9/local_baseline` | Smaller baseline for local testing (3 epochs, reduced model size) |
| `qm9/chemprop` | Full model with REPA loss using ChemProp encoder |
| `qm9/local_chemprop` | Smaller REPA variant for local testing |
| `qm9/repa` | REPA variant with default encoder |
| `qm9/mace` | REPA variant using MACE encoder |

#### GEOM Dataset

| Experiment | Description |
|------------|-------------|
| `geom/mild` | Conservative training settings |
| `geom/hot` | Aggressive training settings |
| `geom/spicy` | Most aggressive settings |

### Examples

```bash
# Quick local test (3 epochs, small model)
uv run python scripts/tabasco/train_tabasco.py experiment=qm9/local_baseline

# Full baseline training
uv run python scripts/tabasco/train_tabasco.py experiment=qm9/baseline

# Train with REPA loss (ChemProp encoder)
uv run python scripts/tabasco/train_tabasco.py experiment=qm9/chemprop
```

### Outputs

Training outputs are saved to `outputs/<date>/<time>/`:
- `checkpoints/` - Model checkpoints (top 3 + last)
- `.hydra/` - Config snapshots
- `train.log` - Training logs

### Resume Training

```bash
uv run python scripts/tabasco/train_tabasco.py experiment=qm9/baseline ckpt_path=/path/to/checkpoint.ckpt
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

Proteina is an optional protein design submodule.

### Local Setup (Mac/CPU)

1. Complete the [Quick Start](#quick-start) steps above (`make setup`)
2. Install proteina dependencies and verify:
   ```bash
   make setup-proteina
   ```
   This installs all pip-installable deps including PyTorch Geometric (CPU wheels built from source), then runs a smoke test to confirm the install.
3. If you need `mmseqs2` (sequence search):
   ```bash
   conda install -c bioconda mmseqs2
   ```

### HPC Setup (GPU/CUDA)

1. Complete the [Quick Start](#quick-start) steps above (`make setup`)
2. Load a CUDA module so PyG can link against the right CUDA version:
   ```bash
   module load cuda/12.1  # adjust to your cluster's available version
   ```
3. Install proteina dependencies. Either use the make target (builds PyG from source using the loaded CUDA):
   ```bash
   make setup-proteina
   ```
   Or use pre-built PyG CUDA wheels (faster, avoids compiler issues):
   ```bash
   # Check your torch+cuda version first
   uv run python -c "import torch; print(torch.__version__)"  # e.g. 2.5.1+cu121
   uv sync --group proteina --extra-index-url https://data.pyg.org/whl/torch-2.5.1+cu121.html
   uv run python -c "import proteinfoundation; import torch_geometric; print('OK')"
   ```
4. Load `mmseqs2` via your cluster's module system:
   ```bash
   module load mmseqs2
   ```

> **Note:** `mmseqs2` is a bioconda-only package and cannot be installed via uv.
