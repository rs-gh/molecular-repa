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

Run the setup command to install dependencies and configure pre-commit hooks:
```bash
make setup
```

This will:
- Install all project dependencies using uv
- Set up pre-commit hooks with ruff for code linting and formatting

### Development

The project includes the following Make commands:

- `make setup` - Set up the development environment
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
uv run python scripts/train_tabasco.py experiment=<experiment_name>
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
uv run python scripts/train_tabasco.py experiment=qm9/local_baseline

# Full baseline training
uv run python scripts/train_tabasco.py experiment=qm9/baseline

# Train with REPA loss (ChemProp encoder)
uv run python scripts/train_tabasco.py experiment=qm9/chemprop
```

### Outputs

Training outputs are saved to `outputs/<date>/<time>/`:
- `checkpoints/` - Model checkpoints (top 3 + last)
- `.hydra/` - Config snapshots
- `train.log` - Training logs

### Resume Training

```bash
uv run python scripts/train_tabasco.py experiment=qm9/baseline ckpt_path=/path/to/checkpoint.ckpt
```

## Proteina

Environment management and training for Proteina is not yet supported but will be added shortly.
