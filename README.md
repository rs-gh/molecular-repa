# molecular-repa

A repository for molecular representation learning.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and requires Python 3.12+.

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
