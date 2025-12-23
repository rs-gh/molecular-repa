.PHONY: setup
setup:
	@echo "Setting up molecular-repa development environment..."
	@command -v uv >/dev/null 2>&1 || { echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/"; exit 1; }
	uv sync --all-groups
	uv run pre-commit install
	@echo "Setup complete! Virtual environment and pre-commit hooks are ready."

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: format
format:
	uv run ruff format .

.PHONY: check
check: lint
	uv run ruff check --select I --fix .

.PHONY: clean
clean:
	rm -rf .venv
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
