.PHONY: setup
setup:
	@echo "Setting up molecular-repa development environment..."
	@command -v uv >/dev/null 2>&1 || { echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/"; exit 1; }
	uv sync
	uv run pre-commit install
	@echo "Setup complete! Virtual environment and pre-commit hooks are ready."

.PHONY: setup-proteina
setup-proteina:
	@echo "Installing proteina dependencies..."
	uv sync --group proteina
	@echo "Verifying proteina installation..."
	uv run python -c "import proteinfoundation; import torch_geometric; print('Proteina installed successfully')"
	@echo "Done! Note: mmseqs2 must be installed separately (conda install -c bioconda mmseqs2)"

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: format
format:
	uv run ruff format .

.PHONY: check
check: lint
	uv run ruff check --select I --fix .

.PHONY: test
test:
	PROJECT_ROOT=$(CURDIR)/src/tabasco uv run python -m pytest tests/ -v

DATA_PATH ?= /rds/user/sr2173/hpc-work/proteina/data

.PHONY: inode-usage
inode-usage:
	@echo "=== RDS inode quota ==="
	@quota 2>/dev/null | grep -E 'Filesystem|rds-d6.*P:' || echo "Could not read quota"
	@echo ""
	@echo "=== File counts ==="
	@for dir in $(DATA_PATH)/pdb_train/raw $(DATA_PATH)/pdb_train/processed $(DATA_PATH)/d_FS/raw $(DATA_PATH)/d_FS/processed; do \
		if [ -d "$$dir" ]; then \
			printf "  %-50s %s files\n" "$$dir" "$$(ls "$$dir" | wc -l)"; \
		fi; \
	done

.PHONY: clean-raw
clean-raw:
	@echo "=== Cleaning raw/processed files to free inodes ==="
	@echo "This will delete raw .cif.gz and legacy .pt files."
	@echo "LMDB files will NOT be touched."
	@read -p "Dataset to clean (pdb/d_FS/all): " ds; \
	if [ "$$ds" = "pdb" ] || [ "$$ds" = "all" ]; then \
		echo "Cleaning PDB..."; \
		rm -rf $(DATA_PATH)/pdb_train/raw && mkdir -p $(DATA_PATH)/pdb_train/raw; \
		rm -rf $(DATA_PATH)/pdb_train/processed && mkdir -p $(DATA_PATH)/pdb_train/processed; \
	fi; \
	if [ "$$ds" = "d_FS" ] || [ "$$ds" = "all" ]; then \
		echo "Cleaning AFDB..."; \
		rm -rf $(DATA_PATH)/d_FS/raw && mkdir -p $(DATA_PATH)/d_FS/raw; \
		rm -rf $(DATA_PATH)/d_FS/processed && mkdir -p $(DATA_PATH)/d_FS/processed; \
	fi; \
	echo "Done."

.PHONY: clean
clean:
	rm -rf .venv
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
