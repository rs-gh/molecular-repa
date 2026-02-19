#!/usr/bin/env python
"""Tabasco training entrypoint for molecular-repa.

This script sets up PROJECT_ROOT and then runs tabasco's training code.
Run from the repo root:
    uv run python scripts/tabasco/train_tabasco.py experiment=qm9/local_chemprop
"""

import os
import subprocess
import sys
from pathlib import Path

# Compute paths
_repo_root = Path(__file__).resolve().parent.parent.parent
_tabasco_root = _repo_root / "src" / "tabasco"
_train_script = _tabasco_root / "src" / "train.py"

# Set PROJECT_ROOT environment variable
env = os.environ.copy()
env["PROJECT_ROOT"] = str(_tabasco_root)

# Run the tabasco train.py with all arguments passed through
result = subprocess.run(
    [sys.executable, str(_train_script)] + sys.argv[1:],
    env=env,
    cwd=_repo_root,  # Run from repo root
)

sys.exit(result.returncode)
