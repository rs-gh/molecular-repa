# ruff: noqa: F401
"""Compatibility shim — forwards to ``probelib.probes.cath``."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from probelib.probes.cath import CATHResult, run_cath_probe  # noqa: E402

__all__ = ["CATHResult", "run_cath_probe"]
