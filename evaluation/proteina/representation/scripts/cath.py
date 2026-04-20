# ruff: noqa: F401
"""Compatibility shim — forwards to ``lib.probes.cath``."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent  # .../representation — contains the `lib` package
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from lib.probes.cath import CATHResult, run_cath_probe  # noqa: E402

__all__ = ["CATHResult", "run_cath_probe"]
