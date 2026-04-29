# ruff: noqa: F401
"""Compatibility shim - forwards to ``lib.probes.contact``."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent  # .../representation - contains the `lib` package
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from lib.probes.contact import (  # noqa: E402
    ContactResult,
    linear_probe_contacts,
    run_contact_probe,
)

__all__ = ["ContactResult", "linear_probe_contacts", "run_contact_probe"]
