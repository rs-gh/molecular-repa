# ruff: noqa: F401
"""Compatibility shim — forwards to ``probelib.probes.contact``."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from probelib.probes.contact import (  # noqa: E402
    ContactResult,
    linear_probe_contacts,
    run_contact_probe,
)

__all__ = ["ContactResult", "linear_probe_contacts", "run_contact_probe"]
