"""Probe heads — each takes ``[B, N, D]`` reps and produces a scalar summary."""

from probelib.probes.cath import CATHResult, run_cath_probe
from probelib.probes.contact import (
    ContactResult,
    linear_probe_contacts,
    run_contact_probe,
)

__all__ = [
    "ContactResult",
    "linear_probe_contacts",
    "run_contact_probe",
    "CATHResult",
    "run_cath_probe",
]
