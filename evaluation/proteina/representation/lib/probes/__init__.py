"""Probe heads — each takes ``[B, N, D]`` reps and produces a scalar summary."""

from probelib.probes.cath import CATHResult, run_cath_probe
from probelib.probes.contact import (
    ContactResult,
    MultiSeedResult,
    evaluate_contact_from_scores,
    linear_probe_contacts,
    linear_probe_contacts_multi,
    run_contact_probe,
    run_contact_probe_full,
)

__all__ = [
    "ContactResult",
    "MultiSeedResult",
    "linear_probe_contacts",
    "linear_probe_contacts_multi",
    "run_contact_probe",
    "run_contact_probe_full",
    "evaluate_contact_from_scores",
    "CATHResult",
    "run_cath_probe",
]
