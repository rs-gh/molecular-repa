"""P1 — long-range contact prediction (P@L/5) probe (proteina).

Thin wrapper around `linear_probe_contacts` in utils.py. Trains a small MLP
on pair features and reports precision-at-top-L/k for the three canonical
cutoffs (L, L/2, L/5). Long-range means |i − j| ≥ 24, contact threshold 8 Å.
"""

from __future__ import annotations

from typing import Dict

import torch

from utils import ContactResult, linear_probe_contacts


def run_contact_probe(
    reps: torch.Tensor,  # [B, N, D] CPU
    batch: Dict,  # must contain 'coords' (ang, [B, N, 37, 3]) and 'mask' ([B, N]), 'lengths'
) -> ContactResult:
    ca = batch["coords"][:, :, 1, :].cpu()
    mask = batch["mask"].cpu()
    lengths = batch["lengths"].cpu()
    return linear_probe_contacts(reps, ca, mask, lengths)
