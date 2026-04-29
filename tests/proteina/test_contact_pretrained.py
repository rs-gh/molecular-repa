"""Tests for the pretrained-split contact probe.

Covers:
  - train_contact_probe returns a trained nn.Module that produces finite logits
  - eval_contact_probe produces P@L/k values in [0, 1] on held-out features
  - the head's output depends on the input reps (no leakage)
  - run_pretrained_contact_probe emits the JSONL-compatible schema
  - save_heads round-trip: torch.save -> torch.load -> re-instantiate via
    _build_head -> re-evaluate -> identical metric

Uses synthetic random-rep features + synthetic CA coordinates (geometric
labels) so the tests run without a checkpoint or real LMDB.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPR_LIB = os.path.join(REPO_ROOT, "evaluation", "proteina", "representation")
sys.path.insert(0, REPR_LIB)


@pytest.fixture
def synthetic_batch():
    """Minimal batch dict + reps matching the shape expected by the probes.

    Uses a random walk for CA coordinates so pair distances have a realistic
    long-range distribution (some contacts, many non-contacts), and modest-rank
    random reps so the probe has something nontrivial to learn from.
    """
    torch.manual_seed(0)
    B, N, D = 8, 80, 32
    coords = torch.cumsum(torch.randn(B, N, 3) * 1.5, dim=1)  # [B, N, 3] A
    full_coords = torch.zeros(B, N, 37, 3)
    full_coords[:, :, 1, :] = coords
    mask = torch.ones(B, N, dtype=torch.bool)
    lengths = torch.full((B,), N, dtype=torch.long)
    U = torch.randn(B, N, 16)
    W = torch.randn(16, D)
    reps = U @ W
    batch = {"coords": full_coords, "mask": mask, "lengths": lengths}
    return reps, batch


def test_train_contact_probe_returns_trained_head(synthetic_batch):
    from lib.probes.contact_pretrained import train_contact_probe

    reps, batch = synthetic_batch
    ca = batch["coords"][:, :, 1, :]
    head = train_contact_probe(
        reps,
        ca,
        batch["mask"],
        batch["lengths"],
        head_type="linear",
        epochs=2,
        device="cpu",
    )
    assert isinstance(head, torch.nn.Module)
    D = reps.shape[-1]
    with torch.no_grad():
        out = head(torch.randn(10, 3 * D))
    assert out.shape == (10, 1)
    assert torch.isfinite(out).all()


def test_eval_contact_probe_metrics_in_range(synthetic_batch):
    from lib.probes.contact_pretrained import (
        eval_contact_probe,
        train_contact_probe,
    )

    reps, batch = synthetic_batch
    ca = batch["coords"][:, :, 1, :]
    head = train_contact_probe(
        reps,
        ca,
        batch["mask"],
        batch["lengths"],
        head_type="linear",
        epochs=2,
        device="cpu",
    )
    result = eval_contact_probe(
        head,
        reps,
        ca,
        batch["mask"],
        batch["lengths"],
        min_length=50,
        device="cpu",
    )
    for v in (result.p_at_L, result.p_at_L_2, result.p_at_L_5):
        assert 0.0 <= v <= 1.0
    assert result.n_proteins_test >= 1


def test_train_eval_disjoint_features(synthetic_batch):
    """Eval metric must depend on the reps fed in, not leak from train reps."""
    from lib.probes.contact_pretrained import (
        eval_contact_probe,
        train_contact_probe,
    )

    reps_train, batch = synthetic_batch
    ca = batch["coords"][:, :, 1, :]
    head = train_contact_probe(
        reps_train,
        ca,
        batch["mask"],
        batch["lengths"],
        head_type="linear",
        epochs=5,
        device="cpu",
        seed=123,
    )
    r_train = eval_contact_probe(
        head, reps_train, ca, batch["mask"], batch["lengths"], device="cpu"
    )
    torch.manual_seed(999)
    reps_random = torch.randn_like(reps_train)
    r_random = eval_contact_probe(
        head, reps_random, ca, batch["mask"], batch["lengths"], device="cpu"
    )
    assert r_train.p_at_L_5 != r_random.p_at_L_5
    assert 0.0 <= r_random.p_at_L_5 <= 1.0


def test_run_pretrained_contact_probe_schema(synthetic_batch):
    from lib.probes.contact_pretrained import run_pretrained_contact_probe

    reps, batch = synthetic_batch
    out = run_pretrained_contact_probe(
        reps_train=reps,
        train_batch=batch,
        reps_eval=reps,
        eval_batch=batch,
        head_type="mlp",
        epochs=2,
        seed=42,
    )
    for k in ("p_at_L", "p_at_L_2", "p_at_L_5", "n_proteins_test"):
        assert k in out
    assert "mlp" in out
    for k in ("p_at_L", "p_at_L_2", "p_at_L_5", "n_proteins_test", "seed"):
        assert k in out["mlp"]
    assert out["mlp"]["seed"] == 42


def test_saved_head_roundtrip(synthetic_batch):
    """Save head state_dict -> reload via _build_head -> metrics match exactly.

    This is the contract the --save_heads sweep flag relies on: stored heads
    must be re-loadable for downstream re-scoring on test.lmdb / weight
    inspection / ensembling without re-running the (run, step, layer) probe.
    """
    from lib.probes.contact import _build_head
    from lib.probes.contact_pretrained import (
        eval_contact_probe,
        train_contact_probe,
    )

    reps, batch = synthetic_batch
    ca = batch["coords"][:, :, 1, :]
    head = train_contact_probe(
        reps,
        ca,
        batch["mask"],
        batch["lengths"],
        head_type="mlp",
        epochs=3,
        device="cpu",
        seed=7,
    )
    r_orig = eval_contact_probe(
        head, reps, ca, batch["mask"], batch["lengths"], device="cpu"
    )

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "head.pt"
        D = reps.shape[-1]
        torch.save(
            {
                "state_dict": head.state_dict(),
                "head_type": "mlp",
                "in_dim": 3 * D,
                "rep_dim": D,
            },
            path,
        )
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        head2 = _build_head(
            bundle["in_dim"], device="cpu", head_type=bundle["head_type"]
        )
        head2.load_state_dict(bundle["state_dict"])
        head2.eval()

    r_reload = eval_contact_probe(
        head2, reps, ca, batch["mask"], batch["lengths"], device="cpu"
    )
    # Floats are deterministic for this BCE+MLP path -> metrics should match exactly.
    assert r_orig.p_at_L_5 == pytest.approx(r_reload.p_at_L_5)
    assert r_orig.p_at_L_2 == pytest.approx(r_reload.p_at_L_2)
    assert r_orig.p_at_L == pytest.approx(r_reload.p_at_L)
