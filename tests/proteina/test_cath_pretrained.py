"""Tests for the pretrained-split CATH probe.

Covers:
  - train_cath_probe fits a linear head on synthetic separable reps
  - eval_cath_probe reports accuracy/macro_f1 in [0, 1] and finite confidence
  - OOV bookkeeping: eval-only classes are reported via n_eval_oov, not silently dropped
  - run_pretrained_cath_probe emits the JSONL-compatible flat schema
  - too-few-classes path returns NaN row instead of crashing
  - MLP head path runs end-to-end
  - Round-trip pickle of the linear head preserves predictions

Synthetic reps are linearly-separable per-class so a linear probe should hit
near-perfect accuracy in seconds.
"""

import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPR_LIB = os.path.join(REPO_ROOT, "evaluation", "proteina", "representation")
sys.path.insert(0, REPR_LIB)


def _make_data(cath_codes):
    """Build a list of torch_geometric.data.Data with a cath_code attribute."""
    from torch_geometric.data import Data

    out = []
    for cc in cath_codes:
        d = Data()
        d.cath_code = cc
        out.append(d)
    return out


def _make_separable_reps(
    labels: np.ndarray, n_classes: int, dim: int = 32, seed: int = 0
):
    """Return reps [B, dim] with one fixed per-class centroid + small noise."""
    rng = np.random.RandomState(seed)
    centroids = rng.randn(n_classes, dim) * 5.0
    noise = rng.randn(len(labels), dim) * 0.3
    reps = centroids[labels] + noise
    return torch.from_numpy(reps).float()


@pytest.fixture
def synthetic_cath_batch():
    """30 proteins, three CATH topologies (T-level), linearly separable reps."""
    rng = np.random.RandomState(0)
    n_per_class = 10
    classes = ["1.10.10", "2.40.50", "3.30.70"]
    labels = np.repeat(np.arange(len(classes)), n_per_class)
    rng.shuffle(labels)
    cath_codes = [classes[lab] for lab in labels]  # plain string shape
    raw = _make_data(cath_codes)
    reps = _make_separable_reps(labels, n_classes=len(classes), dim=24, seed=1)
    # 2D reps (already pooled) — train_cath_probe accepts these directly.
    return reps, raw, labels, classes


def test_train_cath_probe_linear_separable(synthetic_cath_batch):
    """Linear probe on perfectly-separable reps should hit ~100% on the same set."""
    from lib.probes.cath_pretrained import (
        eval_cath_probe,
        train_cath_probe,
    )

    reps, raw, _, _ = synthetic_cath_batch
    head, meta = train_cath_probe(
        reps, mask=None, raw=raw, level="T", head_type="linear"
    )
    assert head is not None
    assert meta["level"] == "T"
    assert len(meta["vocab"]) == 3
    assert meta["in_dim"] == reps.shape[1]
    assert meta["n_train"] == len(raw)

    res = eval_cath_probe(head, meta, reps, mask=None, raw=raw)
    assert res.n_eval == len(raw)
    assert res.n_eval_in_vocab == len(raw)
    assert res.n_eval_oov == 0
    assert 0.95 <= res.accuracy <= 1.0
    assert 0.95 <= res.macro_f1 <= 1.0
    assert 0.0 <= res.top1_conf_mean <= 1.0
    assert res.n_classes == 3


def test_oov_bookkeeping(synthetic_cath_batch):
    """Eval proteins with classes not seen at train time go into n_eval_oov."""
    from lib.probes.cath_pretrained import (
        eval_cath_probe,
        train_cath_probe,
    )

    reps_train, raw_train, _, _ = synthetic_cath_batch
    head, meta = train_cath_probe(
        reps_train, mask=None, raw=raw_train, level="T", head_type="linear"
    )

    # Build an eval set that adds a fourth class never seen at train time.
    rng = np.random.RandomState(7)
    extra_centroid = rng.randn(reps_train.shape[1]) * 5.0
    extra_reps = torch.from_numpy(
        extra_centroid + rng.randn(5, reps_train.shape[1]) * 0.3
    ).float()
    extra_codes = ["9.99.99"] * 5

    reps_eval = torch.cat([reps_train, extra_reps], dim=0)
    raw_eval = raw_train + _make_data(extra_codes)

    res = eval_cath_probe(head, meta, reps_eval, mask=None, raw=raw_eval)
    assert res.n_eval == len(raw_eval)
    assert res.n_eval_oov == 5
    assert res.n_eval_in_vocab == len(raw_train)
    # Confidence is reported across all labelled proteins, including OOV.
    assert 0.0 <= res.top1_conf_mean <= 1.0


def test_unlabelled_dropped_from_n_eval(synthetic_cath_batch):
    """Proteins with no cath_code at all don't count toward n_eval."""
    from lib.probes.cath_pretrained import (
        eval_cath_probe,
        train_cath_probe,
    )

    reps_train, raw_train, _, _ = synthetic_cath_batch
    head, meta = train_cath_probe(reps_train, mask=None, raw=raw_train, level="T")

    # 3 proteins with empty cath_code = unlabelled; should be excluded from n_eval.
    extra_reps = torch.randn(3, reps_train.shape[1])
    raw_eval = raw_train + _make_data([[], None, []])
    reps_eval = torch.cat([reps_train, extra_reps], dim=0)

    res = eval_cath_probe(head, meta, reps_eval, mask=None, raw=raw_eval)
    assert res.n_eval == len(raw_train)  # the 3 unlabelled don't count


def test_too_few_classes_returns_nan_row():
    """If only one surviving class after filtering, return NaN row, no crash."""
    from lib.probes.cath_pretrained import (
        eval_cath_probe,
        train_cath_probe,
    )

    raw = _make_data(["1.10.10"] * 12)  # only one class
    reps = torch.randn(12, 16)
    head, meta = train_cath_probe(reps, mask=None, raw=raw, level="T")
    assert head is None
    assert meta["vocab"] == []

    res = eval_cath_probe(head, meta, reps, mask=None, raw=raw)
    assert np.isnan(res.accuracy)
    assert np.isnan(res.macro_f1)
    assert res.n_classes == 0


def test_run_pretrained_cath_probe_schema(synthetic_cath_batch):
    """Flat dict matches what pretrain_probe_sweep.py will append to JSONL."""
    from lib.probes.cath_pretrained import run_pretrained_cath_probe

    reps, raw, _, _ = synthetic_cath_batch
    out = run_pretrained_cath_probe(
        reps_train=reps,
        train_batch={"mask": None},
        raw_train=raw,
        reps_eval=reps,
        eval_batch={"mask": None},
        raw_eval=raw,
        level="T",
    )
    expected = {
        "cath_level",
        "cath_accuracy",
        "cath_macro_f1",
        "cath_top1_conf_mean",
        "cath_n_classes",
        "cath_n_train",
        "cath_n_eval",
        "cath_n_eval_in_vocab",
        "cath_n_eval_oov",
    }
    assert expected.issubset(set(out.keys()))
    assert out["cath_level"] == "T"
    assert out["cath_n_classes"] == 3


def test_per_residue_input_pools(synthetic_cath_batch):
    """3D reps [B, N, D] with a mask should pool internally and reach the same accuracy."""
    from lib.probes.cath_pretrained import (
        eval_cath_probe,
        train_cath_probe,
    )

    reps_2d, raw, _, _ = synthetic_cath_batch
    B, D = reps_2d.shape
    N = 20
    # Tile each [D] rep across N residues so mean-pool returns the same vector.
    reps_3d = reps_2d.unsqueeze(1).expand(B, N, D).contiguous()
    mask = torch.ones(B, N, dtype=torch.bool)

    head, meta = train_cath_probe(reps_3d, mask=mask, raw=raw, level="T")
    res = eval_cath_probe(head, meta, reps_3d, mask=mask, raw=raw)
    assert 0.95 <= res.accuracy <= 1.0


def test_mlp_head_runs(synthetic_cath_batch):
    """MLP path should fit and produce sensible metrics on separable data."""
    from lib.probes.cath_pretrained import (
        eval_cath_probe,
        train_cath_probe,
    )

    reps, raw, _, _ = synthetic_cath_batch
    head, meta = train_cath_probe(
        reps,
        mask=None,
        raw=raw,
        level="T",
        head_type="mlp",
        mlp_epochs=200,
        device="cpu",
    )
    assert head is not None
    res = eval_cath_probe(head, meta, reps, mask=None, raw=raw, device="cpu")
    assert 0.9 <= res.accuracy <= 1.0
    assert 0.0 <= res.top1_conf_mean <= 1.0


def test_pickle_roundtrip_linear_head(synthetic_cath_batch):
    """The linear head + meta must pickle round-trip without changing predictions.

    This is the contract build_cath_classifier.py and the generation suite rely
    on: a head trained once and shipped via pickle to other processes / hosts.
    """
    from lib.probes.cath_pretrained import (
        eval_cath_probe,
        train_cath_probe,
    )

    reps, raw, _, _ = synthetic_cath_batch
    head, meta = train_cath_probe(reps, mask=None, raw=raw, level="T")
    res_orig = eval_cath_probe(head, meta, reps, mask=None, raw=raw)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "head.pkl"
        with open(path, "wb") as f:
            pickle.dump({"head": head, "train_meta": meta}, f)
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        res_reload = eval_cath_probe(
            bundle["head"], bundle["train_meta"], reps, mask=None, raw=raw
        )

    assert res_orig.accuracy == pytest.approx(res_reload.accuracy)
    assert res_orig.macro_f1 == pytest.approx(res_reload.macro_f1)
    assert res_orig.top1_conf_mean == pytest.approx(res_reload.top1_conf_mean)


def test_level_extraction_truncation():
    """Confirm level masking: '1.10.10.20' at level=A should give '1.10'."""
    from lib.probes.cath_pretrained import train_cath_probe

    # Two A-level classes, each with two T-level subdivisions, 6 proteins per A-class.
    cath_codes = (
        ["1.10.10.20"] * 6
        + ["1.10.20.30"] * 6
        + ["3.40.50.60"] * 6
        + ["3.40.70.80"] * 6
    )
    raw = _make_data(cath_codes)
    # Reps separable at A-level (two clusters), not at T-level.
    a_labels = np.array([0] * 12 + [1] * 12)
    reps = _make_separable_reps(a_labels, n_classes=2, dim=16, seed=2)

    _, meta_A = train_cath_probe(reps, mask=None, raw=raw, level="A")
    assert set(meta_A["vocab"]) == {"1.10", "3.40"}
    _, meta_T = train_cath_probe(reps, mask=None, raw=raw, level="T")
    assert set(meta_T["vocab"]) == {"1.10.10", "1.10.20", "3.40.50", "3.40.70"}
