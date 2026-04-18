"""Unit tests for LengthBucketedBatchSampler.

Isolated: the sampler module is not imported by any existing training code.
These tests only exercise index logic with synthetic length arrays — no
model, no LMDB, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteinfoundation.utils.length_bucket_sampler import LengthBucketedBatchSampler


# ----- validation -----


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="must equal"):
        LengthBucketedBatchSampler(
            lengths=[50, 100, 200],
            bucket_boundaries=[128, 256],
            bucket_batch_sizes=[8, 4, 2],
        )


def test_rejects_non_ascending_boundaries():
    with pytest.raises(ValueError, match="ascending"):
        LengthBucketedBatchSampler(
            lengths=[50, 100, 200],
            bucket_boundaries=[256, 128],
            bucket_batch_sizes=[8, 4],
        )


def test_rejects_nonpositive_batch_size():
    with pytest.raises(ValueError, match="positive"):
        LengthBucketedBatchSampler(
            lengths=[50],
            bucket_boundaries=[128],
            bucket_batch_sizes=[0],
        )


# ----- bucket assignment -----


def test_boundary_belongs_to_lower_bucket():
    """Length == boundary should land in the *lower* bucket (inclusive upper edge)."""
    lengths = [128, 129, 256, 257]
    s = LengthBucketedBatchSampler(
        lengths=lengths,
        bucket_boundaries=[128, 256, 384],
        bucket_batch_sizes=[1, 1, 1],
        shuffle=False,
        drop_last=False,
    )
    # bucket 0 has length-128
    assert list(s.buckets[0]) == [0]
    # bucket 1 has length-129 and length-256
    assert set(s.buckets[1].tolist()) == {1, 2}
    # bucket 2 has length-257
    assert list(s.buckets[2]) == [3]


def test_over_max_boundary_is_dropped():
    """Samples above the last boundary are dropped with a warning."""
    with pytest.warns(UserWarning, match="length > 128"):
        s = LengthBucketedBatchSampler(
            lengths=[50, 100, 200],  # 200 exceeds last boundary=128
            bucket_boundaries=[128],
            bucket_batch_sizes=[1],
        )
    # only 2 samples assigned
    assert sum(len(b) for b in s.buckets) == 2


# ----- iteration -----


def test_every_batch_is_bucket_homogeneous():
    """Every yielded batch must have lengths all in one bucket."""
    rng = np.random.default_rng(42)
    lengths = rng.integers(50, 512, size=1000)
    s = LengthBucketedBatchSampler(
        lengths=lengths,
        bucket_boundaries=[128, 256, 384, 512],
        bucket_batch_sizes=[16, 8, 4, 2],
        shuffle=True,
        seed=0,
    )

    for batch in s:
        batch_lens = lengths[batch]
        # which bucket does the first element belong to?
        # all others must match
        expected_bucket = np.searchsorted(
            [128, 256, 384, 512], batch_lens[0], side="left"
        )
        actual_buckets = np.searchsorted([128, 256, 384, 512], batch_lens, side="left")
        assert np.all(actual_buckets == expected_bucket), (
            f"Batch has mixed buckets: lens={batch_lens.tolist()}, "
            f"buckets={actual_buckets.tolist()}"
        )


def test_batch_sizes_match_bucket_spec():
    """Every batch yielded has exactly the BS configured for its bucket."""
    rng = np.random.default_rng(0)
    lengths = rng.integers(50, 512, size=500)
    bs_spec = [16, 8, 4, 2]
    s = LengthBucketedBatchSampler(
        lengths=lengths,
        bucket_boundaries=[128, 256, 384, 512],
        bucket_batch_sizes=bs_spec,
        drop_last=True,
    )

    for batch in s:
        batch_lens = lengths[batch]
        bucket = int(np.searchsorted([128, 256, 384, 512], batch_lens[0], side="left"))
        assert (
            len(batch) == bs_spec[bucket]
        ), f"Wrong BS in bucket {bucket}: got {len(batch)}, expected {bs_spec[bucket]}"


def test_len_matches_actual_batches_yielded():
    rng = np.random.default_rng(7)
    lengths = rng.integers(50, 512, size=731)  # non-round to expose drop_last
    s = LengthBucketedBatchSampler(
        lengths=lengths,
        bucket_boundaries=[128, 256, 384, 512],
        bucket_batch_sizes=[16, 8, 4, 2],
        drop_last=True,
    )
    n_yielded = sum(1 for _ in s)
    assert len(s) == n_yielded


def test_drop_last_false_keeps_trailing():
    """drop_last=False yields smaller trailing batches when bucket size isn't divisible by BS."""
    lengths = [50] * 10 + [200] * 5  # 10 in bucket 0, 5 in bucket 1
    s = LengthBucketedBatchSampler(
        lengths=lengths,
        bucket_boundaries=[128, 256],
        bucket_batch_sizes=[
            4,
            3,
        ],  # 10/4 = 2 full + 1 partial (2), 5/3 = 1 full + 1 partial (2)
        shuffle=False,
        drop_last=False,
    )
    batches = list(s)
    # 2 full + 1 partial for bucket 0 = 3 batches
    # 1 full + 1 partial for bucket 1 = 2 batches
    assert len(batches) == 5

    sizes_b0 = sorted(len(b) for b in batches if all(lengths[i] <= 128 for i in b))
    sizes_b1 = sorted(len(b) for b in batches if all(lengths[i] > 128 for i in b))
    assert sizes_b0 == [2, 4, 4]  # one partial of size 2
    assert sizes_b1 == [2, 3]  # one partial of size 2


def test_drop_last_true_drops_trailing():
    lengths = [50] * 10 + [200] * 5
    s = LengthBucketedBatchSampler(
        lengths=lengths,
        bucket_boundaries=[128, 256],
        bucket_batch_sizes=[4, 3],
        drop_last=True,
    )
    batches = list(s)
    # 10//4=2 full, 5//3=1 full → 3 total
    assert len(batches) == 3
    for b in batches:
        assert len(b) in (3, 4)


# ----- shuffle semantics -----


def test_shuffle_changes_order_across_epochs():
    lengths = list(range(50, 400))
    s = LengthBucketedBatchSampler(
        lengths=lengths,
        bucket_boundaries=[128, 256, 384],
        bucket_batch_sizes=[8, 4, 2],
        shuffle=True,
        seed=42,
    )

    s.set_epoch(0)
    epoch_0 = [tuple(b) for b in s]
    s.set_epoch(1)
    epoch_1 = [tuple(b) for b in s]

    assert epoch_0 != epoch_1, "shuffle should yield different orderings per epoch"


def test_shuffle_false_is_deterministic():
    lengths = list(range(50, 400))
    s1 = LengthBucketedBatchSampler(
        lengths=lengths,
        bucket_boundaries=[128, 256, 384],
        bucket_batch_sizes=[8, 4, 2],
        shuffle=False,
    )
    s2 = LengthBucketedBatchSampler(
        lengths=lengths,
        bucket_boundaries=[128, 256, 384],
        bucket_batch_sizes=[8, 4, 2],
        shuffle=False,
    )
    assert [tuple(b) for b in s1] == [tuple(b) for b in s2]


def test_same_seed_same_order():
    lengths = list(range(50, 400))
    s1 = LengthBucketedBatchSampler(
        lengths=lengths,
        bucket_boundaries=[128, 256, 384],
        bucket_batch_sizes=[8, 4, 2],
        shuffle=True,
        seed=99,
    )
    s2 = LengthBucketedBatchSampler(
        lengths=lengths,
        bucket_boundaries=[128, 256, 384],
        bucket_batch_sizes=[8, 4, 2],
        shuffle=True,
        seed=99,
    )
    assert [tuple(b) for b in s1] == [tuple(b) for b in s2]


# ----- real data shape smoke -----


def test_describe_runs_on_real_shape():
    """Sampler with the proteina-data-shaped inputs should describe cleanly."""
    rng = np.random.default_rng(123)
    # rough PDB length distribution approximation: mean 223, max 511
    lengths = np.clip(rng.normal(223, 100, size=10_000).astype(int), 50, 511)
    s = LengthBucketedBatchSampler(
        lengths=lengths,
        bucket_boundaries=[128, 256, 384, 512],
        bucket_batch_sizes=[90, 24, 10, 5],
    )
    desc = s.describe()
    assert "bucket 0" in desc and "bucket 3" in desc
    assert "BS=90" in desc
    # should yield a non-trivial number of batches
    assert len(s) > 100
