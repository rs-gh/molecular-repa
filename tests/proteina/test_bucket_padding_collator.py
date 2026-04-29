"""Unit tests for bucket-aware DensePaddingCollater padding.

Exercises the `bucket_boundaries` path that pads node-dim tensors to the
smallest boundary >= max(num_nodes) in the batch, producing fixed `(B, N)`
shapes per bucket. See p2_compile_multishape.py for the torch.compile
verification this enables.

Isolated: synthetic PyG Data only; no dataset, no model, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

import proteinfoundation.repa.pyg_compat  # noqa: F401
from proteinfoundation.utils.dense_padding_data_loader import (
    DensePaddingDataLoader,
    FLOAT_PADDING_VALUE,
    _target_length_for_batch,
    dense_padded_from_data_list,
)
from proteinfoundation.utils.length_bucket_sampler import LengthBucketedBatchSampler


def _make_sample(n: int) -> Data:
    return Data(
        coords=torch.randn(n, 4, 3),
        seq=torch.randint(0, 20, (n,)),
        res_seq_pdb_idx=torch.arange(n, dtype=torch.long),
        chain_break_per_res=torch.zeros(n, dtype=torch.long),
        num_nodes=n,
    )


# ----- _target_length_for_batch -----


def test_target_length_none_when_no_boundaries():
    datas = [_make_sample(5), _make_sample(10)]
    assert _target_length_for_batch(datas, None) is None


def test_target_length_picks_smallest_upper_bound():
    datas = [_make_sample(5), _make_sample(10), _make_sample(100)]
    assert _target_length_for_batch(datas, [128, 256, 512]) == 128
    assert _target_length_for_batch(datas, [64, 128, 256]) == 128
    # When batch_max == boundary, side='left' in sampler puts it in the
    # lower bucket; the collator just needs smallest boundary >= batch_max.
    datas = [_make_sample(128)]
    assert _target_length_for_batch(datas, [128, 256]) == 128


def test_target_length_raises_when_overflow():
    datas = [_make_sample(600)]
    with pytest.raises(ValueError, match="exceeds the largest bucket boundary"):
        _target_length_for_batch(datas, [128, 256, 512])


# ----- dense_padded_from_data_list -----


def test_no_boundaries_preserves_existing_behavior():
    lengths = [7, 12, 20]
    datas = [_make_sample(n) for n in lengths]
    batch = dense_padded_from_data_list(datas)
    assert tuple(batch["coords"].shape) == (3, 20, 4, 3)
    assert not hasattr(batch, "bucket_length")


def test_pads_to_bucket_upper_edge():
    lengths = [7, 12, 20]
    datas = [_make_sample(n) for n in lengths]
    batch = dense_padded_from_data_list(datas, bucket_boundaries=[32, 64])
    assert tuple(batch["coords"].shape) == (3, 32, 4, 3)
    assert tuple(batch["seq"].shape) == (3, 32)
    assert batch.bucket_length == 32


def test_mask_correct_after_bucket_pad():
    lengths = [7, 12, 20]
    datas = [_make_sample(n) for n in lengths]
    batch = dense_padded_from_data_list(datas, bucket_boundaries=[32, 64])
    mask = batch.mask_dict["coords"][:, :, 0, 0]  # [B, N] bool
    for i, n in enumerate(lengths):
        assert bool(mask[i, :n].all()), f"real region must be True for sample {i}"
        assert not bool(mask[i, n:].any()), f"pad region must be False for sample {i}"


def test_no_extension_when_boundary_equals_batch_max():
    lengths = [7, 12, 20]
    datas = [_make_sample(n) for n in lengths]
    batch = dense_padded_from_data_list(datas, bucket_boundaries=[20, 40])
    assert tuple(batch["coords"].shape) == (3, 20, 4, 3)
    assert batch.bucket_length == 20


def test_pad_value_matches_dtype():
    """Float tensors pad with FLOAT_PADDING_VALUE; int tensors with
    NON_FLOAT_PADDING_VALUE; bool masks with False."""
    datas = [_make_sample(8)]
    batch = dense_padded_from_data_list(datas, bucket_boundaries=[16])
    coords = batch["coords"]  # float
    seq = batch["seq"]  # int
    mask = batch.mask_dict["coords"]  # bool
    # Beyond n=8 should be padding
    assert torch.all(coords[0, 8:] == FLOAT_PADDING_VALUE)
    assert torch.all(seq[0, 8:] == -1)  # NON_FLOAT_PADDING_VALUE
    assert not mask[0, 8:].any()


# ----- end-to-end via DataLoader + sampler -----


class _SynthDataset(Dataset):
    def __init__(self, lengths):
        self._lengths = np.asarray(lengths, dtype=np.int64)

    def __len__(self):
        return len(self._lengths)

    def __getitem__(self, idx):
        return _make_sample(int(self._lengths[idx]))

    def get_lengths(self):
        return self._lengths


def test_bucketed_dataloader_yields_fixed_shapes_per_bucket():
    rng = np.random.default_rng(0)
    lengths = rng.integers(50, 500, size=120).astype(np.int64)
    ds = _SynthDataset(lengths)
    boundaries = [128, 256, 384, 512]
    bucket_bs = [16, 8, 4, 2]

    sampler = LengthBucketedBatchSampler(
        lengths=ds.get_lengths(),
        bucket_boundaries=boundaries,
        bucket_batch_sizes=bucket_bs,
        shuffle=True,
        drop_last=True,
        seed=0,
    )
    loader = DensePaddingDataLoader(
        ds,
        batch_sampler=sampler,
        bucket_boundaries=boundaries,
        num_workers=0,
    )

    shapes = set()
    for batch in loader:
        shapes.add(tuple(batch["coords"].shape))

    expected = {(bs, b, 4, 3) for bs, b in zip(bucket_bs, boundaries)}
    assert shapes == expected


def test_bucketed_dataloader_respects_drop_last_batch_sizes():
    """drop_last=True means every yielded batch has exactly the bucket's BS -
    required for torch.compile to see the same (B, N) per bucket."""
    lengths = np.concatenate(
        [np.full(10, 100), np.full(20, 300), np.full(30, 500)]
    ).astype(np.int64)
    ds = _SynthDataset(lengths)
    boundaries = [128, 384, 512]
    bucket_bs = [4, 7, 8]

    sampler = LengthBucketedBatchSampler(
        lengths=ds.get_lengths(),
        bucket_boundaries=boundaries,
        bucket_batch_sizes=bucket_bs,
        shuffle=False,
        drop_last=True,
        seed=0,
    )
    loader = DensePaddingDataLoader(
        ds,
        batch_sampler=sampler,
        bucket_boundaries=boundaries,
        num_workers=0,
    )

    for batch in loader:
        B, N = batch["coords"].shape[:2]
        assert (B, N) in {(bs, b) for bs, b in zip(bucket_bs, boundaries)}
