"""Smoke tests for feature_cache.py - import, path helpers, round-trip."""

import os
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPR_LIB = os.path.join(REPO_ROOT, "evaluation", "proteina", "representation")
sys.path.insert(0, REPR_LIB)


def test_cache_path_conventions():
    from lib.feature_cache import cache_path

    p = cache_path(Path("/tmp/cache"), "baseline", 400000, t_value=1.0)
    assert str(p).endswith("features_baseline_step400000_t1.00")


def test_cache_roundtrip_and_purge():
    from lib.feature_cache import (
        _tensor_path,
        cache_is_complete,
        cache_path,
        load_cached_layer,
        purge_cache,
    )

    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        d = cache_path(cache_dir, "baseline", 123, t_value=1.0)
        d.mkdir(parents=True, exist_ok=True)
        # Write one fp16 tensor per split + per layer manually; extract_and_cache
        # needs a real model so we stub only what the loader consumes.
        layers = [0, 1]
        for L in layers:
            t = torch.randn(4, 10, 8, dtype=torch.float16)
            torch.save(t, _tensor_path(cache_dir, "baseline", 123, 1.0, "train", L))
            torch.save(t, _tensor_path(cache_dir, "baseline", 123, 1.0, "eval", L))
        assert cache_is_complete(cache_dir, "baseline", 123, 1.0, layers)

        back = load_cached_layer(
            cache_dir, "baseline", 123, 1.0, "train", 0, dtype=torch.float32
        )
        assert back.shape == (4, 10, 8)
        assert back.dtype == torch.float32

        purge_cache(cache_dir, "baseline", 123, 1.0)
        assert not cache_path(cache_dir, "baseline", 123, 1.0).exists()
        assert not cache_is_complete(cache_dir, "baseline", 123, 1.0, layers)
