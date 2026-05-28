"""Ephemeral per-checkpoint feature cache for the pretrained-probe pipeline.

Extracting features over (N_train train.lmdb proteins + N_eval val.lmdb
proteins) at every transformer layer is the expensive step — it requires a
full model forward pass and dominates per-checkpoint wall time. Once those
features exist, probe heads train in seconds on pure pair-MLP. Caching
amortises extraction across:
    - all N_layers probes for a single checkpoint (one backbone forward covers
      all layers)
    - SLURM preemption / Lustre hiccups mid-checkpoint (resume skips work
      already on disk)
    - re-running the probe head with different hyperparameters without
      re-extracting features

The cache is **per-checkpoint-and-ephemeral**: after all layer probes for a
checkpoint are done, ``purge_cache`` deletes the directory to keep peak disk
use bounded. At N_train=2000, max_size=256, D=512, 10 layers, fp16 the
per-checkpoint footprint is ~5 GB.

Layout on disk:

    cache/features_{run}_step{step}_t{t_tag}/
      train_L0.pt   train_L1.pt  ...  train_LN.pt     # each [B_train, N, D] fp16
      eval_L0.pt    eval_L1.pt   ...  eval_LN.pt      # each [B_eval,  N, D] fp16
      meta.json                                        # manifest tags, shapes, timestamps

Features are stored in fp16 to halve disk and to keep downstream probe
training identical (Adam + BCE is numerically robust at fp16 input cast
back to fp32 before the head).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from lib.extract import extract_model_hidden_states_multilayer


def _t_tag(t_value: float) -> str:
    """Canonical timestep string for filesystem paths. Matches run_sweep.py."""
    return f"{t_value:.2f}"


def cache_path(cache_dir: Path, run: str, step: int, t_value: float = 1.0) -> Path:
    """Canonical per-checkpoint cache directory.

    Example:
        cache_dir / "features_baseline_step400000_t1.00"
    """
    return Path(cache_dir) / f"features_{run}_step{step}_t{_t_tag(t_value)}"


def _tensor_path(
    cache_dir: Path, run: str, step: int, t_value: float, split: str, layer: int
) -> Path:
    return cache_path(cache_dir, run, step, t_value) / f"{split}_L{layer}.pt"


def _meta_path(cache_dir: Path, run: str, step: int, t_value: float) -> Path:
    return cache_path(cache_dir, run, step, t_value) / "meta.json"


def cache_is_complete(
    cache_dir: Path,
    run: str,
    step: int,
    t_value: float,
    layers: List[int],
) -> bool:
    """True iff every (split × layer) tensor exists on disk for this checkpoint."""
    d = cache_path(cache_dir, run, step, t_value)
    if not d.exists():
        return False
    for layer in layers:
        for split in ("train", "eval"):
            if not _tensor_path(cache_dir, run, step, t_value, split, layer).exists():
                return False
    return True


def extract_and_cache(
    model,
    train_batch: Dict,
    eval_batch: Dict,
    layers: List[int],
    cache_dir: Path,
    run: str,
    step: int,
    t_value: float = 1.0,
    chunk_size: int = 8,
    train_manifest_tag: Optional[str] = None,
    eval_manifest_tag: Optional[str] = None,
) -> None:
    """Extract train+eval features across all requested layers, save to disk as fp16.

    Flow:
        1. reps_train = extract_model_hidden_states_multilayer(model, train_batch, layers)
        2. reps_eval  = extract_model_hidden_states_multilayer(model, eval_batch,  layers)
        3. for each layer:
             torch.save(reps_train[L].half(), train_L{L}.pt)
             torch.save(reps_eval [L].half(), eval_L{L}.pt)
        4. write meta.json with shapes, manifest tags, timestamps

    Two separate forward passes (train + eval) because the batches are
    different sizes; combining them would require a larger sub-batch loop
    and the saving is already the simpler code path.
    """
    d = cache_path(cache_dir, run, step, t_value)
    d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    reps_train = extract_model_hidden_states_multilayer(
        model, train_batch, layers, chunk_size=chunk_size, t_value=t_value
    )
    t_train = time.time() - t0

    t0 = time.time()
    reps_eval = extract_model_hidden_states_multilayer(
        model, eval_batch, layers, chunk_size=chunk_size, t_value=t_value
    )
    t_eval = time.time() - t0

    for L in layers:
        torch.save(
            reps_train[L].half(),
            _tensor_path(cache_dir, run, step, t_value, "train", L),
        )
        torch.save(
            reps_eval[L].half(),
            _tensor_path(cache_dir, run, step, t_value, "eval", L),
        )

    B_train, N_train, D = reps_train[layers[0]].shape
    B_eval, N_eval, _ = reps_eval[layers[0]].shape
    meta = {
        "run": run,
        "step": int(step),
        "t_value": float(t_value),
        "layers": list(layers),
        "train_shape": [B_train, N_train, D],
        "eval_shape": [B_eval, N_eval, D],
        "train_manifest_tag": train_manifest_tag,
        "eval_manifest_tag": eval_manifest_tag,
        "dtype_on_disk": "fp16",
        "extract_s_train": float(t_train),
        "extract_s_eval": float(t_eval),
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(_meta_path(cache_dir, run, step, t_value), "w") as f:
        json.dump(meta, f, indent=2)


def extract_reps_in_memory(
    model,
    train_batch: Dict,
    eval_batch: Dict,
    layers: List[int],
    chunk_size: int = 8,
    t_value: float = 1.0,
    dtype: torch.dtype = torch.float16,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Extract train+eval features for all layers, return them in-memory (no disk).

    Mirror of ``extract_and_cache`` minus the persistence: same two forward
    passes, same fp16 cast for memory budget, but the result is returned as
    ``{layer: {"train": [B_train, N, D], "eval": [B_eval, N, D]}}`` on CPU
    rather than written under ``cache_dir/...``.

    Use case: SLURM fan-out where each task processes one checkpoint
    end-to-end. Trades the per-checkpoint disk footprint (~5-12 GB at
    n=128, ~12-24 GB at n=256) for RAM. The caller is responsible for
    dropping individual layer entries as soon as their probes finish to
    keep peak RAM bounded.

    See ``extract_and_cache`` for the cached-on-disk variant used in the
    default ``--probe_from_cache``-friendly mode.
    """
    t0 = time.time()
    reps_train = extract_model_hidden_states_multilayer(
        model, train_batch, layers, chunk_size=chunk_size, t_value=t_value
    )
    t_train = time.time() - t0

    t0 = time.time()
    reps_eval = extract_model_hidden_states_multilayer(
        model, eval_batch, layers, chunk_size=chunk_size, t_value=t_value
    )
    t_eval = time.time() - t0

    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for L in layers:
        out[L] = {
            "train": reps_train[L].to(dtype).cpu(),
            "eval": reps_eval[L].to(dtype).cpu(),
        }
    print(
        f"    [in-memory] extracted {len(layers)} layers; "
        f"train={t_train:.1f}s eval={t_eval:.1f}s; "
        f"per-layer fp16 size: "
        f"train={reps_train[layers[0]].numel() * 2 / 1e6:.0f}MB "
        f"eval={reps_eval[layers[0]].numel() * 2 / 1e6:.0f}MB"
    )
    return out


def load_cached_layer(
    cache_dir: Path,
    run: str,
    step: int,
    t_value: float,
    split: str,  # "train" or "eval"
    layer: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load one layer's cached features from disk and cast back to float32."""
    p = _tensor_path(cache_dir, run, step, t_value, split, layer)
    if not p.exists():
        raise FileNotFoundError(f"cache miss: {p}")
    t = torch.load(p, map_location="cpu", weights_only=False)
    return t.to(dtype)


def purge_cache(cache_dir: Path, run: str, step: int, t_value: float = 1.0) -> None:
    """Delete the per-checkpoint cache directory after all layer probes are done."""
    d = cache_path(cache_dir, run, step, t_value)
    if d.exists():
        shutil.rmtree(d)
