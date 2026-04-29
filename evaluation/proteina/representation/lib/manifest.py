"""Reproducible protein sampling via a versioned manifest.

The legacy ``load_proteina_batch`` takes the first ``n`` proteins ≤ max_size
that it encounters in the LMDB cursor. That's deterministic but biased by
insertion order — you get a fixed non-random subset that happens to be
"the shortest ones at the top of the LMDB." For comparing probe numbers
across sources, that's fine (all sources see the same 40 proteins) but for
reporting a number with n_test=200 it's a methodological red flag.

This module provides:

    sample_manifest(lmdb_path, n, max_size, seed) -> dict
        Uniform random sample of n LMDB keys whose protein length ≤ max_size.
        Uses `*_lengths.npy` / `*_keys.pkl` sidecars when present for speed;
        falls back to a one-pass LMDB cursor scan otherwise.

    build_or_load_manifest(dir, version, *, lmdb_path, n, max_size, seed) -> dict
        Cache-aware wrapper. Reads ``dir/batch_manifest_<version>.json`` if
        present; otherwise samples, persists, and returns the dict.

    load_proteina_batch_from_manifest(manifest, device) -> (batch, raw)
        Fetch the exact proteins named in a manifest from their LMDB and pad
        them into a batch dict matching the legacy ``load_proteina_batch``
        output shape.

Schema of the persisted JSON:

    {
      "manifest_version": "v2",
      "seed": 42,
      "n_proteins": 1000,
      "max_size": 384,
      "lmdb_path": "/rds/.../val.lmdb",
      "keys":    ["key1", "key2", ...],   # LMDB keys, utf-8 decoded
      "lengths": [150, 220, ...]           # residue counts before padding
    }

Records produced against this manifest should tag themselves with
``"manifest": "v2"`` in the JSONL so downstream plotting can filter
apples-to-apples.
"""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from lib.data import _default_device


def _sidecars(lmdb_path: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Locate ``*_keys.pkl`` and ``*_lengths.npy`` sidecars next to an LMDB.

    Layout on disk (proteina PDB LMDB build):
        pdb_train/lmdb/val.lmdb
        pdb_train/lmdb/val_keys.pkl
        pdb_train/lmdb/val_lengths.npy
    """
    p = Path(lmdb_path)
    split = p.stem  # "val", "train", "test"
    keys = p.parent / f"{split}_keys.pkl"
    lens = p.parent / f"{split}_lengths.npy"
    return (keys if keys.exists() else None, lens if lens.exists() else None)


def _sample_via_sidecars(
    keys_path: Path,
    lens_path: Path,
    n: int,
    max_size: int,
    seed: int,
    lmdb_path: str | None = None,
    require_cath: bool = False,
) -> Tuple[List[str], List[int]]:
    """Uniform sample using precomputed key + length arrays.

    Flow:
        keys    = pickle.load(...)           # list of bytes / strs, length = n_lmdb
        lengths = np.load(...)               # int array, length = n_lmdb
        eligible_idx = where(lengths <= max_size)
        sampled_idx  = rng.choice(eligible_idx, n, replace=False)
        return [keys[i] for i in sampled_idx], [int(lengths[i]) for i in sampled_idx]

    When ``require_cath=True``, the LMDB is opened and each candidate's
    ``cath_code`` is inspected; unlabelled records are skipped so the
    returned manifest contains exactly ``n`` labelled proteins. The pool
    is shuffled deterministically and walked until either ``n`` are found
    or the pool is exhausted.
    """
    with open(keys_path, "rb") as f:
        keys = pickle.load(f)
    lengths = np.load(lens_path)
    assert len(keys) == len(
        lengths
    ), f"sidecar mismatch: {len(keys)} keys vs {len(lengths)} lengths"

    eligible = np.flatnonzero(lengths <= max_size)  # [E]
    if len(eligible) < n:
        raise RuntimeError(
            f"only {len(eligible)} proteins ≤ {max_size} residues; requested n={n}"
        )
    rng = np.random.default_rng(seed)

    if not require_cath:
        sel = rng.choice(eligible, size=n, replace=False)  # [n]
        sel.sort()  # sort for cache-friendly LMDB reads
        return [
            k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            for k in (keys[i] for i in sel)
        ], [int(lengths[i]) for i in sel]

    # Labelled-only path: stream candidates, deserialize each, accept only if
    # cath_code is non-empty. We open the LMDB read-only and reuse a single
    # transaction for all probes — same access pattern as
    # ``load_proteina_batch_from_manifest``.
    if lmdb_path is None:
        raise ValueError("require_cath=True needs lmdb_path to inspect cath_code")
    perm = rng.permutation(eligible)
    db = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        subdir=False,
        readahead=False,
        meminit=False,
    )
    selected_idx: List[int] = []
    n_inspected = 0
    with db.begin() as txn:
        for i in perm:
            if len(selected_idx) >= n:
                break
            n_inspected += 1
            k = keys[i]
            if not isinstance(k, (bytes, bytearray)):
                k = str(k).encode()
            v = txn.get(k)
            if v is None:
                continue
            g = pickle.loads(v)
            cc = getattr(g, "cath_code", None)
            if cc is None:
                continue
            if hasattr(cc, "__len__") and len(cc) == 0:
                continue
            selected_idx.append(int(i))
    db.close()
    if len(selected_idx) < n:
        raise RuntimeError(
            f"only {len(selected_idx)} labelled proteins ≤ {max_size} residues "
            f"in {lmdb_path} (inspected {n_inspected} / {len(eligible)} eligible); "
            f"requested n={n}. Lower n or raise max_size."
        )
    selected_idx.sort()  # cache-friendly LMDB reads
    return [
        k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
        for k in (keys[i] for i in selected_idx)
    ], [int(lengths[i]) for i in selected_idx]


def _sample_via_cursor(
    lmdb_path: str,
    n: int,
    max_size: int,
    seed: int,
    require_cath: bool = False,
) -> Tuple[List[str], List[int]]:
    """Fallback: one-pass reservoir sample over the LMDB cursor.

    Slower than the sidecar path (must deserialize every entry to get length)
    but works when sidecars aren't present. O(n_lmdb) deserializations.

    When ``require_cath=True``, candidates without a non-empty ``cath_code``
    are skipped before entering the reservoir.
    """
    db = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        subdir=False,
        readahead=False,
        meminit=False,
    )
    rng = random.Random(seed)
    reservoir: List[Tuple[str, int]] = []
    seen_eligible = 0
    with db.begin() as txn:
        cursor = txn.cursor()
        for k, v in cursor:
            g = pickle.loads(v)
            n_res = g.coords.shape[0] if hasattr(g, "coords") else g.num_nodes
            if n_res > max_size:
                continue
            if require_cath:
                cc = getattr(g, "cath_code", None)
                if cc is None or (hasattr(cc, "__len__") and len(cc) == 0):
                    continue
            key_str = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            seen_eligible += 1
            if len(reservoir) < n:
                reservoir.append((key_str, int(n_res)))
            else:
                # Classic reservoir sampling (Vitter's R).
                j = rng.randint(0, seen_eligible - 1)
                if j < n:
                    reservoir[j] = (key_str, int(n_res))
    db.close()
    if seen_eligible < n:
        msg_tag = "labelled " if require_cath else ""
        raise RuntimeError(
            f"only {seen_eligible} {msg_tag}proteins ≤ {max_size} residues "
            f"in {lmdb_path}; requested n={n}"
        )
    reservoir.sort(key=lambda kv: kv[0])  # deterministic order post-sampling
    return [kv[0] for kv in reservoir], [kv[1] for kv in reservoir]


def sample_manifest(
    lmdb_path: str,
    n: int,
    max_size: int,
    seed: int,
    require_cath: bool = False,
) -> Dict:
    """Uniform random sample of ``n`` LMDB keys with protein length ≤ max_size.

    Returns a manifest dict (unsaved). Uses sidecars for speed when available.

    When ``require_cath=True``, only proteins with a non-empty ``cath_code``
    attribute count toward ``n``. The cursor-fallback path also honours the
    flag.
    """
    keys_path, lens_path = _sidecars(lmdb_path)
    if keys_path is not None and lens_path is not None:
        keys, lengths = _sample_via_sidecars(
            keys_path,
            lens_path,
            n,
            max_size,
            seed,
            lmdb_path=lmdb_path,
            require_cath=require_cath,
        )
    else:
        keys, lengths = _sample_via_cursor(
            lmdb_path,
            n,
            max_size,
            seed,
            require_cath=require_cath,
        )
    return {
        "manifest_version": "v2",
        "seed": int(seed),
        "n_proteins": int(n),
        "max_size": int(max_size),
        "require_cath": bool(require_cath),
        "lmdb_path": str(lmdb_path),
        "keys": list(keys),  # [n] utf-8 strings
        "lengths": list(lengths),  # [n] ints
    }


def build_or_load_manifest(
    outdir: Path,
    version: str,
    *,
    lmdb_path: str,
    n: int,
    max_size: int,
    seed: int,
    require_cath: bool = False,
) -> Dict:
    """Return the manifest at ``outdir/batch_manifest_<version>.json``.

    Build it (via reservoir sample) if missing; otherwise load and validate
    that the on-disk manifest matches the requested (n, max_size, seed,
    require_cath). If not, raise — the caller must pick a new version tag
    rather than silently using a different sample.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"batch_manifest_{version}.json"
    if path.exists():
        with open(path) as f:
            manifest = json.load(f)
        # Default require_cath to False on legacy manifests so existing
        # caches keep validating.
        cached_req = bool(manifest.get("require_cath", False))
        if (
            manifest.get("n_proteins") != n
            or manifest.get("max_size") != max_size
            or manifest.get("seed") != seed
            or cached_req != require_cath
        ):
            raise RuntimeError(
                f"manifest at {path} has "
                f"n_proteins={manifest.get('n_proteins')}, "
                f"max_size={manifest.get('max_size')}, "
                f"seed={manifest.get('seed')}, "
                f"require_cath={cached_req} — does not match requested "
                f"(n={n}, max_size={max_size}, seed={seed}, "
                f"require_cath={require_cath}). "
                f"Bump --manifest_version to regenerate."
            )
        return manifest
    manifest = sample_manifest(
        lmdb_path,
        n=n,
        max_size=max_size,
        seed=seed,
        require_cath=require_cath,
    )
    manifest["manifest_version"] = version  # allow versions other than "v2"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    tag = " labelled" if require_cath else ""
    print(
        f"  wrote manifest: {path} ({n}{tag} proteins, max_size={max_size}, seed={seed})"
    )
    return manifest


def load_proteina_batch_from_manifest(
    manifest: Dict,
    device: str = None,
) -> Tuple[Dict, List[Data]]:
    """Fetch the proteins named in ``manifest["keys"]`` from the LMDB.

    Same output shape as the legacy ``load_proteina_batch``: a batch dict of
    [B, ...] tensors on ``device`` plus the list of raw Data objects.

    Flow:
        for key in manifest["keys"]:
            raw.append(pickle.loads(lmdb[key]))
        # then pad every per-residue tensor to [B, max_size, ...]
        # then stack to [B, ...] and build the [B, N] mask from real lengths.
    """
    if device is None:
        device = _default_device()

    lmdb_path = manifest["lmdb_path"]
    keys: List[str] = manifest["keys"]
    max_size: int = manifest["max_size"]

    db = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        subdir=False,
        readahead=False,
        meminit=False,
    )
    raw: List[Data] = []
    missing: List[str] = []
    with db.begin() as txn:
        for k in keys:
            v = txn.get(k.encode() if isinstance(k, str) else k)
            if v is None:
                missing.append(k)
                continue
            raw.append(pickle.loads(v))
    db.close()

    if missing:
        raise RuntimeError(
            f"{len(missing)} manifest keys not found in {lmdb_path}; "
            f"first few: {missing[:3]}"
        )

    B = len(raw)

    keys_tensor = set()
    for g in raw:
        for k, v in g:
            if isinstance(v, torch.Tensor):
                keys_tensor.add(k)

    batch: Dict[str, torch.Tensor] = {}
    for k in keys_tensor:
        stacked = []
        for g in raw:
            v = g[k]
            if v.dim() == 0:
                stacked.append(v)
                continue
            pad_size = max_size - v.shape[0]
            if pad_size > 0:
                pad_tuple = [0] * (2 * v.dim())
                pad_tuple[-1] = pad_size
                v = F.pad(v, tuple(pad_tuple), mode="constant", value=0)
            elif pad_size < 0:
                v = v[:max_size]
            stacked.append(v)
        try:
            batch[k] = torch.stack(stacked, dim=0).to(device)  # [B, ...]
        except Exception as e:
            print(f"  [warn] skipping key '{k}' in batch (stack failed: {e})")

    lengths = torch.tensor([g.coords.shape[0] for g in raw], device=device)  # [B]
    arange = torch.arange(max_size, device=device).unsqueeze(0).expand(B, -1)
    mask = arange < lengths.unsqueeze(1)  # [B, N] bool
    batch["mask"] = mask
    batch["lengths"] = lengths

    return batch, raw
