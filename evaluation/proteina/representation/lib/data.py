"""LMDB loading, padding, batching.

Produces the shared batch dict every rep source and probe is scored against.
"""

from __future__ import annotations

import pickle
from typing import Dict, List, Tuple

import lmdb
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from probelib.checkpoints import LMDB_PATH


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_proteina_batch(
    n: int,
    max_size: int = 256,
    lmdb_path: str = LMDB_PATH,
    device: str = None,
) -> Tuple[Dict, List[Data]]:
    """Read ``n`` PyG Data objects from the LMDB, filter to length ≤ max_size,
    pad all tensors to max_size, stack into a batched dict compatible with
    ``model.nn.forward``.

    Shape flow:
        lmdb entries (pickled Data) -> filter len ≤ max_size
                                    -> right-pad each tensor key to [max_size, ...]
                                    -> torch.stack over the batch dim

    Inputs:
        n:         target batch size
        max_size:  pad / truncate to this length
        lmdb_path: override env-resolved path
        device:    where to place the batch (default cuda if available)

    Returns:
        batch:  dict of named tensors, all first-dim B:
                  coords        [B, N, 37, 3]  Å (CA at index 1)
                  mask          [B, N]         bool, True = real residue
                  residue_type  [B, N]         long, OpenFold AA indices 0..19
                  chain_break_per_res  [B, N]  long
                  lengths       [B]            unmasked residue counts
                (other per-residue tensor fields from the source Data are
                passed through if they stack cleanly.)
        raw:    list of the original Data objects — needed for CATH labels
                and other per-protein string attrs that don't tensorize.
    """
    if device is None:
        device = _default_device()

    db = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        subdir=False,
        readahead=False,
        meminit=False,
    )
    raw: List[Data] = []
    with db.begin() as txn:
        cursor = txn.cursor()
        for _, v in cursor:
            g = pickle.loads(v)
            n_res = g.coords.shape[0] if hasattr(g, "coords") else g.num_nodes
            if n_res <= max_size:
                raw.append(g)
            if len(raw) >= n:
                break
    db.close()

    if not raw:
        raise RuntimeError(f"No proteins ≤ {max_size} residues found in {lmdb_path}")

    B = len(raw)

    # Collect all tensor-valued keys across the batch so we can attempt a stack
    # for each one.
    keys_tensor = set()
    for g in raw:
        for k, v in g:
            if isinstance(v, torch.Tensor):
                keys_tensor.add(k)

    batch: Dict[str, torch.Tensor] = {}
    for k in keys_tensor:
        stacked = []
        for g in raw:
            v = g[k]  # arbitrary shape; leading dim is residues for per-residue fields
            if v.dim() == 0:
                # Scalar per-protein attribute — stack as-is.
                stacked.append(v)
                continue
            pad_size = max_size - v.shape[0]
            if pad_size > 0:
                # F.pad order is right-to-left over dims; set the last pair to
                # pad dim 0 after.
                pad_tuple = [0] * (2 * v.dim())
                pad_tuple[-1] = pad_size
                v = F.pad(v, tuple(pad_tuple), mode="constant", value=0)
            elif pad_size < 0:
                v = v[:max_size]
            stacked.append(v)
        try:
            batch[k] = torch.stack(stacked, dim=0).to(device)  # [B, ...]
        except Exception as e:
            # Shapes don't line up across proteins for this key — skip it.
            print(f"  [warn] skipping key '{k}' in batch (stack failed: {e})")

    # Build an explicit [B, N] bool mask from the original lengths. The model's
    # cond_factory does ``~mask``, which only works on bool/int tensors.
    lengths = torch.tensor([g.coords.shape[0] for g in raw], device=device)  # [B]
    arange = torch.arange(max_size, device=device).unsqueeze(0).expand(B, -1)
    mask = arange < lengths.unsqueeze(1)  # [B, N] bool
    batch["mask"] = mask
    batch["lengths"] = lengths

    return batch, raw
