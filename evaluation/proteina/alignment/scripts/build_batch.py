"""Build the frozen n=256 PDB batch used for all CKNNA cells.

Loads ~48 proteins (≤256 residues) from the proteina LMDB via the existing
``load_proteina_batch`` helper, persists the batch tensors + the residue-level
flattening manifest so every feature-extraction script sees the SAME proteins
and the SAME (batch_i, residue_j) ordering for the flattened residue axis.

Run:
    source .venv/bin/activate
    export PROJECT_ROOT=$(pwd)/src/proteina
    python evaluation/proteina/alignment/scripts/build_batch.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ALIGN_ROOT = HERE.parent  # evaluation/proteina/alignment
REP_ROOT = ALIGN_ROOT.parent / "representation"  # for `lib` (load_proteina_batch)
PROTEINA_ROOT = Path("/home/sr2173/git/molecular-repa/src/proteina")

for p in (str(REP_ROOT), str(PROTEINA_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib import LMDB_PATH, _default_device, load_proteina_batch  # noqa: E402

OUT_DIR = ALIGN_ROOT / "results"
OUT_PATH = OUT_DIR / "frozen_batch_n256.pt"

N_PROTEINS = 64  # 64 × ~160 real residues ≈ 10k residues, matches REPA paper N=10k
MAX_SIZE = 256


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        print(f"Already exists: {OUT_PATH}; remove to rebuild.")
        return

    device = _default_device()
    lmdb_path = os.environ.get("PROBES_LMDB_PATH", LMDB_PATH)
    print(f"Loading {N_PROTEINS} proteins ≤{MAX_SIZE} residues from {lmdb_path}")
    batch, raw = load_proteina_batch(
        n=N_PROTEINS, max_size=MAX_SIZE, lmdb_path=lmdb_path, device=device
    )

    # Move to CPU for storage; downstream scripts move back to GPU as needed.
    batch_cpu = {
        k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
    }
    mask = batch_cpu["mask"].bool()  # [B, N]
    B, N = mask.shape
    n_real = int(mask.sum().item())
    print(f"Loaded {len(raw)} proteins; mask sum (real residues) = {n_real}")

    # Stable flattening: row-major over mask. Each downstream feature dump
    # uses the SAME mask, so this identical ordering is reproduced everywhere.
    # We persist (batch_i, residue_j) pairs so the manifest is debuggable.
    bi, rj = mask.nonzero(as_tuple=True)
    residue_index = torch.stack([bi, rj], dim=1).contiguous()  # [n_real, 2]

    payload = {
        "batch": batch_cpu,
        "residue_index": residue_index,  # [n_real, 2] int64
        "n_real_residues": n_real,
        "n_proteins": len(raw),
        "max_size": MAX_SIZE,
        "lmdb_path": str(lmdb_path),
    }
    torch.save(payload, OUT_PATH)
    print(f"Wrote {OUT_PATH}  (n_real_residues={n_real}, n_proteins={len(raw)})")


if __name__ == "__main__":
    main()
