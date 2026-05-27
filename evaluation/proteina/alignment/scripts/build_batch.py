"""Build the frozen protein fixture used for all CKNNA cells.

The PDB val set (the held-out CKNNA target) has only 3190 proteins ≤256
residues, so the protein fixture is capped at 3000 (was 10000, which failed
the sidecar pool check). The per-residue subsample stays at 10k residues
(matches the REPA paper) — easily covered by ~3000 proteins.

Two derivatives are dumped alongside the manifest:
  - ``protein_keys``: LMDB keys for the 10k proteins (reservoir-sampled via the
    existing ``build_or_load_manifest`` helper), in deterministic order.
  - ``per_residue_indices``: 10k uniformly-sampled ``(protein_i, residue_j)``
    pairs across all real residues — used for the per-residue CKNNA matrix.
  - ``lengths``: real (unpadded) residue count per protein — used to mean-pool
    correctly during extraction.

All feature-extraction shards index into this fixture, so every CKNNA cell
is computed on identical samples.

Run:
    source .venv/bin/activate
    export PROJECT_ROOT=$(pwd)/src/proteina
    python evaluation/proteina/alignment/scripts/build_batch.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ALIGN_ROOT = HERE.parent
REP_ROOT = ALIGN_ROOT.parent / "representation"
PROTEINA_ROOT = Path("/home/sr2173/git/molecular-repa/src/proteina")

for p in (str(REP_ROOT), str(PROTEINA_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib import LMDB_PATH  # noqa: E402
from lib.manifest import build_or_load_manifest  # noqa: E402

OUT_DIR = ALIGN_ROOT / "results"
MANIFEST_PATH = OUT_DIR / "frozen_batch_n10k.json"
RESIDUE_INDEX_PATH = OUT_DIR / "frozen_batch_n10k_residues.pt"

N_PROTEINS = 3_000  # PDB val pool is 3190 proteins ≤256 residues (see module docstring)
MAX_SIZE = 256
SEED = 42
N_RESIDUES_SUBSAMPLE = 10_000  # for per-residue CKNNA matrix; matches REPA paper
RESIDUE_SEED = 43


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lmdb_path = os.environ.get("PROBES_LMDB_PATH", LMDB_PATH)

    # 1. Reservoir-sample 10k LMDB keys (≤256 residues each). Reuses existing
    #    representation/lib helper; manifest is JSON, ~few hundred KB.
    print(
        f"Sampling {N_PROTEINS} proteins (≤{MAX_SIZE} residues, seed={SEED}) "
        f"from {lmdb_path} ..."
    )
    manifest = build_or_load_manifest(
        outdir=OUT_DIR,
        version="cknna_n10k_v1",
        lmdb_path=lmdb_path,
        n=N_PROTEINS,
        max_size=MAX_SIZE,
        seed=SEED,
    )
    # `build_or_load_manifest` already wrote batch_manifest_cknna_n10k_v1.json.
    # We also write a friendlier-named copy at MANIFEST_PATH for self-contained
    # alignment artefacts.
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest at {MANIFEST_PATH}")

    lengths = manifest["lengths"]  # list[int], len = N_PROTEINS
    n_proteins = len(lengths)
    total_real_residues = sum(lengths)
    print(f"  {n_proteins} proteins, {total_real_residues} total real residues")

    # 2. Uniformly subsample residue positions across all real (non-padding)
    #    residues. Persist (protein_i, residue_j) pairs so every extraction
    #    shard knows which residues it needs to keep.
    print(
        f"Sampling {N_RESIDUES_SUBSAMPLE} residues uniformly (seed={RESIDUE_SEED})..."
    )
    g = torch.Generator(device="cpu").manual_seed(RESIDUE_SEED)
    flat_offsets = torch.cumsum(torch.tensor([0] + lengths[:-1]), dim=0)  # [N_PROTEINS]
    flat_idx = torch.randperm(total_real_residues, generator=g)[:N_RESIDUES_SUBSAMPLE]
    flat_idx, _ = flat_idx.sort()  # ascending for cleaner sharding
    # Convert each flat residue index to (protein_i, residue_j).
    # protein_i = largest p such that flat_offsets[p] <= flat_idx
    protein_i = torch.bucketize(flat_idx, flat_offsets, right=True) - 1
    residue_j = flat_idx - flat_offsets[protein_i]
    residue_index = torch.stack([protein_i, residue_j], dim=1).contiguous().long()
    # Sanity: every (p, j) should satisfy 0 <= j < lengths[p]
    bad = residue_index[:, 1] >= torch.tensor(lengths)[residue_index[:, 0]]
    assert not bad.any(), f"{bad.sum()} bad (protein,residue) pairs in subsample"

    payload = {
        "residue_index": residue_index,  # [10000, 2] int64
        "n_proteins": n_proteins,
        "n_real_residues_total": total_real_residues,
        "n_residues_subsample": N_RESIDUES_SUBSAMPLE,
        "max_size": MAX_SIZE,
        "lmdb_path": str(lmdb_path),
        "protein_keys": manifest["keys"],
        "lengths": lengths,
        "seed": SEED,
        "residue_seed": RESIDUE_SEED,
    }
    torch.save(payload, RESIDUE_INDEX_PATH)
    print(f"  wrote {RESIDUE_INDEX_PATH}")
    print(f"  residue_index shape: {tuple(residue_index.shape)}")


if __name__ == "__main__":
    main()
