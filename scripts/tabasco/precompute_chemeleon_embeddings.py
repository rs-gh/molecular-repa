"""Pre-compute CheMeleon embeddings for all unique SMILES in a dataset.

Since CheMeleon is frozen and 2-D (bond-graph only, no 3-D geometry), its
output for a given SMILES is constant across all training steps, epochs, and
conformers.  This script computes the embeddings once and stores them in an
LMDB so CachedChemPropEncoder can replace the on-the-fly encoder during
training, reducing per-step encoder time from ~7 s to ~milliseconds.

Output LMDB schema
------------------
Key:   canonical SMILES string (UTF-8 bytes)
Value: pickle of float16 numpy array, shape [n_atoms, encoder_dim]

Typical run (HPC A100 node, GEOM train ~1.1M conformers, ~450K unique SMILES)
---------------------------------------------------------------------------
    srun --partition=ampere --gres=gpu:1 --cpus-per-task=8 --time=04:00:00 --pty bash
    source /home/sr2173/git/molecular-repa/.venv/bin/activate
    cd /home/sr2173/git/molecular-repa
    python scripts/tabasco/precompute_chemeleon_embeddings.py \\
        --lmdb-in   src/tabasco/data/lmdb_geom/train.lmdb \\
        --lmdb-out  src/tabasco/data/chemeleon_geom/train_embeddings.lmdb \\
        --batch-size 512

Run for val and test splits too (they're small, takes minutes):
    python scripts/tabasco/precompute_chemeleon_embeddings.py \\
        --lmdb-in  src/tabasco/data/lmdb_geom/val.lmdb \\
        --lmdb-out src/tabasco/data/chemeleon_geom/val_embeddings.lmdb

Storage: ~100 KB per unique SMILES × ~450K unique → ~45 GB for the full
training set in float16.  The LMDB map_size is set conservatively at 200 GB
(virtual, not RSS) so it never needs to be extended.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

# ── project paths ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src" / "tabasco" / "src"
sys.path.insert(0, str(SRC_PATH))

CHEMELEON = Path.home() / ".chemprop" / "chemeleon_mp.pt"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--lmdb-in",
        required=True,
        help="Input LMDB produced by UnconditionalLMDBDataset (stores RDKit mols)",
    )
    p.add_argument(
        "--lmdb-out", required=True, help="Output LMDB path for pre-computed embeddings"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Molecules per CheMeleon forward pass (default 512)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip SMILES already present in lmdb-out (allows resuming)",
    )
    return p.parse_args()


def load_chemeleon(device: str):
    from chemprop.nn import BondMessagePassing

    if not CHEMELEON.exists():
        raise FileNotFoundError(
            f"CheMeleon weights not found at {CHEMELEON}. "
            "Run training once with pretrained='chemeleon' to trigger the download."
        )
    weights = torch.load(CHEMELEON, map_location="cpu", weights_only=True)
    mp = BondMessagePassing(**weights["hyper_parameters"])
    mp.load_state_dict(weights["state_dict"])
    mp = mp.to(device).eval()
    encoder_dim = weights["hyper_parameters"]["d_h"]
    print(f"Loaded CheMeleon ({encoder_dim}-dim) → {device}")
    return mp, encoder_dim


def iter_unique_smiles(lmdb_path: str):
    """Yield (canonical_smiles, n_heavy_atoms) for each unique SMILES in the LMDB.

    The LMDB stores RDKit Mol objects (with Hs already stripped) under
    sequential integer keys.  We deduplicate by canonical SMILES so each
    unique molecule is processed exactly once, regardless of how many
    conformers are stored.
    """
    db = lmdb.open(
        lmdb_path,
        map_size=10 * (1024**3),
        create=False,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    seen = set()
    with db.begin() as txn:
        cursor = txn.cursor()
        for val in cursor.iternext(keys=False, values=True):
            data = pickle.loads(val)
            mol = data.get("molecule")
            if mol is None:
                continue
            try:
                mol = Chem.RemoveAllHs(mol)
                smi = Chem.MolToSmiles(mol)
                if smi and smi not in seen:
                    seen.add(smi)
                    yield smi, mol.GetNumAtoms()
            except Exception:
                pass
    db.close()


def compute_embeddings_batch(smiles_batch, featurizer, mp, device):
    """Run CheMeleon on a list of SMILES; return list of float16 numpy arrays."""
    from chemprop.data import BatchMolGraph

    molgraphs, atom_counts, valid_idx = [], [], []
    for i, smi in enumerate(smiles_batch):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mg = featurizer(mol)
                molgraphs.append(mg)
                atom_counts.append(mol.GetNumAtoms())
                valid_idx.append(i)
        except Exception:
            pass

    if not molgraphs:
        return [None] * len(smiles_batch)

    bmg = BatchMolGraph(molgraphs)
    bmg.to(device)
    with torch.no_grad():
        atom_embs = mp(bmg)  # [total_atoms, encoder_dim]  on device

    atom_embs_cpu = atom_embs.cpu().to(torch.float16).numpy()

    results = [None] * len(smiles_batch)
    offset = 0
    for local_i, global_i in enumerate(valid_idx):
        n = atom_counts[local_i]
        results[global_i] = atom_embs_cpu[offset : offset + n]  # [n, D] float16
        offset += n

    return results


def main():
    args = parse_args()

    out_path = Path(args.lmdb_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mp, encoder_dim = load_chemeleon(args.device)

    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

    featurizer = SimpleMoleculeMolGraphFeaturizer()

    # Open (or create) output LMDB
    db_out = lmdb.open(
        str(out_path),
        map_size=200 * (1024**3),
        create=True,
        subdir=False,
        readonly=False,
    )

    # Count existing entries if resuming
    existing = set()
    if args.skip_existing:
        with db_out.begin() as txn:
            cursor = txn.cursor()
            for key in cursor.iternext(keys=True, values=False):
                existing.add(key.decode())
        print(f"  Skipping {len(existing):,} already-computed SMILES")

    print(f"\nScanning {args.lmdb_in} for unique SMILES …")
    all_unique = [
        (smi, n) for smi, n in iter_unique_smiles(args.lmdb_in) if smi not in existing
    ]
    print(
        f"  {len(all_unique):,} unique SMILES to embed "
        f"(encoder_dim={encoder_dim}, device={args.device})"
    )

    # Estimate output size
    avg_atoms = np.mean([n for _, n in all_unique]) if all_unique else 25
    est_gb = len(all_unique) * avg_atoms * encoder_dim * 2 / 1e9
    print(
        f"  Estimated output size: {est_gb:.1f} GB "
        f"(avg {avg_atoms:.0f} atoms/mol, float16)"
    )

    n_written = 0
    n_failed = 0
    t_start = time.perf_counter()

    smiles_only = [smi for smi, _ in all_unique]
    batch_size = args.batch_size

    with tqdm(total=len(smiles_only), unit="mol", desc="Embedding") as pbar:
        for start in range(0, len(smiles_only), batch_size):
            batch_smiles = smiles_only[start : start + batch_size]
            results = compute_embeddings_batch(
                batch_smiles, featurizer, mp, args.device
            )

            with db_out.begin(write=True) as txn:
                for smi, emb in zip(batch_smiles, results):
                    if emb is not None:
                        txn.put(smi.encode(), pickle.dumps(emb, protocol=4))
                        n_written += 1
                    else:
                        n_failed += 1

            pbar.update(len(batch_smiles))
            elapsed = time.perf_counter() - t_start
            rate = (start + len(batch_smiles)) / elapsed
            pbar.set_postfix(
                written=n_written, failed=n_failed, rate=f"{rate:.0f} mol/s"
            )

    db_out.close()

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Written:  {n_written:,} embeddings")
    print(f"  Failed:   {n_failed:,}")
    print(f"  Output:   {out_path}")
    print(f"  Rate:     {n_written/elapsed:.0f} mol/s")


if __name__ == "__main__":
    main()
