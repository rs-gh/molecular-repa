"""Pre-compute MACE embeddings for all conformers in a dataset.

Unlike CheMeleon (2-D, same embedding per SMILES), MACE is 3-D aware so
different conformers of the same molecule produce different embeddings.
This script computes per-conformer embeddings and stores them in an LMDB
keyed by the source LMDB key (byte-string integer), enabling
CachedMACEEncoder to replace the on-the-fly encoder during training.

MACE is rotation-invariant (l_max=0, invariants_only=True), so the
pre-computed embeddings are valid for any rotation of the same conformer.

Output LMDB schema
------------------
Key:   source LMDB key string (e.g. b"0", b"1", b"10" — same byte keys)
Value: pickle of float16 numpy array, shape [n_heavy_atoms, encoder_dim]

Typical run (HPC A100 node, GEOM train ~1.1M conformers)
---------------------------------------------------------
    srun --partition=ampere --gres=gpu:1 --cpus-per-task=8 --time=04:00:00 --pty bash
    source /home/sr2173/git/molecular-repa/.venv/bin/activate
    cd /home/sr2173/git/molecular-repa
    python playground/tabasco/mace/precompute_embeddings.py \
        --lmdb-in   src/tabasco/data/lmdb_geom/train.lmdb \
        --lmdb-out  /rds/user/sr2173/hpc-work/tabasco/data/mace_geom/train_embeddings.lmdb \
        --batch-size 256

Run for val and test splits too (they're small, takes minutes):
    python playground/tabasco/mace/precompute_embeddings.py \
        --lmdb-in  src/tabasco/data/lmdb_geom/val.lmdb \
        --lmdb-out /rds/user/sr2173/hpc-work/tabasco/data/mace_geom/val_embeddings.lmdb

Storage: GEOM ~11 GB, QM9 ~0.3 GB (float16).
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import lmdb
import torch
from rdkit import Chem
from tqdm import tqdm

# ── project paths ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src" / "tabasco" / "src"
sys.path.insert(0, str(SRC_PATH))

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


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
        "--lmdb-out",
        required=True,
        help="Output LMDB path for pre-computed MACE embeddings",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Molecules per MACE forward pass (default 256)",
    )
    p.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Max atoms for padding (auto-detected from data if not set)",
    )
    p.add_argument(
        "--model-name",
        default="small",
        choices=["small", "medium", "large"],
        help="MACE-OFF model size (default: small)",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip keys already present in lmdb-out (allows resuming)",
    )
    return p.parse_args()


def iter_entries(lmdb_path: str):
    """Yield (lmdb_key_str, rdkit_mol) for each entry in the source LMDB."""
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
    with db.begin() as txn:
        cursor = txn.cursor()
        for key, val in cursor.iternext(keys=True, values=True):
            data = pickle.loads(val)
            mol = data.get("molecule")
            if mol is None:
                continue
            try:
                mol = Chem.RemoveAllHs(mol)
                if mol.GetNumAtoms() > 0:
                    yield key.decode(), mol
            except Exception:
                pass
    db.close()


def compute_embeddings_batch(batch_keys, batch_mols, encoder, max_atoms, device):
    """Run MACEEncoder on a batch of RDKit mols.

    Returns list of (key, float16_numpy_array) tuples.
    """
    from tabasco.chem.convert import MoleculeConverter

    converter = MoleculeConverter()

    # Convert mols to tensors
    coords_list, atomics_list, padding_list = [], [], []
    valid_keys = []
    atom_counts = []

    for key, mol in zip(batch_keys, batch_mols):
        try:
            td = converter.to_tensor(mol, pad_to_size=max_atoms)
            coords_list.append(td["coords"])
            atomics_list.append(td["atomics"])
            padding_list.append(td["padding_mask"])
            valid_keys.append(key)
            atom_counts.append(mol.GetNumAtoms())
        except Exception:
            pass

    if not valid_keys:
        return []

    coords = torch.stack(coords_list).to(device)
    atomics = torch.stack(atomics_list).to(device)
    padding_mask = torch.stack(padding_list).to(device)

    with torch.no_grad():
        embeddings = encoder(coords, atomics, padding_mask)  # [B, N, D]

    embeddings_cpu = embeddings.cpu().to(torch.float16).numpy()

    results = []
    for i, key in enumerate(valid_keys):
        n = atom_counts[i]
        emb = embeddings_cpu[i, :n]  # [n_atoms, D] float16
        results.append((key, emb))

    return results


def main():
    args = parse_args()

    out_path = Path(args.lmdb_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load MACE encoder
    from tabasco.models.components.encoders import MACEEncoder

    encoder = MACEEncoder(model_name=args.model_name)
    encoder = encoder.to(args.device).eval()
    encoder_dim = encoder.encoder_dim
    print(f"Loaded MACE-OFF {args.model_name} ({encoder_dim}-dim) → {args.device}")

    # Auto-detect max_atoms if not specified
    if args.max_atoms is None:
        print(f"\nScanning {args.lmdb_in} for max atom count …")
        max_atoms = 0
        n_entries = 0
        total_atoms = 0
        for _key, mol in iter_entries(args.lmdb_in):
            n = mol.GetNumAtoms()
            max_atoms = max(max_atoms, n)
            total_atoms += n
            n_entries += 1
        args.max_atoms = max_atoms
        avg_atoms = total_atoms / n_entries if n_entries else 0
        print(
            f"  {n_entries:,} entries, max {max_atoms} atoms, "
            f"avg {avg_atoms:.1f} atoms"
        )
    else:
        # Still need to count entries for progress bar
        n_entries = 0
        for _key, _mol in iter_entries(args.lmdb_in):
            n_entries += 1
        avg_atoms = args.max_atoms / 2  # rough estimate

    # Estimate output size
    est_gb = n_entries * avg_atoms * encoder_dim * 2 / 1e9
    print(
        f"  Estimated output size: {est_gb:.1f} GB "
        f"(avg {avg_atoms:.0f} atoms/mol, float16)"
    )

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
        print(f"  Skipping {len(existing):,} already-computed entries")

    n_written = 0
    n_failed = 0
    n_skipped = 0
    t_start = time.perf_counter()
    batch_keys, batch_mols = [], []

    with tqdm(total=n_entries, unit="mol", desc="Embedding") as pbar:
        for key, mol in iter_entries(args.lmdb_in):
            if key in existing:
                n_skipped += 1
                pbar.update(1)
                continue

            batch_keys.append(key)
            batch_mols.append(mol)

            if len(batch_keys) >= args.batch_size:
                results = compute_embeddings_batch(
                    batch_keys, batch_mols, encoder, args.max_atoms, args.device
                )
                with db_out.begin(write=True) as txn:
                    for k, emb in results:
                        txn.put(k.encode(), pickle.dumps(emb, protocol=4))
                        n_written += 1
                n_failed += len(batch_keys) - len(results)
                pbar.update(len(batch_keys))
                elapsed = time.perf_counter() - t_start
                rate = (n_written + n_failed) / elapsed if elapsed > 0 else 0
                pbar.set_postfix(
                    written=n_written, failed=n_failed, rate=f"{rate:.0f} mol/s"
                )
                batch_keys, batch_mols = [], []

        # Final partial batch
        if batch_keys:
            results = compute_embeddings_batch(
                batch_keys, batch_mols, encoder, args.max_atoms, args.device
            )
            with db_out.begin(write=True) as txn:
                for k, emb in results:
                    txn.put(k.encode(), pickle.dumps(emb, protocol=4))
                    n_written += 1
            n_failed += len(batch_keys) - len(results)
            pbar.update(len(batch_keys))

    db_out.close()

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Written:  {n_written:,} embeddings")
    print(f"  Skipped:  {n_skipped:,} (already in cache)")
    print(f"  Failed:   {n_failed:,}")
    print(f"  Output:   {out_path}")
    if elapsed > 0:
        print(f"  Rate:     {n_written/elapsed:.0f} mol/s")


if __name__ == "__main__":
    main()
