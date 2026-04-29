#!/usr/bin/env python
"""End-to-end verification that the data pipeline delivers correct data to CheMeleon.

Tests the full chain:
    DataLoader -> TensorDictCollator -> apply_random_rotation -> ChemPropEncoder
for both QM9 and GEOM, checking that no data is silently dropped or corrupted.

Usage:
    source .venv/bin/activate
    export PROJECT_ROOT=$(pwd)/src/tabasco
    python encoder_profiling/tabasco/chemeleon/verify_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "tabasco" / "src"))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.logger().setLevel(RDLogger.CRITICAL)

from tabasco.chem.constants import ATOM_NAMES  # noqa: E402
from tabasco.data.components.lmdb_unconditional import UnconditionalLMDBDataset  # noqa: E402
from tabasco.data.utils import TensorDictCollator  # noqa: E402
from tabasco.data.transforms import apply_random_rotation  # noqa: E402
from tabasco.models.components.encoders import ChemPropEncoder  # noqa: E402

DATA_DIR = REPO_ROOT / "src" / "tabasco" / "data"
ATOM_DIM = len(ATOM_NAMES)  # 9


def load_dataset(name: str):
    if name == "qm9":
        lmdb_dir = str(DATA_DIR / "lmdb_qm9")
        data_dir = str(DATA_DIR / "processed_qm9_train.pt")
    elif name == "geom":
        lmdb_dir = str(DATA_DIR / "lmdb_geom")
        data_dir = str(DATA_DIR / "processed_geom_train.pt")
    else:
        raise ValueError(name)

    return UnconditionalLMDBDataset(
        data_dir=data_dir,
        split="train",
        add_random_rotation=True,  # Match production
        add_random_permutation=False,
        reorder_to_smiles_order=True,  # Match production
        remove_hydrogens=True,
        lmdb_dir=lmdb_dir,
    )


# ===================================================================
# Test 1: Individual items have SMILES
# ===================================================================
def test_individual_items(ds, ds_name: str, n_items: int = 500):
    print(f"\n  Test 1: Individual item SMILES presence ({ds_name})")
    rng = np.random.RandomState(42)
    indices = rng.choice(len(ds), size=min(n_items, len(ds)), replace=False)

    n_with_smiles = 0
    n_without_smiles = 0
    n_empty_smiles = 0
    n_invalid_smiles = 0

    for idx in indices:
        item = ds[int(idx)]
        try:
            smi = item.get_non_tensor("smiles")
        except (KeyError, AttributeError):
            n_without_smiles += 1
            continue

        if not smi or smi.strip() == "":
            n_empty_smiles += 1
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid_smiles += 1
            continue

        n_with_smiles += 1

    total = len(indices)
    print(f"    Total sampled:    {total}")
    print(f"    With valid SMILES: {n_with_smiles} ({100*n_with_smiles/total:.1f}%)")
    print(f"    Missing SMILES:   {n_without_smiles}")
    print(f"    Empty SMILES:     {n_empty_smiles}")
    print(f"    Invalid SMILES:   {n_invalid_smiles}")

    if n_without_smiles > 0 or n_empty_smiles > 0 or n_invalid_smiles > 0:
        print("    FAIL: Some items missing or have invalid SMILES")
        return False
    print("    PASS")
    return True


# ===================================================================
# Test 2: Batch collation preserves SMILES
# ===================================================================
def test_batch_collation(ds, ds_name: str, batch_size: int = 64, n_batches: int = 5):
    print(f"\n  Test 2: Batch collation preserves SMILES ({ds_name})")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=TensorDictCollator(),
        shuffle=True,
    )

    n_batches_checked = 0
    total_mols = 0
    n_smiles_present = 0
    n_smiles_missing = 0
    shape_errors = 0

    for batch in loader:
        if n_batches_checked >= n_batches:
            break

        B = batch["coords"].shape[0]
        N = batch["coords"].shape[1]
        total_mols += B

        # Check tensor shapes
        expected_coord_shape = (B, N, 3)
        expected_atomic_shape = (B, N, ATOM_DIM)
        expected_mask_shape = (B, N)

        if batch["coords"].shape != expected_coord_shape:
            print(
                f"    FAIL: coords shape {batch['coords'].shape} != {expected_coord_shape}"
            )
            shape_errors += 1
        if batch["atomics"].shape != expected_atomic_shape:
            print(
                f"    FAIL: atomics shape {batch['atomics'].shape} != {expected_atomic_shape}"
            )
            shape_errors += 1
        if batch["padding_mask"].shape != expected_mask_shape:
            print(
                f"    FAIL: mask shape {batch['padding_mask'].shape} != {expected_mask_shape}"
            )
            shape_errors += 1

        # Check SMILES
        try:
            smiles_list = list(batch.get_non_tensor("smiles"))
            if len(smiles_list) != B:
                print(f"    FAIL: SMILES count {len(smiles_list)} != batch size {B}")
                n_smiles_missing += B
            else:
                for smi in smiles_list:
                    if smi and smi.strip():
                        n_smiles_present += 1
                    else:
                        n_smiles_missing += 1
        except (KeyError, AttributeError):
            print("    FAIL: No SMILES field in batch")
            n_smiles_missing += B

        n_batches_checked += 1

    print(f"    Batches checked:  {n_batches_checked}")
    print(f"    Total molecules:  {total_mols}")
    print(f"    SMILES present:   {n_smiles_present}")
    print(f"    SMILES missing:   {n_smiles_missing}")
    print(f"    Shape errors:     {shape_errors}")

    ok = n_smiles_missing == 0 and shape_errors == 0
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


# ===================================================================
# Test 3: apply_random_rotation preserves SMILES
# ===================================================================
def test_augmentation_preserves_smiles(
    ds, ds_name: str, batch_size: int = 32, n_aug: int = 10
):
    print(f"\n  Test 3: apply_random_rotation preserves SMILES ({ds_name})")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=TensorDictCollator(),
        shuffle=True,
    )

    batch = next(iter(loader))
    B_orig = batch["coords"].shape[0]

    try:
        orig_smiles = list(batch.get_non_tensor("smiles"))
    except (KeyError, AttributeError):
        print("    FAIL: No SMILES in batch before augmentation")
        return False

    augmented = apply_random_rotation(batch, n_augmentations=n_aug)
    B_aug = augmented["coords"].shape[0]
    expected_B = B_orig * (n_aug + 1)

    if B_aug != expected_B:
        print(f"    FAIL: Augmented batch size {B_aug} != expected {expected_B}")
        return False

    try:
        aug_smiles = list(augmented.get_non_tensor("smiles"))
    except (KeyError, AttributeError):
        print("    FAIL: SMILES lost during augmentation!")
        return False

    if len(aug_smiles) != expected_B:
        print(f"    FAIL: Augmented SMILES count {len(aug_smiles)} != {expected_B}")
        return False

    # Verify block-wise repetition: each block of B_orig should have identical SMILES
    for aug_idx in range(n_aug + 1):
        block = aug_smiles[aug_idx * B_orig : (aug_idx + 1) * B_orig]
        if block != orig_smiles:
            print(f"    FAIL: Augmentation block {aug_idx} SMILES don't match original")
            return False

    # Verify no empty SMILES
    n_empty = sum(1 for s in aug_smiles if not s or not s.strip())
    if n_empty > 0:
        print(f"    FAIL: {n_empty} empty SMILES after augmentation")
        return False

    print(f"    Original batch:  {B_orig} molecules")
    print(f"    Augmented batch: {B_aug} molecules ({n_aug+1}x)")
    print(f"    SMILES count:    {len(aug_smiles)} (all present)")
    print(f"    Block repetition verified for all {n_aug+1} augmentations")
    print("    PASS")
    return True


# ===================================================================
# Test 4: ChemPropEncoder produces non-zero embeddings for all real atoms
# ===================================================================
def test_encoder_outputs(
    ds, ds_name: str, encoder, batch_size: int = 64, n_batches: int = 10
):
    print(f"\n  Test 4: ChemPropEncoder output validity ({ds_name})")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=TensorDictCollator(),
        shuffle=True,
    )

    total_mols = 0
    total_real_atoms = 0
    total_padded_atoms = 0
    n_zero_real_atoms = 0
    n_nonzero_padded = 0
    n_nan_or_inf = 0
    n_encoder_failures = 0
    embedding_norms = []

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        B = batch["coords"].shape[0]
        total_mols += B

        try:
            smiles_list = list(batch.get_non_tensor("smiles"))
        except (KeyError, AttributeError):
            smiles_list = None
            print(f"    WARNING: Batch {i} has no SMILES, using slow path")

        with torch.no_grad():
            emb = encoder(
                batch["coords"],
                batch["atomics"],
                batch["padding_mask"],
                smiles=smiles_list,
            )

        # Check for NaN/Inf
        if torch.isnan(emb).any() or torch.isinf(emb).any():
            n_nan_or_inf += (
                torch.isnan(emb).sum().item() + torch.isinf(emb).sum().item()
            )

        for j in range(B):
            mask = batch["padding_mask"][j]
            n_real = (~mask).sum().item()
            n_pad = mask.sum().item()
            total_real_atoms += n_real
            total_padded_atoms += n_pad

            # Check real atoms have non-zero embeddings
            real_emb = emb[j, :n_real]
            real_norms = real_emb.norm(dim=-1)
            n_zero = (real_norms == 0).sum().item()
            n_zero_real_atoms += n_zero

            if n_zero > 0 and smiles_list is not None:
                n_encoder_failures += 1

            embedding_norms.extend(real_norms.tolist())

            # Check padded atoms have zero embeddings
            if n_pad > 0:
                pad_emb = emb[j, n_real:]
                n_nonzero_pad = (pad_emb.norm(dim=-1) > 0).sum().item()
                n_nonzero_padded += n_nonzero_pad

    embedding_norms = np.array(embedding_norms)

    print(f"    Batches checked:    {min(i + 1, n_batches)}")
    print(f"    Total molecules:    {total_mols}")
    print(f"    Total real atoms:   {total_real_atoms}")
    print(f"    Total padded atoms: {total_padded_atoms}")
    print(
        f"    Zero-embedding real atoms: {n_zero_real_atoms} "
        f"({100*n_zero_real_atoms/max(total_real_atoms,1):.2f}%)"
    )
    print(f"    Non-zero padded atoms:     {n_nonzero_padded}")
    print(f"    NaN/Inf values:            {n_nan_or_inf}")
    print(f"    Encoder failures (zero emb with SMILES): {n_encoder_failures}")
    print(
        f"    Embedding norm: mean={embedding_norms.mean():.2f}, "
        f"std={embedding_norms.std():.2f}, "
        f"min={embedding_norms.min():.2f}, max={embedding_norms.max():.2f}"
    )

    ok = n_zero_real_atoms == 0 and n_nonzero_padded == 0 and n_nan_or_inf == 0
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


# ===================================================================
# Test 5: SMILES-to-atom-type consistency
# ===================================================================
def test_smiles_atom_consistency(ds, ds_name: str, n_items: int = 500):
    print(f"\n  Test 5: SMILES atom types match tensor atom types ({ds_name})")
    rng = np.random.RandomState(42)
    indices = rng.choice(len(ds), size=min(n_items, len(ds)), replace=False)

    n_checked = 0
    n_match = 0
    n_mismatch = 0
    mismatch_examples = []

    for idx in indices:
        item = ds[int(idx)]
        try:
            smi = item.get_non_tensor("smiles")
        except (KeyError, AttributeError):
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        n_real = (~item["padding_mask"]).sum().item()
        smi_atoms = [
            mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumHeavyAtoms())
        ]
        tensor_atoms = [
            ATOM_NAMES[item["atomics"][j].argmax().item()] for j in range(n_real)
        ]

        n_checked += 1
        if smi_atoms == tensor_atoms:
            n_match += 1
        else:
            n_mismatch += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append((smi, smi_atoms, tensor_atoms))

    print(f"    Checked:    {n_checked}")
    print(f"    Matching:   {n_match} ({100*n_match/max(n_checked,1):.1f}%)")
    print(f"    Mismatched: {n_mismatch}")
    if mismatch_examples:
        for smi, smi_a, ten_a in mismatch_examples:
            print(f"      {smi}: SMILES={smi_a}, tensor={ten_a}")

    ok = n_mismatch == 0
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


# ===================================================================
# Test 6: Full augmented pipeline through encoder
# ===================================================================
def test_full_augmented_pipeline(
    ds, ds_name: str, encoder, batch_size: int = 32, n_aug: int = 10
):
    print(f"\n  Test 6: Full augmented pipeline -> encoder ({ds_name})")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=TensorDictCollator(),
        shuffle=True,
    )

    batch = next(iter(loader))
    B_orig = batch["coords"].shape[0]

    # Apply augmentation (as FlowMatchingModel.forward does)
    augmented = apply_random_rotation(batch, n_augmentations=n_aug)
    B_aug = augmented["coords"].shape[0]

    try:
        smiles = list(augmented.get_non_tensor("smiles"))
    except (KeyError, AttributeError):
        print("    FAIL: No SMILES after augmentation")
        return False

    if len(smiles) != B_aug:
        print(f"    FAIL: SMILES count {len(smiles)} != batch size {B_aug}")
        return False

    # Run encoder on augmented batch
    with torch.no_grad():
        emb = encoder(
            augmented["coords"],
            augmented["atomics"],
            augmented["padding_mask"],
            smiles=smiles,
        )

    # Check shape
    N = augmented["coords"].shape[1]
    encoder_dim = encoder.encoder_dim
    expected_shape = (B_aug, N, encoder_dim)
    if emb.shape != expected_shape:
        print(f"    FAIL: Output shape {emb.shape} != expected {expected_shape}")
        return False

    # Check all real atoms have non-zero embeddings
    n_zero = 0
    total_real = 0
    for j in range(B_aug):
        n_real = (~augmented["padding_mask"][j]).sum().item()
        total_real += n_real
        real_norms = emb[j, :n_real].norm(dim=-1)
        n_zero += (real_norms == 0).sum().item()

    # Check NaN/Inf
    has_nan = torch.isnan(emb).any().item()
    has_inf = torch.isinf(emb).any().item()

    # Check that augmented copies of the same molecule get identical embeddings
    # (CheMeleon is 2D, so rotation should not change embeddings when using SMILES path)
    n_rotation_mismatches = 0
    for mol_idx in range(B_orig):
        ref_emb = emb[mol_idx]  # First copy (augmentation 0)
        for aug_idx in range(1, n_aug + 1):
            aug_emb = emb[aug_idx * B_orig + mol_idx]
            if not torch.allclose(ref_emb, aug_emb, atol=1e-6):
                n_rotation_mismatches += 1
                break  # One mismatch per molecule is enough

    print(f"    Augmented batch: {B_aug} molecules ({B_orig} x {n_aug+1})")
    print(f"    Output shape:    {emb.shape}")
    print(f"    Real atoms:      {total_real}")
    print(f"    Zero-emb atoms:  {n_zero}")
    print(f"    NaN: {has_nan}, Inf: {has_inf}")
    print(
        f"    Rotation invariance: {B_orig - n_rotation_mismatches}/{B_orig} molecules "
        f"have identical embeddings across augmentations"
    )
    if n_rotation_mismatches > 0:
        print(f"    NOTE: {n_rotation_mismatches} molecules differ across rotations.")
        print(
            "      This is EXPECTED if per-item random_rotation in __getitem__ changes coords"
        )
        print(
            "      before SMILES is computed. But since SMILES is computed from the stored mol"
        )
        print(
            "      (not coords), the SMILES-based encoder should be rotation-invariant."
        )

    ok = n_zero == 0 and not has_nan and not has_inf
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


# ===================================================================
# Test 7: No data loss - check dataset length vs DataLoader yield
# ===================================================================
def test_no_data_loss(ds, ds_name: str, batch_size: int = 256):
    print(f"\n  Test 7: No data loss in DataLoader ({ds_name})")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=TensorDictCollator(),
        shuffle=False,  # Deterministic for counting
    )

    total_items_yielded = 0
    n_batches = 0
    for batch in loader:
        total_items_yielded += batch["coords"].shape[0]
        n_batches += 1

    expected = len(ds)
    print(f"    Dataset length:     {expected}")
    print(f"    Items from loader:  {total_items_yielded}")
    print(f"    Batches:            {n_batches}")
    print(f"    Last batch size:    {expected - batch_size * (n_batches - 1)}")

    ok = total_items_yielded == expected
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 70)
    print("Pipeline Verification: DataLoader -> Collation -> Augmentation -> Encoder")
    print("=" * 70)

    encoder = ChemPropEncoder(pretrained="chemeleon")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Loaded CheMeleon encoder (dim={encoder.encoder_dim})")

    all_results = {}

    for ds_name in ["qm9", "geom"]:
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name.upper()}")
        print(f"{'='*70}")

        ds = load_dataset(ds_name)
        print(f"  Dataset size: {len(ds)}")

        results = {}
        results["items"] = test_individual_items(ds, ds_name)
        results["collation"] = test_batch_collation(ds, ds_name)
        results["augmentation"] = test_augmentation_preserves_smiles(ds, ds_name)
        results["encoder"] = test_encoder_outputs(ds, ds_name, encoder)
        results["atom_consistency"] = test_smiles_atom_consistency(ds, ds_name)
        results["full_pipeline"] = test_full_augmented_pipeline(ds, ds_name, encoder)

        # Skip full data loss test for GEOM (too slow - 1.1M items)
        if ds_name == "qm9":
            results["no_data_loss"] = test_no_data_loss(ds, ds_name)
        else:
            print("\n  Test 7: Skipped for GEOM (too large for full iteration)")
            results["no_data_loss"] = None

        all_results[ds_name] = results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Test':<45s} {'QM9':>6s} {'GEOM':>6s}")
    print(f"  {'-'*45} {'-'*6} {'-'*6}")

    test_names = {
        "items": "Individual items have SMILES",
        "collation": "Batch collation preserves SMILES",
        "augmentation": "Augmentation preserves SMILES",
        "encoder": "Encoder outputs valid (non-zero, finite)",
        "atom_consistency": "SMILES atom types match tensors",
        "full_pipeline": "Full augmented pipeline -> encoder",
        "no_data_loss": "No data loss in DataLoader",
    }

    all_pass = True
    for key, label in test_names.items():
        qm9_r = all_results["qm9"].get(key)
        geom_r = all_results["geom"].get(key)
        qm9_str = "PASS" if qm9_r else ("SKIP" if qm9_r is None else "FAIL")
        geom_str = "PASS" if geom_r else ("SKIP" if geom_r is None else "FAIL")
        print(f"  {label:<45s} {qm9_str:>6s} {geom_str:>6s}")
        if qm9_r is False or geom_r is False:
            all_pass = False

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print(f"{'='*70}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
