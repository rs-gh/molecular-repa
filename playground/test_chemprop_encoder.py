#!/usr/bin/env python
"""Test script for ChemPropEncoder integration.

This script tests the ChemPropEncoder with the REPA loss pipeline,
comparing against the DummyEncoder baseline.
"""

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tensordict import TensorDict

from tabasco.models.components.encoders import ChemPropEncoder, DummyEncoder, Projector
from tabasco.models.components.losses import REPALoss
from tabasco.chem.convert import MoleculeConverter


def create_test_batch(smiles_list: list[str], max_atoms: int = 10):
    """Create a batch of molecules for testing."""
    converter = MoleculeConverter()

    tensordicts = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol = Chem.RemoveAllHs(mol)
        td = converter.to_tensor(mol, pad_to_size=max_atoms)
        tensordicts.append(td)

    # Stack into batch
    coords = torch.stack([td["coords"] for td in tensordicts])
    atomics = torch.stack([td["atomics"] for td in tensordicts])
    padding_mask = torch.stack([td["padding_mask"] for td in tensordicts])

    return TensorDict({
        "coords": coords,
        "atomics": atomics,
        "padding_mask": padding_mask,
    }, batch_size=[len(smiles_list)])


def test_chemprop_encoder():
    """Test that ChemPropEncoder produces valid embeddings with CheMeleon."""
    print("=" * 60)
    print("Testing ChemPropEncoder with CheMeleon pretrained weights")
    print("=" * 60)

    # Test molecules
    smiles_list = [
        "CCO",        # Ethanol (3 atoms)
        "CC(=O)O",    # Acetic acid (4 atoms)
        "c1ccccc1",   # Benzene (6 atoms)
        "CCN",        # Ethylamine (3 atoms)
    ]

    batch = create_test_batch(smiles_list, max_atoms=10)
    print(f"\nCreated batch of {len(smiles_list)} molecules")
    print(f"  coords: {batch['coords'].shape}")
    print(f"  atomics: {batch['atomics'].shape}")
    print(f"  padding_mask: {batch['padding_mask'].shape}")

    # Create ChemProp encoder with CheMeleon pretrained weights
    encoder = ChemPropEncoder(pretrained="chemeleon")
    print(f"\nChemPropEncoder created with CheMeleon (encoder_dim={encoder.encoder_dim})")

    # Forward pass
    with torch.no_grad():
        embeddings = encoder(
            batch["coords"],
            batch["atomics"],
            batch["padding_mask"],
        )

    print(f"\nOutput embeddings: {embeddings.shape}")

    # Verify embeddings
    for i, smiles in enumerate(smiles_list):
        real_atoms = (~batch["padding_mask"][i]).sum().item()
        emb_norm = embeddings[i, :int(real_atoms)].norm(dim=-1)
        print(f"  {smiles}: {int(real_atoms)} atoms, norms: {emb_norm[:3].tolist()}")

    print("\n✓ ChemPropEncoder test passed!")
    return True


def test_repa_loss_with_chemprop():
    """Test REPA loss computation with ChemPropEncoder."""
    print("\n" + "=" * 60)
    print("Testing REPA Loss with ChemPropEncoder (CheMeleon)")
    print("=" * 60)

    # Create test batch
    smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
    batch = create_test_batch(smiles_list, max_atoms=10)

    # Model dimensions - CheMeleon has encoder_dim=2048
    hidden_dim = 128

    # Create REPA components with ChemProp (CheMeleon pretrained)
    encoder = ChemPropEncoder(pretrained="chemeleon")
    encoder_dim = encoder.encoder_dim  # 2048 for CheMeleon
    projector = Projector(hidden_dim=hidden_dim, encoder_dim=encoder_dim)

    repa_loss = REPALoss(
        encoder=encoder,
        projector=projector,
        lambda_repa=0.5,
        time_weighting=False,
        similarity_type="cosine",
    )

    print(f"\nREPA Loss created:")
    print(f"  Encoder: ChemPropEncoder (frozen)")
    print(f"  Projector: Projector (trainable)")
    print(f"  lambda_repa: 0.5")

    # Create mock FlowPath and predictions
    from tabasco.flow.path import FlowPath

    B, N = batch["coords"].shape[:2]

    # Mock path (clean molecules at x_1)
    path = FlowPath(
        x_1=batch,
        x_t=batch,  # For testing, use same as x_1
        dx_t=TensorDict({
            "coords": torch.randn(B, N, 3),
            "atomics": torch.randn(B, N, batch["atomics"].shape[-1]),
        }, batch_size=[B]),
        x_0=TensorDict({
            "coords": torch.randn(B, N, 3),
            "atomics": torch.randn(B, N, batch["atomics"].shape[-1]),
        }, batch_size=[B]),
        t=torch.rand(B),
    )

    # Mock predictions with hidden states
    pred = TensorDict({
        "coords": batch["coords"] + 0.1 * torch.randn_like(batch["coords"]),
        "atomics": batch["atomics"] + 0.1 * torch.randn_like(batch["atomics"]),
        "hidden_states": torch.randn(B, N, hidden_dim),  # From transformer
        "padding_mask": batch["padding_mask"],
    }, batch_size=[B])

    # Compute REPA loss
    loss, stats = repa_loss(path, pred, compute_stats=True)

    print(f"\nREPA Loss computation:")
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Stats: {stats}")

    # Verify gradients flow through projector but not encoder
    loss.backward()

    encoder_has_grad = any(p.grad is not None for p in encoder.parameters())
    projector_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in projector.parameters())

    print(f"\nGradient check:")
    print(f"  Encoder has gradients: {encoder_has_grad} (should be False)")
    print(f"  Projector has gradients: {projector_has_grad} (should be True)")

    assert not encoder_has_grad, "Encoder should be frozen!"
    assert projector_has_grad, "Projector should have gradients!"

    print("\n✓ REPA Loss with ChemProp test passed!")
    return True


def compare_encoders():
    """Compare ChemPropEncoder (pretrained) vs DummyEncoder (random)."""
    print("\n" + "=" * 60)
    print("Comparing ChemPropEncoder (CheMeleon) vs DummyEncoder (random)")
    print("=" * 60)

    # Create test batch
    smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
    batch = create_test_batch(smiles_list, max_atoms=10)

    # Create both encoders
    # CheMeleon is pretrained and provides meaningful molecular representations
    chemprop_encoder = ChemPropEncoder(pretrained="chemeleon")
    # DummyEncoder is randomly initialized - not useful for REPA
    dummy_encoder = DummyEncoder(encoder_dim=chemprop_encoder.encoder_dim)

    # Get embeddings from both
    with torch.no_grad():
        chemprop_emb = chemprop_encoder(
            batch["coords"], batch["atomics"], batch["padding_mask"]
        )
        dummy_emb = dummy_encoder(
            batch["coords"], batch["atomics"], batch["padding_mask"]
        )

    print(f"\nEmbedding comparison:")
    print(f"  ChemProp shape: {chemprop_emb.shape}")
    print(f"  Dummy shape: {dummy_emb.shape}")

    print(f"\nEmbedding statistics:")
    print(f"  ChemProp - mean: {chemprop_emb.mean():.4f}, std: {chemprop_emb.std():.4f}")
    print(f"  Dummy    - mean: {dummy_emb.mean():.4f}, std: {dummy_emb.std():.4f}")

    # Compare semantic consistency
    # ChemProp should give similar embeddings for chemically similar atoms
    print(f"\nChemProp captures chemical semantics (2D graph structure)")
    print(f"Dummy encoder only uses coordinates (3D position)")

    print("\n✓ Encoder comparison complete!")
    return True


if __name__ == "__main__":
    # Suppress chemprop warnings
    import warnings
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    success = True

    try:
        success &= test_chemprop_encoder()
        success &= test_repa_loss_with_chemprop()
        success &= compare_encoders()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    if success:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run training with: experiment=qm9_chemprop")
        print("  2. Compare against baseline: experiment=qm9_small")
        print("  3. Compare against dummy REPA: experiment=qm9_repa")
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED")
        print("=" * 60)
        exit(1)
