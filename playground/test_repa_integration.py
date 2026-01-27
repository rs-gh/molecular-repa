"""Test script to verify REPA integration into TABASCO.

This script tests that:
1. Hidden states are extracted from TransformerModule
2. REPA loss is computed correctly
3. Gradients flow properly through the projector (but not encoder)
4. Stats are logged correctly
"""

import torch
from tensordict import TensorDict

# Import TABASCO components
from tabasco.models.components.transformer_module import TransformerModule
from tabasco.models.components.encoders import DummyEncoder, Projector
from tabasco.models.components.losses import REPALoss
from tabasco.flow.path import FlowPath


def test_hidden_states_extraction():
    """Test that TransformerModule can return hidden states."""
    print("\n" + "=" * 70)
    print("TEST 1: Hidden States Extraction")
    print("=" * 70)

    # Create a simple transformer
    hidden_dim = 64
    transformer = TransformerModule(
        spatial_dim=3,
        atom_dim=10,
        num_heads=4,
        num_layers=2,
        hidden_dim=hidden_dim,
        implementation="pytorch"
    )

    # Create dummy input
    batch_size = 2
    num_atoms = 10
    coords = torch.randn(batch_size, num_atoms, 3)
    atomics = torch.randn(batch_size, num_atoms, 10)
    atomics = torch.nn.functional.softmax(atomics, dim=-1)  # Make it one-hot-like
    padding_mask = torch.zeros(batch_size, num_atoms, dtype=torch.bool)
    padding_mask[:, 8:] = True  # Last 2 atoms are padding
    t = torch.rand(batch_size)

    # Test without hidden states
    output = transformer(coords, atomics, padding_mask, t, return_hidden_states=False)
    assert len(output) == 2, f"Expected 2 outputs, got {len(output)}"
    print(f"✓ Without hidden states: output shape = {[o.shape for o in output]}")

    # Test with hidden states
    output = transformer(coords, atomics, padding_mask, t, return_hidden_states=True)
    assert len(output) == 3, f"Expected 3 outputs, got {len(output)}"
    coords_out, atomics_out, hidden_states = output
    print(f"✓ With hidden states:")
    print(f"  - Coords shape: {coords_out.shape}")
    print(f"  - Atomics shape: {atomics_out.shape}")
    print(f"  - Hidden states shape: {hidden_states.shape}")

    assert hidden_states.shape == (batch_size, num_atoms, hidden_dim), \
        f"Expected hidden states shape {(batch_size, num_atoms, hidden_dim)}, got {hidden_states.shape}"

    print("✓ TEST 1 PASSED\n")
    return True


def test_repa_loss():
    """Test that REPA loss computes correctly."""
    print("=" * 70)
    print("TEST 2: REPA Loss Computation")
    print("=" * 70)

    # Create encoder and projector
    encoder_dim = 128
    hidden_dim = 64
    encoder = DummyEncoder(input_dim=3, hidden_dim=64, encoder_dim=encoder_dim)
    projector = Projector(hidden_dim=hidden_dim, encoder_dim=encoder_dim, num_layers=2)

    # Create REPA loss
    repa_loss = REPALoss(
        encoder=encoder,
        projector=projector,
        lambda_repa=0.5,
        time_weighting=False
    )

    # Verify encoder is frozen
    for param in repa_loss.encoder.parameters():
        assert not param.requires_grad, "Encoder should be frozen!"
    print("✓ Encoder is frozen")

    # Verify projector is trainable
    for param in repa_loss.projector.parameters():
        assert param.requires_grad, "Projector should be trainable!"
    print("✓ Projector is trainable")

    # Create dummy data
    batch_size = 2
    num_atoms = 10
    coords = torch.randn(batch_size, num_atoms, 3)
    atomics = torch.randn(batch_size, num_atoms, 10)
    atomics = torch.nn.functional.softmax(atomics, dim=-1)
    padding_mask = torch.zeros(batch_size, num_atoms, dtype=torch.bool)
    padding_mask[:, 8:] = True
    t = torch.rand(batch_size)
    hidden_states = torch.randn(batch_size, num_atoms, hidden_dim)

    # Create FlowPath
    x_1 = TensorDict(
        {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
        batch_size=batch_size
    )
    path = FlowPath(
        x_0=x_1,  # Dummy
        x_t=x_1,  # Dummy
        dx_t=x_1,  # Dummy
        x_1=x_1,  # Real clean molecules
        t=t
    )

    # Create pred with hidden states
    pred = TensorDict(
        {
            "coords": coords,
            "atomics": atomics,
            "hidden_states": hidden_states,
            "padding_mask": padding_mask
        },
        batch_size=batch_size
    )

    # Compute loss
    loss, stats = repa_loss(path, pred, compute_stats=True)

    print(f"✓ Loss computed: {loss.item():.6f}")
    print(f"✓ Stats: {stats}")

    assert "repa_loss" in stats, "Stats should contain repa_loss"
    assert "repa_alignment" in stats, "Stats should contain repa_alignment"
    assert loss.item() > 0, "Loss should be positive"

    print("✓ TEST 2 PASSED\n")
    return True


def test_gradient_flow():
    """Test that gradients flow through projector but not encoder."""
    print("=" * 70)
    print("TEST 3: Gradient Flow")
    print("=" * 70)

    # Create components
    encoder_dim = 128
    hidden_dim = 64
    encoder = DummyEncoder(input_dim=3, hidden_dim=64, encoder_dim=encoder_dim)
    projector = Projector(hidden_dim=hidden_dim, encoder_dim=encoder_dim, num_layers=2)

    repa_loss = REPALoss(
        encoder=encoder,
        projector=projector,
        lambda_repa=0.5,
        time_weighting=False
    )

    # Create dummy data
    batch_size = 2
    num_atoms = 10
    coords = torch.randn(batch_size, num_atoms, 3)
    atomics = torch.randn(batch_size, num_atoms, 10)
    atomics = torch.nn.functional.softmax(atomics, dim=-1)
    padding_mask = torch.zeros(batch_size, num_atoms, dtype=torch.bool)
    t = torch.rand(batch_size)
    hidden_states = torch.randn(batch_size, num_atoms, hidden_dim, requires_grad=True)

    x_1 = TensorDict(
        {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
        batch_size=batch_size
    )
    path = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t)
    pred = TensorDict(
        {
            "coords": coords,
            "atomics": atomics,
            "hidden_states": hidden_states,
            "padding_mask": padding_mask
        },
        batch_size=batch_size
    )

    # Compute loss and backpropagate
    loss, _ = repa_loss(path, pred, compute_stats=True)
    loss.backward()

    # Check gradients
    assert hidden_states.grad is not None, "Hidden states should have gradients"
    print(f"✓ Hidden states have gradients: {hidden_states.grad.abs().mean().item():.6f}")

    # Check projector has gradients
    has_grad = False
    for param in projector.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad, "Projector should have gradients"
    print("✓ Projector parameters have gradients")

    # Check encoder does NOT have gradients
    for param in encoder.parameters():
        assert param.grad is None, "Encoder should NOT have gradients (frozen)"
    print("✓ Encoder parameters have NO gradients (frozen)")

    print("✓ TEST 3 PASSED\n")
    return True


def test_time_weighting():
    """Test that time weighting works.

    Time weighting intuition:
    - t ≈ 1: molecule is clean → encoder gives meaningful embeddings → higher weight
    - t ≈ 0: molecule is noisy → encoder embeddings less meaningful → lower weight

    The REPA loss is: lambda * (-cos_sim) * time_weight
    Since cos_sim can be positive or negative, the loss can also be positive or negative.
    The key property is that |loss| scales linearly with time weight.
    """
    print("=" * 70)
    print("TEST 4: Time Weighting")
    print("=" * 70)

    encoder_dim = 128
    hidden_dim = 64
    encoder = DummyEncoder(input_dim=3, hidden_dim=64, encoder_dim=encoder_dim)
    projector = Projector(hidden_dim=hidden_dim, encoder_dim=encoder_dim, num_layers=2)

    # Create REPA loss with time weighting
    repa_with_weight = REPALoss(encoder, projector, lambda_repa=1.0, time_weighting=True)

    # Create dummy data - same for both time conditions
    batch_size = 2
    num_atoms = 10
    coords = torch.randn(batch_size, num_atoms, 3)
    atomics = torch.randn(batch_size, num_atoms, 10)
    atomics = torch.nn.functional.softmax(atomics, dim=-1)
    padding_mask = torch.zeros(batch_size, num_atoms, dtype=torch.bool)
    hidden_states = torch.randn(batch_size, num_atoms, hidden_dim)

    x_1 = TensorDict(
        {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
        batch_size=batch_size
    )
    pred = TensorDict(
        {
            "coords": coords,
            "atomics": atomics,
            "hidden_states": hidden_states,
            "padding_mask": padding_mask
        },
        batch_size=batch_size
    )

    # Test with t close to 0 (noisy molecules)
    t_low = torch.tensor([0.1, 0.1])
    path_low = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t_low)
    loss_low, _ = repa_with_weight(path_low, pred, compute_stats=False)

    # Test with t close to 1 (clean molecules)
    t_high = torch.tensor([0.9, 0.9])
    path_high = FlowPath(x_0=x_1, x_t=x_1, dx_t=x_1, x_1=x_1, t=t_high)
    loss_high, _ = repa_with_weight(path_high, pred, compute_stats=False)

    print(f"✓ Loss with time weighting (t~0.1): {loss_low.item():.6f}")
    print(f"✓ Loss with time weighting (t~0.9): {loss_high.item():.6f}")

    # The base alignment error is the same, so |loss| ratio should match the time ratio
    # |loss_high| / |loss_low| ≈ t_high.mean() / t_low.mean() = 0.9 / 0.1 = 9
    expected_ratio = t_high.mean().item() / t_low.mean().item()
    actual_ratio = abs(loss_high.item()) / abs(loss_low.item())
    print(f"✓ Expected |loss| ratio (t_high/t_low): {expected_ratio:.2f}")
    print(f"✓ Actual |loss| ratio: {actual_ratio:.2f}")

    # Verify the magnitude of weighted loss at t~1 is higher than at t~0
    assert abs(loss_high.item()) > abs(loss_low.item()), \
        f"|loss| at t~1 ({abs(loss_high.item()):.4f}) should be > |loss| at t~0 ({abs(loss_low.item()):.4f})"

    # Verify the ratio is approximately correct (allow some tolerance)
    assert abs(actual_ratio - expected_ratio) < 0.1, \
        f"|loss| ratio ({actual_ratio:.2f}) should be close to time ratio ({expected_ratio:.2f})"

    print("✓ TEST 4 PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TESTING REPA INTEGRATION INTO TABASCO")
    print("=" * 70)

    tests = [
        test_hidden_states_extraction,
        test_repa_loss,
        test_gradient_flow,
        test_time_weighting,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ TEST FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed}/{len(tests)} tests FAILED")
    else:
        print("         ALL TESTS PASSED! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
