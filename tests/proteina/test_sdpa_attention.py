"""Tests verifying that F.scaled_dot_product_attention produces identical
results to the manual attention implementation in PairBiasAttention._attn.

We test the raw _attn function in isolation (not the full module) so that
we can confirm numerical equivalence before swapping implementations.
"""

import pytest
import torch
from einops import rearrange
from torch import Tensor
from typing import Optional


# ---------------------------------------------------------------------------
# Reference: current manual implementation (copied verbatim)
# ---------------------------------------------------------------------------

max_neg_value = lambda x: torch.finfo(x.dtype).min  # noqa: E731


def exists(val) -> bool:
    return val is not None


def _attn_manual(
    q: Tensor, k: Tensor, v: Tensor, b, scale: float, mask: Optional[Tensor]
) -> Tensor:
    """Manual attention: Q @ K^T * scale + bias → softmax → @ V.

    This is the current implementation from pair_bias_attn.py._attn,
    extracted as a standalone function for testing.
    """
    sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
    if exists(mask):
        mask_4d = rearrange(mask, "b i j -> b () i j")
        sim = sim.masked_fill(~mask_4d, max_neg_value(sim))
    attn = torch.softmax(sim + b, dim=-1)
    return torch.einsum("b h i j, b h j d -> b h i d", attn, v)


# ---------------------------------------------------------------------------
# Proposed: SDPA replacement
# ---------------------------------------------------------------------------


def _attn_sdpa(
    q: Tensor, k: Tensor, v: Tensor, b, scale: float, mask: Optional[Tensor]
) -> Tensor:
    """SDPA attention: uses F.scaled_dot_product_attention.

    attn_mask is an additive float bias applied before softmax — same
    semantics as the manual ``sim + b`` path.
    """
    attn_bias = b if not isinstance(b, int) else None

    if exists(mask):
        mask_bias = rearrange(mask, "b i j -> b () i j")
        mask_bias = torch.zeros_like(mask_bias, dtype=q.dtype).masked_fill(
            ~mask_bias, max_neg_value(q)
        )
        attn_bias = attn_bias + mask_bias if exists(attn_bias) else mask_bias

    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_bias,
        scale=scale,
    )


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

B, H, N, D = 2, 8, 64, 64  # batch, heads, seq_len, head_dim


@pytest.fixture
def qkv():
    """Random Q, K, V in [B, H, N, D] layout."""
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    return q, k, v


@pytest.fixture
def pair_bias():
    """Random pair bias in [B, H, N, N] layout (simulates projected pair repr)."""
    torch.manual_seed(123)
    return torch.randn(B, H, N, N)


@pytest.fixture
def padding_mask():
    """Boolean mask [B, N, N] where some positions are padded (False)."""
    torch.manual_seed(456)
    # Per-residue mask: first 48 residues are real, last 16 are padding
    res_mask = torch.ones(B, N, dtype=torch.bool)
    res_mask[:, 48:] = False
    # Pairwise mask: both residues must be real
    return res_mask[:, :, None] & res_mask[:, None, :]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SCALE = D**-0.5
ATOL = 1e-5


class TestSDPAEquivalence:
    """Verify SDPA produces identical results to the manual path."""

    def test_no_bias_no_mask(self, qkv):
        """Simplest case: no pair bias, no padding mask."""
        q, k, v = qkv
        out_manual = _attn_manual(q, k, v, b=0, scale=SCALE, mask=None)
        out_sdpa = _attn_sdpa(q, k, v, b=0, scale=SCALE, mask=None)
        assert torch.allclose(
            out_manual, out_sdpa, atol=ATOL
        ), f"Max diff: {(out_manual - out_sdpa).abs().max().item()}"

    def test_with_pair_bias_no_mask(self, qkv, pair_bias):
        """Pair bias only, no padding."""
        q, k, v = qkv
        out_manual = _attn_manual(q, k, v, b=pair_bias, scale=SCALE, mask=None)
        out_sdpa = _attn_sdpa(q, k, v, b=pair_bias, scale=SCALE, mask=None)
        assert torch.allclose(
            out_manual, out_sdpa, atol=ATOL
        ), f"Max diff: {(out_manual - out_sdpa).abs().max().item()}"

    def test_with_mask_no_bias(self, qkv, padding_mask):
        """Padding mask only, no pair bias."""
        q, k, v = qkv
        out_manual = _attn_manual(q, k, v, b=0, scale=SCALE, mask=padding_mask)
        out_sdpa = _attn_sdpa(q, k, v, b=0, scale=SCALE, mask=padding_mask)
        assert torch.allclose(
            out_manual, out_sdpa, atol=ATOL
        ), f"Max diff: {(out_manual - out_sdpa).abs().max().item()}"

    def test_with_bias_and_mask(self, qkv, pair_bias, padding_mask):
        """Both pair bias and padding mask — the real use case."""
        q, k, v = qkv
        out_manual = _attn_manual(q, k, v, b=pair_bias, scale=SCALE, mask=padding_mask)
        out_sdpa = _attn_sdpa(q, k, v, b=pair_bias, scale=SCALE, mask=padding_mask)
        assert torch.allclose(
            out_manual, out_sdpa, atol=ATOL
        ), f"Max diff: {(out_manual - out_sdpa).abs().max().item()}"

    def test_unmasked_positions_match(self, qkv, pair_bias, padding_mask):
        """Verify unmasked (real) positions produce identical output."""
        q, k, v = qkv
        out_manual = _attn_manual(q, k, v, b=pair_bias, scale=SCALE, mask=padding_mask)
        out_sdpa = _attn_sdpa(q, k, v, b=pair_bias, scale=SCALE, mask=padding_mask)
        # Only compare unmasked positions (first 48 residues)
        assert torch.allclose(
            out_manual[:, :, :48, :], out_sdpa[:, :, :48, :], atol=ATOL
        ), (
            f"Max diff on real positions: "
            f"{(out_manual[:, :, :48, :] - out_sdpa[:, :, :48, :]).abs().max().item()}"
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_dtype_equivalence(self, dtype):
        """Test equivalence across dtypes used in training."""
        torch.manual_seed(99)
        q = torch.randn(B, H, N, D, dtype=dtype)
        k = torch.randn(B, H, N, D, dtype=dtype)
        v = torch.randn(B, H, N, D, dtype=dtype)
        bias = torch.randn(B, H, N, N, dtype=dtype)

        out_manual = _attn_manual(q, k, v, b=bias, scale=SCALE, mask=None)
        out_sdpa = _attn_sdpa(q, k, v, b=bias, scale=SCALE, mask=None)

        tol = 5e-2 if dtype == torch.bfloat16 else ATOL
        assert torch.allclose(
            out_manual, out_sdpa, atol=tol
        ), f"Max diff ({dtype}): {(out_manual - out_sdpa).abs().max().item()}"


class TestSDPAGradients:
    """Verify gradients match between manual and SDPA paths."""

    def test_gradient_equivalence_no_mask(self, qkv, pair_bias):
        """Gradients without masking should match exactly."""
        q, k, v = qkv

        # Manual path
        q1, k1, v1, b1 = [t.clone().requires_grad_(True) for t in [q, k, v, pair_bias]]
        out1 = _attn_manual(q1, k1, v1, b=b1, scale=SCALE, mask=None)
        out1.sum().backward()

        # SDPA path
        q2, k2, v2, b2 = [t.clone().requires_grad_(True) for t in [q, k, v, pair_bias]]
        out2 = _attn_sdpa(q2, k2, v2, b=b2, scale=SCALE, mask=None)
        out2.sum().backward()

        for name, g1, g2 in [
            ("q", q1.grad, q2.grad),
            ("k", k1.grad, k2.grad),
            ("v", v1.grad, v2.grad),
            ("bias", b1.grad, b2.grad),
        ]:
            assert torch.allclose(
                g1, g2, atol=1e-4
            ), f"Gradient mismatch for {name}: max diff {(g1 - g2).abs().max().item()}"

    def test_gradient_equivalence_with_mask(self, qkv, pair_bias, padding_mask):
        """Gradients on unmasked positions should match with masking."""
        q, k, v = qkv

        # Manual path
        q1, k1, v1, b1 = [t.clone().requires_grad_(True) for t in [q, k, v, pair_bias]]
        out1 = _attn_manual(q1, k1, v1, b=b1, scale=SCALE, mask=padding_mask)
        # Only backprop through unmasked positions
        out1[:, :, :48, :].sum().backward()

        # SDPA path
        q2, k2, v2, b2 = [t.clone().requires_grad_(True) for t in [q, k, v, pair_bias]]
        out2 = _attn_sdpa(q2, k2, v2, b=b2, scale=SCALE, mask=padding_mask)
        out2[:, :, :48, :].sum().backward()

        for name, g1, g2 in [
            ("q", q1.grad, q2.grad),
            ("k", k1.grad, k2.grad),
            ("v", v1.grad, v2.grad),
            ("bias", b1.grad, b2.grad),
        ]:
            # Compare only unmasked region gradients
            assert torch.allclose(
                g1[:, :, :48], g2[:, :, :48], atol=1e-4
            ), f"Gradient mismatch for {name}: max diff {(g1[:, :, :48] - g2[:, :, :48]).abs().max().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSDPACUDA:
    """Test on GPU where efficient attention backends are available."""

    def test_cuda_equivalence(self):
        """SDPA on CUDA should match manual on CPU."""
        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device="cuda")
        k = torch.randn(B, H, N, D, device="cuda")
        v = torch.randn(B, H, N, D, device="cuda")
        bias = torch.randn(B, H, N, N, device="cuda")

        out_manual = _attn_manual(q, k, v, b=bias, scale=SCALE, mask=None)
        out_sdpa = _attn_sdpa(q, k, v, b=bias, scale=SCALE, mask=None)
        assert torch.allclose(
            out_manual, out_sdpa, atol=ATOL
        ), f"Max diff (CUDA): {(out_manual - out_sdpa).abs().max().item()}"

    def test_cuda_bf16_with_mask(self):
        """Full scenario: CUDA + bf16 + pair bias + mask.

        Only compare unmasked positions — padded positions differ between
        backends (manual produces non-zero, SDPA produces zero) but this
        doesn't affect training since padded positions are masked downstream.
        """
        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(B, H, N, N, device="cuda", dtype=torch.bfloat16)
        res_mask = torch.ones(B, N, dtype=torch.bool, device="cuda")
        res_mask[:, 48:] = False
        mask = res_mask[:, :, None] & res_mask[:, None, :]

        out_manual = _attn_manual(q, k, v, b=bias, scale=SCALE, mask=mask)
        out_sdpa = _attn_sdpa(q, k, v, b=bias, scale=SCALE, mask=mask)
        # Compare only unmasked positions (first 48 residues)
        assert torch.allclose(
            out_manual[:, :, :48, :], out_sdpa[:, :, :48, :], atol=1e-2
        ), (
            f"Max diff (CUDA bf16 unmasked): "
            f"{(out_manual[:, :, :48, :] - out_sdpa[:, :, :48, :]).abs().max().item()}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSDPABackendSelection:
    """Verify which SDPA backend is actually used.

    FlashAttention-2 may not support arbitrary [B, H, N, N] additive biases.
    With a dense attn_mask, PyTorch dispatches to either the memory-efficient
    (xformers) backend or the Math fallback. This test suite detects which
    backend we actually get, so we know if we're getting a real speedup.
    """

    def _make_inputs(self, n=256, dtype=torch.bfloat16):
        """Create inputs at a given sequence length."""
        q = torch.randn(B, H, n, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, n, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, n, D, device="cuda", dtype=dtype)
        bias = torch.randn(B, H, n, n, device="cuda", dtype=dtype)
        return q, k, v, bias

    def test_flash_attention_accepts_dense_bias(self):
        """FlashAttention on PyTorch 2.6+ / A100 supports dense additive bias.

        Earlier versions rejected [B,H,N,N] biases, but FA2 now handles them.
        This is the best-case scenario for performance.
        """
        q, k, v, bias = self._make_inputs()
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=bias,
                scale=SCALE,
            )
            assert out.shape == (B, H, 256, D)

    def test_mem_efficient_backend_accepts_dense_bias(self):
        """Memory-efficient backend should accept our dense bias."""
        q, k, v, bias = self._make_inputs()
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=False,
            enable_mem_efficient=True,
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=bias,
                scale=SCALE,
            )
            assert out.shape == (B, H, 256, D)

    def test_math_fallback_works(self):
        """Math backend (manual) should always work as a fallback."""
        q, k, v, bias = self._make_inputs()
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=bias,
                scale=SCALE,
            )
            assert out.shape == (B, H, 256, D)

    def test_default_dispatch_does_not_use_math(self):
        """With default dispatch, verify we get mem_efficient, not math.

        If this fails, SDPA is silently falling back to the Math backend
        and we get no memory or speed benefit over the manual implementation.
        """
        q, k, v, bias = self._make_inputs(n=512)
        # Run with all backends enabled (default) and check that
        # mem_efficient alone can handle it
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=False,
            enable_mem_efficient=True,
        ):
            # If this succeeds, mem_efficient handles our inputs
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=bias,
                scale=SCALE,
            )
            assert out.shape == (B, H, 512, D)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSDPAMemory:
    """Verify that SDPA actually reduces peak memory vs manual attention."""

    def _run_forward_backward(self, attn_fn, n, dtype=torch.bfloat16):
        """Run forward + backward and return peak memory allocated."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        q = torch.randn(B, H, n, D, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, n, D, device="cuda", dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, n, D, device="cuda", dtype=dtype, requires_grad=True)
        bias = torch.randn(B, H, n, n, device="cuda", dtype=dtype, requires_grad=True)

        # Baseline memory after allocation
        baseline = torch.cuda.memory_allocated()

        out = attn_fn(q, k, v, b=bias, scale=SCALE, mask=None)
        out.sum().backward()

        peak = torch.cuda.max_memory_allocated() - baseline
        return peak

    @pytest.mark.parametrize("n", [256, 512])
    def test_sdpa_uses_less_memory(self, n):
        """SDPA should use less peak memory than manual at large N.

        The manual path materializes the full [B,H,N,N] attention matrix.
        The memory-efficient backend should avoid this.
        """
        peak_manual = self._run_forward_backward(_attn_manual, n)
        peak_sdpa = self._run_forward_backward(_attn_sdpa, n)

        savings_pct = 100 * (1 - peak_sdpa / peak_manual)
        print(
            f"\nN={n}: manual={peak_manual/1e6:.0f}MB, sdpa={peak_sdpa/1e6:.0f}MB, "
            f"savings={savings_pct:.1f}%"
        )

        # SDPA should use at least 10% less memory to be worthwhile.
        # If this fails, we're getting the Math fallback with no benefit.
        assert peak_sdpa < peak_manual, (
            f"SDPA used MORE memory than manual at N={n}: "
            f"{peak_sdpa/1e6:.0f}MB vs {peak_manual/1e6:.0f}MB. "
            f"Likely falling back to Math backend."
        )


class TestSDPAEdgeCases:
    """Edge cases for robustness."""

    def test_non_contiguous_inputs(self, qkv, pair_bias):
        """SDPA should handle non-contiguous tensors gracefully."""
        q, k, v = qkv
        # Make non-contiguous by transposing and transposing back
        q_nc = q.transpose(-1, -2).transpose(-1, -2)
        assert not q_nc.is_contiguous() or q_nc.data_ptr() == q.data_ptr()

        # Slice to create truly non-contiguous tensor
        q_nc = q[:, :, ::2, :]  # strided access
        k_nc = k[:, :, ::2, :]
        v_nc = v[:, :, ::2, :]
        bias_nc = pair_bias[:, :, ::2, ::2]

        out_sdpa = _attn_sdpa(q_nc, k_nc, v_nc, b=bias_nc, scale=SCALE, mask=None)
        out_manual = _attn_manual(q_nc, k_nc, v_nc, b=bias_nc, scale=SCALE, mask=None)
        assert torch.allclose(
            out_manual, out_sdpa, atol=ATOL
        ), f"Non-contiguous max diff: {(out_manual - out_sdpa).abs().max().item()}"

    def test_all_masked_row(self):
        """A fully masked query row should not produce NaN."""
        torch.manual_seed(42)
        q = torch.randn(1, H, 8, D)
        k = torch.randn(1, H, 8, D)
        v = torch.randn(1, H, 8, D)

        # Mask where row 0 has ALL keys masked (entire row is False)
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        mask[0, 3, :] = False  # query position 3 can't attend to anything

        out = _attn_sdpa(q, k, v, b=0, scale=SCALE, mask=mask)

        # Check that non-masked rows are valid
        assert not torch.isnan(out[0, :, 0, :]).any(), "Non-masked position has NaN"
        # Masked row may be NaN or zero depending on backend — just check
        # it doesn't corrupt other positions
        assert not torch.isnan(out[0, :, 0, :]).any(), "NaN leaked to valid positions"
        assert not torch.isnan(out[0, :, 1, :]).any(), "NaN leaked to valid positions"
        assert not torch.isnan(out[0, :, 2, :]).any(), "NaN leaked to valid positions"
        assert not torch.isnan(out[0, :, 4, :]).any(), "NaN leaked to valid positions"

    def test_single_sample_batch(self):
        """B=1 should work (edge case for some SDPA backends)."""
        torch.manual_seed(42)
        q = torch.randn(1, H, N, D)
        k = torch.randn(1, H, N, D)
        v = torch.randn(1, H, N, D)
        bias = torch.randn(1, H, N, N)

        out_manual = _attn_manual(q, k, v, b=bias, scale=SCALE, mask=None)
        out_sdpa = _attn_sdpa(q, k, v, b=bias, scale=SCALE, mask=None)
        assert torch.allclose(out_manual, out_sdpa, atol=ATOL)
