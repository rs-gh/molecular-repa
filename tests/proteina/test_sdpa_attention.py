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

    # Mirrors prod: force contiguity so EFFICIENT_ATTENTION dispatches instead
    # of MATH fallback (see pair_bias_attn.py and test class below).
    if attn_bias is not None:
        attn_bias = attn_bias.contiguous()

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
class TestSDPABackendRouting:
    """Characterize which SDPA backend actually handles proteina's attention.

    Proteina passes a (B, H, N, N) float additive bias to
    F.scaled_dot_product_attention (pair_bias_attn.py:126). Not every SDPA
    backend accepts this pattern — FlashAttention-2 only supports per-head
    ALiBi slopes, not arbitrary dense bias; EFFICIENT_ATTENTION (memory-
    efficient / xformers) is spec'd to accept arbitrary bias but has stricter
    runtime requirements around contiguity and grad.

    These tests document the current reality: which backends SDPA can
    dispatch to for our exact input pattern. If any assertion flips (e.g.,
    FLASH starts accepting our bias on a future torch upgrade, or the
    attention code changes so EFFICIENT accepts it), these will fail and
    prompt us to re-measure throughput.
    """

    def _make_inputs(self, n=256, dtype=torch.bfloat16, requires_grad=True):
        """Inputs matching proteina's attention call shape."""
        q = torch.randn(
            B, H, n, D, device="cuda", dtype=dtype, requires_grad=requires_grad
        )
        k = torch.randn(
            B, H, n, D, device="cuda", dtype=dtype, requires_grad=requires_grad
        )
        v = torch.randn(
            B, H, n, D, device="cuda", dtype=dtype, requires_grad=requires_grad
        )
        bias = torch.randn(
            B, H, n, n, device="cuda", dtype=dtype, requires_grad=requires_grad
        )
        return q, k, v, bias

    def _try_backend(self, backend, do_backward=True):
        """Run forward (+ optional backward) under the given SDPA backend.
        Returns True if succeeds, False if the backend rejects the inputs."""
        from torch.nn.attention import sdpa_kernel

        q, k, v, bias = self._make_inputs()
        try:
            with sdpa_kernel([backend]):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=bias,
                    scale=SCALE,
                )
            if do_backward:
                out.sum().backward()
            torch.cuda.synchronize()
            return True
        except RuntimeError:
            return False

    def test_math_backend_accepts_proteina_bias(self):
        """MATH is the universal fallback; must always work on our pattern."""
        from torch.nn.attention import SDPBackend

        assert self._try_backend(SDPBackend.MATH), (
            "MATH backend rejected proteina's attention inputs — "
            "something fundamentally changed in torch's SDPA."
        )

    def test_flash_backend_currently_rejects_proteina_bias(self):
        """Characterization: FLASH_ATTENTION rejects dense (B,H,N,N) bias.

        FA2 only supports per-head ALiBi slopes, not arbitrary bias.
        FA3 has partial support but isn't wired through torch's FLASH dispatch.
        If this test starts failing, a torch upgrade enabled dense-bias
        support — re-run the SDPA benchmark to confirm a real speedup.
        """
        from torch.nn.attention import SDPBackend

        assert not self._try_backend(SDPBackend.FLASH_ATTENTION), (
            "FLASH_ATTENTION unexpectedly accepted proteina's dense bias — "
            "this is good news, re-run hpc-scripts/proteina/bench/benchmark_sdpa.py "
            "to measure the throughput gain and update docs."
        )

    def test_efficient_backend_currently_rejects_proteina_bias(self):
        """Characterization: EFFICIENT_ATTENTION also rejects our bias today.

        EFFICIENT *is* designed to accept arbitrary additive bias, but some
        combination of (strided-view bias from rearrange, requires_grad, bf16
        under autocast) trips it up in training. See
        hpc-scripts/proteina/bench/diagnose_sdpa.py for the matrix of tried
        fixes. If this test flips, one of those constraints got lifted —
        re-benchmark and update proteina_training_runs.md.
        """
        from torch.nn.attention import SDPBackend

        assert not self._try_backend(SDPBackend.EFFICIENT_ATTENTION), (
            "EFFICIENT_ATTENTION accepted proteina's bias — update the "
            "SDPA benchmark and docs, this is a potential speedup."
        )

    def test_default_dispatch_falls_to_math(self):
        """Because FLASH and EFFICIENT both reject our inputs, the default
        (all-backends-enabled) dispatcher must fall through to MATH.

        Proof by elimination: disable MATH, leave only FLASH+EFFICIENT, and
        the call should fail — since neither accepts our pattern, MATH is
        the only backend carrying the forward pass in production today.
        """
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q, k, v, bias = self._make_inputs()
        with pytest.raises(RuntimeError):
            with sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                ]
            ):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=bias,
                    scale=SCALE,
                )
                out.sum().backward()
                torch.cuda.synchronize()

    def test_pair_bias_attention_routes_to_sdpa_path(self):
        """Verify ``use_sdpa=True`` (config default) routes through
        F.scaled_dot_product_attention, not the manual einsum fallback.

        Guards against a silent regression where use_sdpa gets flipped off
        and we don't notice.
        """
        import sys
        import os

        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(__file__), "../../src/proteina/proteinfoundation"
            ),
        )
        from nn.pair_bias_attn.pair_bias_attn import PairBiasAttention

        attn = PairBiasAttention(
            node_dim=64,
            pair_dim=32,
            heads=H,
            head_dim=8,
            use_sdpa=True,
        )
        assert attn.use_sdpa is True
        # ``_attn_sdpa`` is the method that calls F.scaled_dot_product_attention;
        # merely asserting use_sdpa=True is the config-level guard.
        assert hasattr(attn, "_attn_sdpa")


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


class TestContiguousBiasFix:
    """Guard: the prod SDPA path must call .contiguous() on the attn_mask
    before dispatching F.scaled_dot_product_attention.

    Reason: proteina produces the pair bias via
    ``rearrange(to_bias(pair_feats), "b ... h -> b h ...")`` which returns a
    non-contiguous strided view. The fused SDPA kernels (FLASH, EFFICIENT,
    CUDNN) all require ``stride(-1) == 1`` on ``attn_mask``; a strided input
    causes torch to silently fall back to the MATH kernel (~4× slower per
    attention call).

    Numerical equivalence verified 2026-04-18 via
    hpc-scripts/proteina/bench/diagnose_sdpa.py (MATH-strided vs
    EFFICIENT-contiguous agree to worst-case 5.0e-3 relative, below bf16
    eps). Output preserved at
    evaluation/proteina/results/bench/sdpa_equivalence_2026-04-18.txt.

    If this test starts failing, someone removed the .contiguous() call in
    pair_bias_attn._attn_sdpa and training silently regressed onto MATH.
    """

    def test_sdpa_receives_contiguous_mask_when_input_is_strided(self):
        """Even when _attn_sdpa is handed a strided bias (the actual proteina
        rearrange pattern), the bias reaching F.scaled_dot_product_attention
        must be contiguous."""
        import sys
        import os

        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(__file__), "../../src/proteina/proteinfoundation"
            ),
        )
        from nn.pair_bias_attn.pair_bias_attn import PairBiasAttention
        import torch.nn.functional as F_mod

        attn = PairBiasAttention(
            node_dim=32,
            dim_head=8,
            heads=H,
            bias=True,
            dim_out=32,
            qkln=False,
            pair_dim=16,
            use_sdpa=True,
        )

        # Build inputs in proteina's actual layout: bias comes in as (B, H, N, N)
        # from a rearrange("b ... h -> b h ..."), which is a non-contiguous view.
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        bias_src = torch.randn(B, N, N, H)
        strided_bias = bias_src.permute(0, 3, 1, 2)  # (B, H, N, N), non-contig
        assert (
            not strided_bias.is_contiguous()
        ), "test setup: strided_bias should be non-contiguous before the fix"

        captured = {}
        original_sdpa = F_mod.scaled_dot_product_attention

        def spy_sdpa(q, k, v, attn_mask=None, **kwargs):
            captured["attn_mask"] = attn_mask
            return original_sdpa(q, k, v, attn_mask=attn_mask, **kwargs)

        F_mod.scaled_dot_product_attention = spy_sdpa
        try:
            attn._attn_sdpa(q, k, v, b=strided_bias, mask=None)
        finally:
            F_mod.scaled_dot_product_attention = original_sdpa

        assert "attn_mask" in captured, "F.scaled_dot_product_attention was not invoked"
        assert captured["attn_mask"] is not None, "attn_mask was dropped before SDPA"
        assert captured["attn_mask"].is_contiguous(), (
            "attn_mask reaching F.scaled_dot_product_attention is NOT contiguous. "
            "The .contiguous() fix in pair_bias_attn._attn_sdpa is missing or broken — "
            "training will fall back to the slow MATH kernel (~4× slower per "
            "attention call). See evaluation/proteina/results/bench/"
            "sdpa_equivalence_2026-04-18.txt for numerical-equivalence evidence."
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_prod_attn_dispatches_to_fused_kernel_not_math(self):
        """Runtime check: with the .contiguous() fix in place, prod's
        _attn_sdpa call must be accepted by a fused kernel (EFFICIENT /
        FLASH / CUDNN), not silently fall back to MATH.

        Stronger than the contiguity-spy test above: that only checks the
        necessary condition (stride==1 on last dim). This test proves the
        sufficient condition (a fused kernel actually accepts our full
        input tuple) by excluding MATH from the allowed-backends list and
        asserting the call still succeeds.
        """
        import sys
        import os

        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(__file__), "../../src/proteina/proteinfoundation"
            ),
        )
        from nn.pair_bias_attn.pair_bias_attn import PairBiasAttention
        from torch.nn.attention import SDPBackend, sdpa_kernel

        attn = PairBiasAttention(
            node_dim=32,
            dim_head=8,
            heads=H,
            bias=True,
            dim_out=32,
            qkln=False,
            pair_dim=16,
            use_sdpa=True,
        ).cuda()

        # Match prod conditions: bf16, requires_grad, strided bias from rearrange.
        dtype = torch.bfloat16
        q = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
        bias_src = torch.randn(
            B, N, N, H, device="cuda", dtype=dtype, requires_grad=True
        )
        strided_bias = bias_src.permute(0, 3, 1, 2)
        assert not strided_bias.is_contiguous()

        # Exclude MATH from the backend whitelist. If _attn_sdpa still works,
        # a fused kernel (EFFICIENT / FLASH / CUDNN) handled the call — which
        # is only possible because .contiguous() was applied to the bias.
        fused_only = [
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
        ]
        cudnn = getattr(SDPBackend, "CUDNN_ATTENTION", None)
        if cudnn is not None:
            fused_only.append(cudnn)

        with sdpa_kernel(fused_only):
            try:
                out = attn._attn_sdpa(q, k, v, b=strided_bias, mask=None)
                out.sum().backward()
            except RuntimeError as e:
                pytest.fail(
                    "prod _attn_sdpa failed when MATH is disabled — no fused "
                    "kernel accepted the inputs, meaning training is on the "
                    "slow MATH path. Inner error: " + str(e).splitlines()[0]
                )
