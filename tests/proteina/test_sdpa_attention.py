"""Tests for PairBiasAttention's SDPA vs manual attention paths.

All tests import and exercise the live module rather than duplicating
the implementation, so they can't silently drift from production code.
"""

import os
import sys

import pytest
import torch

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "../../src/proteina/proteinfoundation"),
)
from nn.pair_bias_attn.pair_bias_attn import PairBiasAttention

# ---------------------------------------------------------------------------
# Shared constants and fixtures
# ---------------------------------------------------------------------------

B, H, N, D = 2, 8, 64, 64  # batch, heads, seq_len, head_dim
SCALE = D**-0.5
ATOL = 1e-5


def _make_attn(**kwargs) -> PairBiasAttention:
    """Return a PairBiasAttention with minimal config for unit testing."""
    defaults = dict(
        node_dim=32,
        dim_head=D // H,
        heads=H,
        bias=True,
        dim_out=32,
        qkln=False,
        pair_dim=16,
        use_sdpa=True,
    )
    defaults.update(kwargs)
    return PairBiasAttention(**defaults)


@pytest.fixture
def attn():
    return _make_attn()


@pytest.fixture
def qkv():
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    return q, k, v


@pytest.fixture
def pair_bias():
    torch.manual_seed(123)
    return torch.randn(B, H, N, N)


@pytest.fixture
def padding_mask():
    torch.manual_seed(456)
    res_mask = torch.ones(B, N, dtype=torch.bool)
    res_mask[:, 48:] = False
    return res_mask[:, :, None] & res_mask[:, None, :]


# ---------------------------------------------------------------------------
# Equivalence: SDPA vs manual path
# ---------------------------------------------------------------------------


class TestSDPAEquivalence:
    """_attn_sdpa and _attn must agree numerically on all input combinations."""

    def test_no_bias_no_mask(self, attn, qkv):
        q, k, v = qkv
        out_manual = attn._attn(q, k, v, b=0, mask=None)
        out_sdpa = attn._attn_sdpa(q, k, v, b=0, mask=None)
        assert torch.allclose(
            out_manual, out_sdpa, atol=ATOL
        ), f"Max diff: {(out_manual - out_sdpa).abs().max().item()}"

    def test_with_pair_bias_no_mask(self, attn, qkv, pair_bias):
        q, k, v = qkv
        out_manual = attn._attn(q, k, v, b=pair_bias, mask=None)
        out_sdpa = attn._attn_sdpa(q, k, v, b=pair_bias, mask=None)
        assert torch.allclose(
            out_manual, out_sdpa, atol=ATOL
        ), f"Max diff: {(out_manual - out_sdpa).abs().max().item()}"

    def test_with_mask_no_bias(self, attn, qkv, padding_mask):
        q, k, v = qkv
        out_manual = attn._attn(q, k, v, b=0, mask=padding_mask)
        out_sdpa = attn._attn_sdpa(q, k, v, b=0, mask=padding_mask)
        assert torch.allclose(
            out_manual, out_sdpa, atol=ATOL
        ), f"Max diff: {(out_manual - out_sdpa).abs().max().item()}"

    def test_with_bias_and_mask(self, attn, qkv, pair_bias, padding_mask):
        q, k, v = qkv
        out_manual = attn._attn(q, k, v, b=pair_bias, mask=padding_mask)
        out_sdpa = attn._attn_sdpa(q, k, v, b=pair_bias, mask=padding_mask)
        assert torch.allclose(
            out_manual, out_sdpa, atol=ATOL
        ), f"Max diff: {(out_manual - out_sdpa).abs().max().item()}"

    def test_unmasked_positions_match(self, attn, qkv, pair_bias, padding_mask):
        q, k, v = qkv
        out_manual = attn._attn(q, k, v, b=pair_bias, mask=padding_mask)
        out_sdpa = attn._attn_sdpa(q, k, v, b=pair_bias, mask=padding_mask)
        assert torch.allclose(
            out_manual[:, :, :48, :], out_sdpa[:, :, :48, :], atol=ATOL
        ), (
            f"Max diff on real positions: "
            f"{(out_manual[:, :, :48, :] - out_sdpa[:, :, :48, :]).abs().max().item()}"
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_dtype_equivalence(self, attn, dtype):
        torch.manual_seed(99)
        q = torch.randn(B, H, N, D, dtype=dtype)
        k = torch.randn(B, H, N, D, dtype=dtype)
        v = torch.randn(B, H, N, D, dtype=dtype)
        bias = torch.randn(B, H, N, N, dtype=dtype)
        attn_typed = attn.to(dtype)
        out_manual = attn_typed._attn(q, k, v, b=bias, mask=None)
        out_sdpa = attn_typed._attn_sdpa(q, k, v, b=bias, mask=None)
        # bf16 on CPU MATH backend accumulates more rounding error than GPU EFFICIENT.
        # 0.12 = ~15 bf16 LSBs at scale 1.0, consistent with a 64-token softmax sum.
        tol = 0.12 if dtype == torch.bfloat16 else ATOL
        assert torch.allclose(
            out_manual, out_sdpa, atol=tol
        ), f"Max diff ({dtype}): {(out_manual - out_sdpa).abs().max().item()}"

    def test_single_sample_batch(self, attn):
        torch.manual_seed(42)
        q = torch.randn(1, H, N, D)
        k = torch.randn(1, H, N, D)
        v = torch.randn(1, H, N, D)
        bias = torch.randn(1, H, N, N)
        out_manual = attn._attn(q, k, v, b=bias, mask=None)
        out_sdpa = attn._attn_sdpa(q, k, v, b=bias, mask=None)
        assert torch.allclose(out_manual, out_sdpa, atol=ATOL)


# ---------------------------------------------------------------------------
# Gradient equivalence
# ---------------------------------------------------------------------------


class TestSDPAGradients:
    def test_gradient_equivalence_no_mask(self, attn, qkv, pair_bias):
        q, k, v = qkv
        q1, k1, v1, b1 = [t.clone().requires_grad_(True) for t in [q, k, v, pair_bias]]
        attn._attn(q1, k1, v1, b=b1, mask=None).sum().backward()

        q2, k2, v2, b2 = [t.clone().requires_grad_(True) for t in [q, k, v, pair_bias]]
        attn._attn_sdpa(q2, k2, v2, b=b2, mask=None).sum().backward()

        for name, g1, g2 in [
            ("q", q1.grad, q2.grad),
            ("k", k1.grad, k2.grad),
            ("v", v1.grad, v2.grad),
            ("bias", b1.grad, b2.grad),
        ]:
            assert torch.allclose(
                g1, g2, atol=1e-4
            ), f"Gradient mismatch for {name}: max diff {(g1 - g2).abs().max().item()}"

    def test_gradient_equivalence_with_mask(self, attn, qkv, pair_bias, padding_mask):
        q, k, v = qkv
        q1, k1, v1, b1 = [t.clone().requires_grad_(True) for t in [q, k, v, pair_bias]]
        attn._attn(q1, k1, v1, b=b1, mask=padding_mask)[:, :, :48, :].sum().backward()

        q2, k2, v2, b2 = [t.clone().requires_grad_(True) for t in [q, k, v, pair_bias]]
        attn._attn_sdpa(q2, k2, v2, b=b2, mask=padding_mask)[
            :, :, :48, :
        ].sum().backward()

        for name, g1, g2 in [
            ("q", q1.grad, q2.grad),
            ("k", k1.grad, k2.grad),
            ("v", v1.grad, v2.grad),
            ("bias", b1.grad, b2.grad),
        ]:
            assert torch.allclose(g1[:, :, :48], g2[:, :, :48], atol=1e-4), (
                f"Gradient mismatch for {name}: max diff "
                f"{(g1[:, :, :48] - g2[:, :, :48]).abs().max().item()}"
            )


# ---------------------------------------------------------------------------
# use_sdpa default and forward routing
# ---------------------------------------------------------------------------


class TestPairBiasAttentionRouting:
    """Verify the module uses the SDPA path by default and routes correctly."""

    def test_default_is_sdpa(self):
        attn = _make_attn()
        assert attn.use_sdpa is True

    def test_use_sdpa_false_uses_manual(self):
        attn = _make_attn(use_sdpa=False)
        assert attn.use_sdpa is False

    def test_forward_sdpa_and_manual_agree(self):
        """Full forward pass through PairBiasAttention should agree regardless of use_sdpa."""
        torch.manual_seed(7)
        node = torch.randn(B, N, 32)
        pair = torch.randn(B, N, N, 16)
        res_mask = torch.ones(B, N, dtype=torch.bool)
        res_mask[:, 48:] = False
        mask = res_mask[:, :, None] & res_mask[:, None, :]

        attn_sdpa = _make_attn(use_sdpa=True)
        attn_manual = _make_attn(use_sdpa=False)
        # Share weights so we compare paths, not random inits
        attn_manual.load_state_dict(attn_sdpa.state_dict())

        with torch.no_grad():
            out_sdpa = attn_sdpa(node, pair, mask)
            out_manual = attn_manual(node, pair, mask)

        assert torch.allclose(
            out_sdpa, out_manual, atol=1e-4
        ), f"Forward pass mismatch: max diff {(out_sdpa - out_manual).abs().max().item()}"

    def test_sdpa_path_calls_f_sdpa(self):
        """Spy: confirm _attn_sdpa actually calls F.scaled_dot_product_attention."""
        import torch.nn.functional as F_mod

        attn = _make_attn()
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        bias = torch.randn(B, H, N, N)
        captured = {}
        original = F_mod.scaled_dot_product_attention

        def spy(q, k, v, attn_mask=None, **kwargs):
            captured["called"] = True
            captured["attn_mask"] = attn_mask
            return original(q, k, v, attn_mask=attn_mask, **kwargs)

        F_mod.scaled_dot_product_attention = spy
        try:
            attn._attn_sdpa(q, k, v, b=bias, mask=None)
        finally:
            F_mod.scaled_dot_product_attention = original

        assert captured.get("called"), "F.scaled_dot_product_attention was not invoked"


# ---------------------------------------------------------------------------
# Contiguous bias guard
# ---------------------------------------------------------------------------


class TestContiguousBiasFix:
    """Guard: the .contiguous() call in _attn_sdpa must remain.

    proteina produces the pair bias via rearrange("b ... h -> b h ...") which
    leaves stride(-1) != 1. Fused SDPA kernels require the last stride to be 1;
    without .contiguous() torch silently falls back to the slow MATH kernel.
    Validated 2026-04-18 via hpc-scripts/proteina/bench/diagnose_sdpa.py.
    """

    def test_attn_mask_is_contiguous_when_bias_is_strided(self):
        """Even when _attn_sdpa receives a strided bias (the real rearrange
        pattern), the tensor reaching F.sdpa must be contiguous."""
        import torch.nn.functional as F_mod

        attn = _make_attn()
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        # Simulate proteina's rearrange("b ... h -> b h ..."): non-contiguous view
        bias_src = torch.randn(B, N, N, H)
        strided_bias = bias_src.permute(0, 3, 1, 2)
        assert not strided_bias.is_contiguous(), "test setup error"

        captured = {}
        original = F_mod.scaled_dot_product_attention

        def spy(q, k, v, attn_mask=None, **kwargs):
            captured["attn_mask"] = attn_mask
            return original(q, k, v, attn_mask=attn_mask, **kwargs)

        F_mod.scaled_dot_product_attention = spy
        try:
            attn._attn_sdpa(q, k, v, b=strided_bias, mask=None)
        finally:
            F_mod.scaled_dot_product_attention = original

        assert (
            captured.get("attn_mask") is not None
        ), "attn_mask was dropped before SDPA"
        assert captured["attn_mask"].is_contiguous(), (
            "attn_mask reaching F.sdpa is NOT contiguous. The .contiguous() fix "
            "in _attn_sdpa is missing - training will fall back to the slow MATH kernel."
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_prod_attn_dispatches_to_fused_kernel_not_math(self):
        """With .contiguous() in place, a fused kernel must accept our inputs.

        Excludes MATH from the allowed-backends list. If the call succeeds,
        a fused kernel (EFFICIENT / FLASH / CUDNN) handled it.
        """
        from torch.nn.attention import SDPBackend, sdpa_kernel

        attn = _make_attn().cuda()
        dtype = torch.bfloat16
        q = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
        bias_src = torch.randn(
            B, N, N, H, device="cuda", dtype=dtype, requires_grad=True
        )
        strided_bias = bias_src.permute(0, 3, 1, 2)
        assert not strided_bias.is_contiguous()

        fused_only = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
        cudnn = getattr(SDPBackend, "CUDNN_ATTENTION", None)
        if cudnn is not None:
            fused_only.append(cudnn)

        with sdpa_kernel(fused_only):
            try:
                out = attn._attn_sdpa(q, k, v, b=strided_bias, mask=None)
                out.sum().backward()
            except RuntimeError as e:
                pytest.fail(
                    "prod _attn_sdpa failed when MATH is disabled - no fused "
                    "kernel accepted the inputs. Inner error: " + str(e).splitlines()[0]
                )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSDPAEdgeCases:
    def test_non_contiguous_inputs(self, attn, qkv, pair_bias):
        q, k, v = qkv
        q_nc = q[:, :, ::2, :]
        k_nc = k[:, :, ::2, :]
        v_nc = v[:, :, ::2, :]
        bias_nc = pair_bias[:, :, ::2, ::2]
        out_sdpa = attn._attn_sdpa(q_nc, k_nc, v_nc, b=bias_nc, mask=None)
        out_manual = attn._attn(q_nc, k_nc, v_nc, b=bias_nc, mask=None)
        assert torch.allclose(
            out_manual, out_sdpa, atol=ATOL
        ), f"Non-contiguous max diff: {(out_manual - out_sdpa).abs().max().item()}"

    def test_all_masked_row_no_nan(self, attn):
        torch.manual_seed(42)
        q = torch.randn(1, H, 8, D)
        k = torch.randn(1, H, 8, D)
        v = torch.randn(1, H, 8, D)
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        mask[0, 3, :] = False  # query position 3 attends to nothing
        out = attn._attn_sdpa(q, k, v, b=0, mask=mask)
        # Unmasked rows must not have NaN
        for row in [0, 1, 2, 4, 5, 6, 7]:
            assert not torch.isnan(out[0, :, row, :]).any(), f"NaN at row {row}"


# ---------------------------------------------------------------------------
# CUDA tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSDPACUDA:
    def test_cuda_equivalence(self, attn):
        torch.manual_seed(42)
        attn = attn.cuda()
        q = torch.randn(B, H, N, D, device="cuda")
        k = torch.randn(B, H, N, D, device="cuda")
        v = torch.randn(B, H, N, D, device="cuda")
        bias = torch.randn(B, H, N, N, device="cuda")
        out_manual = attn._attn(q, k, v, b=bias, mask=None)
        out_sdpa = attn._attn_sdpa(q, k, v, b=bias, mask=None)
        assert torch.allclose(
            out_manual, out_sdpa, atol=ATOL
        ), f"Max diff (CUDA): {(out_manual - out_sdpa).abs().max().item()}"

    def test_cuda_bf16_with_mask(self, attn):
        """Full scenario: CUDA + bf16 + pair bias + mask.

        Only compare unmasked positions - padded positions may differ between
        backends but don't affect training since they're masked downstream.
        """
        torch.manual_seed(42)
        attn = attn.cuda().to(torch.bfloat16)
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(B, H, N, N, device="cuda", dtype=torch.bfloat16)
        res_mask = torch.ones(B, N, dtype=torch.bool, device="cuda")
        res_mask[:, 48:] = False
        mask = res_mask[:, :, None] & res_mask[:, None, :]

        out_manual = attn._attn(q, k, v, b=bias, mask=mask)
        out_sdpa = attn._attn_sdpa(q, k, v, b=bias, mask=mask)
        assert torch.allclose(
            out_manual[:, :, :48, :], out_sdpa[:, :, :48, :], atol=1e-2
        ), (
            f"Max diff (CUDA bf16 unmasked): "
            f"{(out_manual[:, :, :48, :] - out_sdpa[:, :, :48, :]).abs().max().item()}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSDPABackendRouting:
    """Characterize which SDPA backend handles proteina's attention pattern.

    These tests document current reality. If any assertion flips on a torch
    upgrade, re-run hpc-scripts/proteina/bench/benchmark_sdpa.py to measure
    the impact.
    """

    def _make_inputs(self, n=256, dtype=torch.bfloat16, requires_grad=True):
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
        from torch.nn.attention import SDPBackend

        assert self._try_backend(
            SDPBackend.MATH
        ), "MATH backend rejected proteina's attention inputs."

    def test_flash_backend_currently_rejects_proteina_bias(self):
        """FA2 only supports per-head ALiBi slopes, not arbitrary dense bias.
        If this flips, re-benchmark - a torch upgrade may have enabled it."""
        from torch.nn.attention import SDPBackend

        assert not self._try_backend(SDPBackend.FLASH_ATTENTION), (
            "FLASH_ATTENTION unexpectedly accepted proteina's dense bias - "
            "re-run hpc-scripts/proteina/bench/benchmark_sdpa.py."
        )

    def test_efficient_backend_currently_rejects_proteina_bias(self):
        """EFFICIENT rejects our bias under (strided, requires_grad, bf16).
        If this flips, re-benchmark."""
        from torch.nn.attention import SDPBackend

        assert not self._try_backend(
            SDPBackend.EFFICIENT_ATTENTION
        ), "EFFICIENT_ATTENTION accepted proteina's bias - re-benchmark."

    def test_default_dispatch_falls_to_math(self):
        """Without MATH, FLASH+EFFICIENT both reject our inputs, so the call fails."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q, k, v, bias = self._make_inputs()
        with pytest.raises(RuntimeError):
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=bias,
                    scale=SCALE,
                ).sum().backward()
                torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSDPAMemory:
    """Verify that SDPA reduces peak memory vs manual attention."""

    def _run_forward_backward(self, attn, use_sdpa, n, dtype=torch.bfloat16):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        q = torch.randn(B, H, n, D, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, n, D, device="cuda", dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, n, D, device="cuda", dtype=dtype, requires_grad=True)
        bias = torch.randn(B, H, n, n, device="cuda", dtype=dtype, requires_grad=True)
        baseline = torch.cuda.memory_allocated()
        fn = attn._attn_sdpa if use_sdpa else attn._attn
        fn(q, k, v, b=bias, mask=None).sum().backward()
        return torch.cuda.max_memory_allocated() - baseline

    @pytest.mark.parametrize("n", [256, 512])
    def test_sdpa_uses_less_memory(self, n):
        attn = _make_attn().cuda().to(torch.bfloat16)
        peak_manual = self._run_forward_backward(attn, use_sdpa=False, n=n)
        peak_sdpa = self._run_forward_backward(attn, use_sdpa=True, n=n)
        print(
            f"\nN={n}: manual={peak_manual/1e6:.0f}MB sdpa={peak_sdpa/1e6:.0f}MB "
            f"savings={100*(1-peak_sdpa/peak_manual):.1f}%"
        )
        assert peak_sdpa < peak_manual, (
            f"SDPA used MORE memory than manual at N={n}: "
            f"{peak_sdpa/1e6:.0f}MB vs {peak_manual/1e6:.0f}MB - likely MATH fallback."
        )
