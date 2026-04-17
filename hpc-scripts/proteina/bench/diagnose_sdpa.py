"""Diagnose why SDPA's FLASH / EFFICIENT backends reject proteina's bias.

Standalone — reproduces the attention call with realistic shapes
(B=6, H=8, N=512, D=64, bf16) and additive float bias, tries each backend,
prints the raw exception. Then tests whether .contiguous() / .detach()
unlocks EFFICIENT.

Usage (GPU required):
    python hpc-scripts/proteina/bench/diagnose_sdpa.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


B, H, N, D = 6, 8, 512, 64
DEVICE = "cuda"
DTYPE = torch.bfloat16


def make_inputs(
    bias_contiguous: bool,
    bias_requires_grad: bool,
    bias_dtype: torch.dtype = DTYPE,
    via_rearrange: bool = False,
):
    """Build q, k, v, bias matching proteina's attention pattern."""
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)

    if via_rearrange:
        # Match proteina's code: pair_feats (B, N, N, H) -> rearrange to (B, H, N, N).
        # That rearrange produces a strided view; .contiguous() materializes it.
        bias_src = torch.randn(
            B,
            N,
            N,
            H,
            device=DEVICE,
            dtype=bias_dtype,
            requires_grad=bias_requires_grad,
        )
        bias = bias_src.permute(0, 3, 1, 2)  # (B, H, N, N), strided view
    else:
        bias = torch.randn(
            B,
            H,
            N,
            N,
            device=DEVICE,
            dtype=bias_dtype,
            requires_grad=bias_requires_grad,
        )

    if bias_contiguous:
        bias = bias.contiguous()
    return q, k, v, bias


def time_ms(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def try_backend(label: str, backend, q, k, v, bias, do_backward: bool):
    try:

        def run():
            with sdpa_kernel([backend]):
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=bias,
                    scale=1.0 / (D**0.5),
                )
            if do_backward:
                out.sum().backward(retain_graph=True)

        # Actually call once to surface the error (time_ms swallows it in warmup)
        run()
        t = time_ms(run)
        phase = "fwd+bwd" if do_backward else "fwd"
        print(f"  {label:35s}  {phase:7s}  OK     {t:6.2f} ms/iter")
        return True
    except Exception as e:
        phase = "fwd+bwd" if do_backward else "fwd"
        first_line = str(e).splitlines()[0] if str(e) else type(e).__name__
        print(f"  {label:35s}  {phase:7s}  FAIL   {first_line[:120]}")
        return False


def scenario(name, **kwargs):
    do_backward = kwargs.pop("do_backward", True)
    print(f"\n─── {name} ───")
    q, k, v, bias = make_inputs(**kwargs)
    print(
        f"  bias: shape={tuple(bias.shape)} dtype={bias.dtype} "
        f"contiguous={bias.is_contiguous()} requires_grad={bias.requires_grad}"
    )
    for label, backend in [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("MATH", SDPBackend.MATH),
        ("CUDNN_ATTENTION", getattr(SDPBackend, "CUDNN_ATTENTION", None)),
    ]:
        if backend is None:
            continue
        # Fresh tensors per backend so backward doesn't accumulate weirdly
        q2, k2, v2, bias2 = make_inputs(**kwargs)
        try_backend(label, backend, q2, k2, v2, bias2, do_backward=do_backward)


def main():
    print(f"torch: {torch.__version__}")
    print(f"cuda:  {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"shapes: B={B} H={H} N={N} D={D} dtype={DTYPE}")
    print(
        f"flash_enabled={torch.backends.cuda.flash_sdp_enabled()} "
        f"mem_eff_enabled={torch.backends.cuda.mem_efficient_sdp_enabled()} "
        f"math_enabled={torch.backends.cuda.math_sdp_enabled()}"
    )

    # Scenario A: proteina's real pattern (strided view from rearrange, grad, bf16)
    scenario(
        "A: proteina-like (strided bias via rearrange, requires_grad)",
        bias_contiguous=False,
        bias_requires_grad=True,
        via_rearrange=True,
        do_backward=True,
    )

    # B: contiguous bias, grad retained
    scenario(
        "B: contiguous bias, requires_grad",
        bias_contiguous=True,
        bias_requires_grad=True,
        do_backward=True,
    )

    # C: detached bias (no grad), contiguous
    scenario(
        "C: contiguous bias, detached (no grad)",
        bias_contiguous=True,
        bias_requires_grad=False,
        do_backward=True,
    )

    # D: forward-only (skip backward)
    scenario(
        "D: strided bias, requires_grad, FORWARD ONLY",
        bias_contiguous=False,
        bias_requires_grad=True,
        via_rearrange=True,
        do_backward=False,
    )

    # E: no bias at all — baseline for what kernels *can* handle
    print("\n─── E: baseline — no bias ───")
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    for label, backend in [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("MATH", SDPBackend.MATH),
        ("CUDNN_ATTENTION", getattr(SDPBackend, "CUDNN_ATTENTION", None)),
    ]:
        if backend is None:
            continue
        try:
            with sdpa_kernel([backend]):
                out = F.scaled_dot_product_attention(q, k, v, scale=1.0 / (D**0.5))
            out.sum().backward(retain_graph=True)

            def run():
                with sdpa_kernel([backend]):
                    out = F.scaled_dot_product_attention(q, k, v, scale=1.0 / (D**0.5))
                out.sum().backward(retain_graph=True)

            t = time_ms(run)
            print(f"  {label:35s}  fwd+bwd  OK     {t:6.2f} ms/iter")
        except Exception as e:
            first = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"  {label:35s}  fwd+bwd  FAIL   {first[:120]}")


if __name__ == "__main__":
    main()
