"""Diagnose which SDPA backend tabasco's attention actually uses.

Tabasco's `Attention` wraps `nn.MultiheadAttention(batch_first=True)` and
passes a boolean `key_padding_mask`. Internally, MHA converts that to an
additive float `attn_mask`, which typically excludes FLASH_ATTENTION on
most PyTorch builds. This script reproduces tabasco's realistic call
shape at production dims (B=256, H=8, N=71, D=16 per head for hidden=128;
D=32 per head for hidden=256 — both exercised here) and reports which
backend each forced selection resolves to.

It also tests a "bare SDPA" variant: querying `F.scaled_dot_product_attention`
directly with a boolean mask converted to float, to see whether we could
unlock FLASH by restructuring the call.

Usage (GPU required):
    python hpc-scripts/tabasco/bench/diagnose_sdpa.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


DEVICE = "cuda"


def time_ms(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def mha_call(mha, x, key_padding_mask):
    out, _ = mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
    return out


def bare_sdpa_call(q, k, v, additive_mask, scale):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask, scale=scale)


def make_mha_inputs(batch, num_atoms, hidden, heads, dtype, real_atoms):
    x = torch.randn(
        batch, num_atoms, hidden, device=DEVICE, dtype=dtype, requires_grad=True
    )
    key_padding_mask = torch.zeros(batch, num_atoms, dtype=torch.bool, device=DEVICE)
    for b in range(batch):
        key_padding_mask[b, real_atoms:] = True
    mha = torch.nn.MultiheadAttention(
        embed_dim=hidden,
        num_heads=heads,
        batch_first=True,
        bias=True,
    ).to(device=DEVICE, dtype=dtype)
    return mha, x, key_padding_mask


def make_bare_inputs(batch, num_atoms, hidden, heads, dtype, real_atoms):
    head_dim = hidden // heads
    q = torch.randn(
        batch,
        heads,
        num_atoms,
        head_dim,
        device=DEVICE,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        batch,
        heads,
        num_atoms,
        head_dim,
        device=DEVICE,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch,
        heads,
        num_atoms,
        head_dim,
        device=DEVICE,
        dtype=dtype,
        requires_grad=True,
    )
    bool_mask = torch.zeros(batch, num_atoms, dtype=torch.bool, device=DEVICE)
    bool_mask[:, real_atoms:] = True
    neg_inf = torch.finfo(dtype).min
    additive = torch.where(bool_mask[:, None, None, :], neg_inf, 0.0).to(dtype)
    scale = head_dim**-0.5
    return q, k, v, additive, scale


def try_backend(label, backend, fn, do_backward=True):
    try:

        def run():
            with sdpa_kernel([backend]):
                out = fn()
            if do_backward:
                out.sum().backward(retain_graph=True)

        run()
        t = time_ms(run)
        phase = "fwd+bwd" if do_backward else "fwd"
        print(f"  {label:32s}  {phase:7s}  OK     {t:6.2f} ms/iter")
        return True
    except Exception as e:
        phase = "fwd+bwd" if do_backward else "fwd"
        first_line = str(e).splitlines()[0] if str(e) else type(e).__name__
        print(f"  {label:32s}  {phase:7s}  FAIL   {first_line[:120]}")
        return False


def scenario_mha(name, batch, num_atoms, hidden, heads, dtype, real_atoms, padded):
    mask_label = "with padding_mask" if padded else "no mask"
    print(
        f"\n─── {name}  B={batch} N={num_atoms} hidden={hidden} H={heads} "
        f"D={hidden//heads} dtype={dtype} {mask_label} ───"
    )
    for label, backend in [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("MATH", SDPBackend.MATH),
        ("CUDNN_ATTENTION", getattr(SDPBackend, "CUDNN_ATTENTION", None)),
    ]:
        if backend is None:
            continue
        mha, x, kpm = make_mha_inputs(
            batch, num_atoms, hidden, heads, dtype, real_atoms
        )
        kpm_arg = kpm if padded else None
        try_backend(label, backend, lambda: mha_call(mha, x, kpm_arg))


def scenario_bare(name, batch, num_atoms, hidden, heads, dtype, real_atoms, padded):
    mask_label = "additive -inf mask" if padded else "no mask"
    print(
        f"\n─── {name}  B={batch} N={num_atoms} hidden={hidden} H={heads} "
        f"D={hidden//heads} dtype={dtype} {mask_label} (bare SDPA) ───"
    )
    for label, backend in [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("MATH", SDPBackend.MATH),
        ("CUDNN_ATTENTION", getattr(SDPBackend, "CUDNN_ATTENTION", None)),
    ]:
        if backend is None:
            continue
        q, k, v, additive, scale = make_bare_inputs(
            batch, num_atoms, hidden, heads, dtype, real_atoms
        )
        mask_arg = additive if padded else None
        try_backend(label, backend, lambda: bare_sdpa_call(q, k, v, mask_arg, scale))


def main():
    print(f"torch: {torch.__version__}")
    print(f"cuda:  {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name()}")
    print(
        f"flash_enabled={torch.backends.cuda.flash_sdp_enabled()} "
        f"mem_eff_enabled={torch.backends.cuda.mem_efficient_sdp_enabled()} "
        f"math_enabled={torch.backends.cuda.math_sdp_enabled()}"
    )

    # Tabasco GEOM mild: hidden=128, heads=8, head_dim=16. N≈71 (max).
    # Production workload expands BS×8 via augmentation → effective BS=2048.
    # Use 256 here so diagnose runs fast while still exercising the kernels.
    scenario_mha(
        "A: MHA w/ padding_mask (prod path)",
        batch=256,
        num_atoms=71,
        hidden=128,
        heads=8,
        dtype=torch.bfloat16,
        real_atoms=40,
        padded=True,
    )
    scenario_mha(
        "B: MHA w/o mask (best-case for FLASH)",
        batch=256,
        num_atoms=71,
        hidden=128,
        heads=8,
        dtype=torch.bfloat16,
        real_atoms=40,
        padded=False,
    )
    scenario_bare(
        "C: bare SDPA w/ additive -inf mask (what MHA lowers to)",
        batch=256,
        num_atoms=71,
        hidden=128,
        heads=8,
        dtype=torch.bfloat16,
        real_atoms=40,
        padded=True,
    )
    scenario_bare(
        "D: bare SDPA w/o mask",
        batch=256,
        num_atoms=71,
        hidden=128,
        heads=8,
        dtype=torch.bfloat16,
        real_atoms=40,
        padded=False,
    )
    # fp16 variant (current prod precision)
    scenario_mha(
        "E: MHA fp16 w/ padding_mask",
        batch=256,
        num_atoms=71,
        hidden=128,
        heads=8,
        dtype=torch.float16,
        real_atoms=40,
        padded=True,
    )


if __name__ == "__main__":
    main()
