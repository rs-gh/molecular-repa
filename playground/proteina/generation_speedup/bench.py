"""Throughput benchmark: measure generation wall time across (mode, length) grid.

Primary comparison: mode A (eager_fp32, current) vs mode D (compile_bf16, candidate).
B and C are optional diagnostic modes (use --modes all).

Usage:
    python playground/proteina/generation_speedup/bench.py
    python playground/proteina/generation_speedup/bench.py --modes all
    python playground/proteina/generation_speedup/bench.py --lengths 128 512
    python playground/proteina/generation_speedup/bench.py --n_warmup 1 --n_steady 3
    python playground/proteina/generation_speedup/bench.py --max_bs  # probe max batch size per length
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _speedup_utils import prepare_model, MODES, cuda_timer  # noqa: E402
from checkpoints import resolve_active  # noqa: E402

# Sampling params mirrored from inference_fid_60m_baseline.yaml / inference_base.yaml
GENERATE_KWARGS = dict(
    dt=0.0025,
    self_cond=True,
    cath_code=None,
    guidance_weight=1.0,
    autoguidance_ratio=1.0,
    schedule_mode="log",
    schedule_p=2.0,
    sampling_mode="sc",
    sc_scale_noise=0.45,
    sc_scale_score=1.0,
    gt_mode="1/t",
    gt_p=1.0,
    gt_clamp_val=None,
)

BATCH_SIZE = 8


def run_cell(
    model, autocast_ctx, nres: int, batch_size: int, n_batches: int
) -> list[float]:
    """Run `n_batches` generations at `(nres, batch_size)`. Return wall seconds per batch."""
    timings = []
    for _ in range(n_batches):
        with cuda_timer() as t, torch.no_grad():
            with autocast_ctx():
                x = model.generate(
                    nsamples=batch_size,
                    n=nres,
                    dtype=torch.float32,  # generate() internal dtype; autocast wraps the forward
                    **GENERATE_KWARGS,
                )
                _ = model.samples_to_atom37(x).cpu()
        timings.append(t["seconds"])
    return timings


def probe_max_bs(
    model, autocast_ctx, nres: int, start_bs: int, probe_kwargs: dict
) -> int:
    """Doubling search then binary search to find largest batch that fits in VRAM.

    Uses probe_kwargs (typically short dt / few steps) so each trial is fast.
    """

    def try_bs(bs: int) -> bool:
        try:
            torch.cuda.empty_cache()
            with torch.no_grad(), autocast_ctx():
                x = model.generate(
                    nsamples=bs, n=nres, dtype=torch.float32, **probe_kwargs
                )
                _ = model.samples_to_atom37(x).cpu()
            return True
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False

    # doubling phase
    bs = start_bs
    while try_bs(bs):
        if bs >= 512:
            return bs
        bs *= 2

    # binary search between bs//2 (known good) and bs (known OOM)
    lo, hi = bs // 2, bs
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if try_bs(mid):
            lo = mid
        else:
            hi = mid
    return lo


def bench_max_bs(
    modes: list[str], lengths: list[int], start_bs: int, probe_steps: int
) -> pd.DataFrame:
    records = []
    active = resolve_active()
    ckpt = active[0]
    print(f"Using checkpoint: {ckpt.name} -> {ckpt.ckpt_path}")
    print(
        f"Probe config: start_bs={start_bs}, probe_steps={probe_steps} (dt={1/probe_steps:.4f})"
    )

    # Short-run kwargs: same sampler settings but far fewer ODE steps.
    probe_kwargs = {**GENERATE_KWARGS, "dt": 1.0 / probe_steps}

    for mode in modes:
        print(f"\n=== mode={mode} ===")
        model, autocast_ctx = prepare_model(ckpt.ckpt_path, ckpt.is_repa, mode)
        for nres in lengths:
            print(f"  length={nres}: probing max batch size ...", end=" ", flush=True)
            # warmup at start_bs to trigger compile / CUDAGraph capture if needed
            with torch.no_grad(), autocast_ctx():
                _ = model.generate(
                    nsamples=start_bs, n=nres, dtype=torch.float32, **probe_kwargs
                )
            torch.cuda.empty_cache()
            max_bs = probe_max_bs(
                model, autocast_ctx, nres, start_bs=start_bs, probe_kwargs=probe_kwargs
            )
            print(f"max_bs={max_bs}")
            sys.stdout.flush()
            records.append(
                dict(
                    ckpt=ckpt.name,
                    mode=mode,
                    length=nres,
                    max_batch_size=max_bs,
                    probe_steps=probe_steps,
                    start_bs=start_bs,
                )
            )
        del model
        torch.cuda.empty_cache()

    return pd.DataFrame(records)


def bench(
    modes: list[str], lengths: list[int], n_warmup: int, n_steady: int
) -> pd.DataFrame:
    records = []
    active = resolve_active()
    assert len(active) >= 1, "No active checkpoints — uncomment one in checkpoints.py"
    ckpt = active[0]
    print(f"Using checkpoint: {ckpt.name} -> {ckpt.ckpt_path}")

    for mode in modes:
        print(f"\n=== mode={mode} ===")
        # Fresh model load per mode so compile state doesn't bleed across modes.
        model, autocast_ctx = prepare_model(ckpt.ckpt_path, ckpt.is_repa, mode)
        for nres in lengths:
            print(
                f"  length={nres}: warmup×{n_warmup} steady×{n_steady} bs={BATCH_SIZE}"
            )
            sys.stdout.flush()
            warmup_times = run_cell(model, autocast_ctx, nres, BATCH_SIZE, n_warmup)
            steady_times = run_cell(model, autocast_ctx, nres, BATCH_SIZE, n_steady)
            steady_arr = torch.tensor(steady_times)
            rec = dict(
                ckpt=ckpt.name,
                mode=mode,
                length=nres,
                batch_size=BATCH_SIZE,
                n_warmup=n_warmup,
                n_steady=n_steady,
                warmup_seconds_total=sum(warmup_times),
                warmup_seconds_first=warmup_times[0] if warmup_times else float("nan"),
                steady_seconds_mean=float(steady_arr.mean()),
                steady_seconds_std=float(steady_arr.std()) if n_steady > 1 else 0.0,
                s_per_sample=float(steady_arr.mean()) / BATCH_SIZE,
            )
            print(
                f"    warmup[0]={rec['warmup_seconds_first']:.2f}s  "
                f"steady={rec['steady_seconds_mean']:.2f}±{rec['steady_seconds_std']:.2f}s "
                f"({rec['s_per_sample']:.3f}s/sample)"
            )
            sys.stdout.flush()
            records.append(rec)
        # Free model before next mode to keep peak memory bounded
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    # Compute speedup vs eager_fp32 per length
    baseline = df[df["mode"] == "eager_fp32"].set_index("length")["s_per_sample"]
    df["speedup_vs_eager_fp32"] = df.apply(
        lambda r: baseline.get(r["length"], float("nan")) / r["s_per_sample"]
        if r["length"] in baseline.index
        else float("nan"),
        axis=1,
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--modes",
        type=str,
        default="A_D",
        help="'A_D' (eager_fp32, compile_bf16), 'all' (full 4-mode grid), or comma list",
    )
    ap.add_argument("--lengths", type=int, nargs="*", default=[128, 256, 384, 512])
    ap.add_argument("--n_warmup", type=int, default=1)
    ap.add_argument("--n_steady", type=int, default=3)
    ap.add_argument(
        "--max_bs", action="store_true", help="probe max batch size per (mode, length)"
    )
    ap.add_argument(
        "--start_bs",
        type=int,
        default=8,
        help="starting batch size for doubling search (default 8, known safe)",
    )
    ap.add_argument(
        "--probe_steps",
        type=int,
        default=20,
        help="ODE steps for each VRAM probe trial (default 20; fast, ~5%% of full 400)",
    )
    args = ap.parse_args()

    if args.modes == "A_D":
        modes = ["eager_fp32", "compile_bf16"]
    elif args.modes == "all":
        modes = list(MODES)
    else:
        modes = args.modes.split(",")
    for m in modes:
        assert m in MODES, f"unknown mode {m!r}; choose from {MODES}"

    if args.max_bs:
        df = bench_max_bs(
            modes, args.lengths, start_bs=args.start_bs, probe_steps=args.probe_steps
        )
        out_csv = HERE / "max_bs_results.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nWrote {out_csv}")
        print("\n=== Max batch size per (mode, length) ===")
        pivot = df.pivot(index="length", columns="mode", values="max_batch_size")
        print(pivot.to_string())
        return

    df = bench(modes, args.lengths, args.n_warmup, args.n_steady)

    out_csv = HERE / "bench_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Pretty summary
    print("\n=== Summary (s/sample) ===")
    pivot = df.pivot(index="length", columns="mode", values="s_per_sample")
    print(pivot.round(3).to_string())
    print("\n=== Speedup vs eager_fp32 ===")
    spivot = df.pivot(index="length", columns="mode", values="speedup_vs_eager_fp32")
    print(spivot.round(2).to_string())


if __name__ == "__main__":
    main()
