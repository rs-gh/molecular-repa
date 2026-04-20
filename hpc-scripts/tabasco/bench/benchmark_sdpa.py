"""Benchmark SDPA kernel backends for tabasco's transformer attention.

Tabasco's `Attention` module wraps `nn.MultiheadAttention` which dispatches
internally to `F.scaled_dot_product_attention`. The wrapper passes
`batch_first=True`, `need_weights=False` (default forward call path via
`TransformerBlock`), and a boolean `key_padding_mask` — conditions that
*should* allow FLASH or EFFICIENT backends. This script verifies which
backend is actually picked and measures the throughput/memory delta of
forcing each one.

Any backend that rejects tabasco's call shape is reported as `unsupported`
so we can diagnose via `diagnose_sdpa.py`.

Usage:
    python benchmark_sdpa.py \\
        --backends default flash efficient math cudnn \\
        --compile_modes off default \\
        --experiment geom/mild \\
        --num_steps 100 --warmup 30
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(__file__))

from _bench_common import (  # noqa: E402
    GEOM_MAX_ATOMS,
    WARMUP_STEPS_DEFAULT,
    build_fake_tabasco_datamodule,
    build_step_timer_callback,
    clear_compile_cache,
    load_tabasco_cfg,
    print_table,
    run_in_subprocess,
    summarize,
    write_csv,
)


BACKENDS = ["default", "flash", "efficient", "math", "cudnn"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backends",
        nargs="+",
        default=["default", "flash", "efficient", "math"],
        choices=BACKENDS,
    )
    p.add_argument(
        "--compile_modes",
        nargs="+",
        default=["off", "default"],
        choices=["off", "default", "reduce-overhead"],
    )
    p.add_argument("--experiment", default="geom/mild")
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_atoms", type=int, default=GEOM_MAX_ATOMS)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=WARMUP_STEPS_DEFAULT)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument(
        "--output_csv",
        type=str,
        default="evaluation/tabasco/bench/results/sdpa.csv",
    )
    return p.parse_args()


def _sdpa_ctx(backend: str):
    """Context manager forcing the given SDPA backend, or nullcontext for 'default'."""
    import contextlib

    if backend == "default":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    mapping = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
        "cudnn": getattr(SDPBackend, "CUDNN_ATTENTION", None),
    }
    chosen = mapping[backend]
    if chosen is None:
        raise RuntimeError(
            f"SDPBackend.{backend.upper()} not available in this torch build"
        )
    return sdpa_kernel([chosen])


def _run_one(
    result_queue,
    experiment,
    compile_mode,
    backend,
    precision,
    batch_size,
    num_atoms,
    num_steps,
):
    try:
        import hydra
        import lightning as L
        import torch

        overrides = [
            f"model.compile={'true' if compile_mode != 'off' else 'false'}",
        ]
        if compile_mode != "off":
            overrides.append(f"model.compile_mode={compile_mode}")
        cfg = load_tabasco_cfg(experiment, extra_overrides=overrides)

        torch.set_float32_matmul_precision("high")
        L.seed_everything(42)

        lightning_module = hydra.utils.instantiate(cfg.lightning_module)

        # Wrap the inner net.forward in the SDPA-kernel context so every
        # attention call inside dispatches to the forced backend.
        net = lightning_module.model.net
        inner = net._orig_mod if hasattr(net, "_orig_mod") else net
        orig_forward = inner.forward
        ctx_factory = lambda: _sdpa_ctx(backend)  # noqa: E731

        def forward_with_sdpa(*a, **kw):
            with ctx_factory():
                return orig_forward(*a, **kw)

        inner.forward = forward_with_sdpa

        is_repa = "repa_loss" in cfg.get("model", {}) and cfg.model.get("repa_loss")
        datamodule = build_fake_tabasco_datamodule(
            batch_size=batch_size,
            num_atoms=num_atoms,
            with_smiles=bool(is_repa),
            num_workers=0,
        )
        timer = build_step_timer_callback()

        trainer = L.Trainer(
            max_steps=num_steps,
            accelerator="gpu",
            devices=1,
            num_nodes=1,
            callbacks=[timer],
            logger=False,
            log_every_n_steps=num_steps,
            enable_progress_bar=False,
            enable_checkpointing=False,
            check_val_every_n_epoch=None,
            val_check_interval=num_steps + 1,
            strategy="auto",
            precision=precision,
            gradient_clip_algorithm="norm",
            gradient_clip_val=0.5,
        )

        torch.cuda.reset_peak_memory_stats()
        trainer.fit(lightning_module, datamodule)
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

        result_queue.put(
            {
                "status": "ok",
                "times_s": timer.times_s,
                "peak_mem_gb": round(peak_mem_gb, 2),
            }
        )
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            result_queue.put({"status": "oom", "error": str(e)})
        elif "no available kernel" in msg or "not supported" in msg:
            result_queue.put({"status": "unsupported", "error": str(e)})
        else:
            result_queue.put(
                {"status": "error", "error": f"{e}\n{traceback.format_exc()}"}
            )
    except Exception as e:
        result_queue.put({"status": "error", "error": f"{e}\n{traceback.format_exc()}"})


def main():
    args = parse_args()
    rows = []
    for mode in args.compile_modes:
        for backend in args.backends:
            print(
                f"\n>>> exp={args.experiment} backend={backend} compile={mode} "
                f"bs={args.batch_size} atoms={args.num_atoms} steps={args.num_steps}"
            )
            clear_compile_cache()
            res = run_in_subprocess(
                _run_one,
                args=(
                    args.experiment,
                    mode,
                    backend,
                    args.precision,
                    args.batch_size,
                    args.num_atoms,
                    args.num_steps,
                ),
                timeout=args.timeout,
            )
            row = {
                "experiment": args.experiment,
                "sdpa_backend": backend,
                "compile_mode": mode,
                "precision": args.precision,
                "batch_size": args.batch_size,
                "num_atoms": args.num_atoms,
                "status": res["status"],
                "peak_mem_gb": res.get("peak_mem_gb"),
            }
            if res["status"] == "ok":
                stats = summarize(res["times_s"], args.warmup)
                row.update(stats.as_row())
                row["samples_per_sec"] = round(args.batch_size * stats.steps_per_sec, 2)
                print(
                    f"    median={stats.median_s:.3f}s/step  "
                    f"{stats.steps_per_sec:.2f} steps/s  "
                    f"peak={res.get('peak_mem_gb')} GB"
                )
            else:
                print(f"    {res['status'].upper()}: {res.get('error', '')[:200]}")
            rows.append(row)

    print("\n" + "=" * 80)
    print("TABASCO SDPA BACKEND BENCHMARK RESULTS")
    print("=" * 80)
    print_table(rows, drop_cols=["n_total", "n_warmup", "total_s", "p10_s", "p99_s"])

    write_csv(rows, args.output_csv)
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
