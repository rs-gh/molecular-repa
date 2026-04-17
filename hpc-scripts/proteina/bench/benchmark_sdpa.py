"""Benchmark SDPA kernel backends (FLASH / EFFICIENT / MATH / CUDNN).

Proteina calls F.scaled_dot_product_attention with attn_mask set (a pair
bias). On A100 bf16 this typically picks EFFICIENT_ATTENTION; FLASH
may reject a non-None attn_mask on older torch builds. This script
forces each backend via torch.nn.attention.sdpa_kernel and measures
steady-state training steps/sec.

Uses fake data so it's fast and isolates the attention backend.

Usage:
    python benchmark_sdpa.py \\
        --backends default flash efficient math cudnn \\
        --with_compile --without_compile \\
        --num_steps 100 --warmup 30
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(__file__))

from _bench_common import (  # noqa: E402
    WARMUP_STEPS_DEFAULT,
    build_fake_pdb_datamodule,
    build_step_timer_callback,
    clear_compile_cache,
    load_proteina_cfg,
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
        default=["default", "efficient", "math"],
        choices=BACKENDS,
    )
    p.add_argument(
        "--compile_modes",
        nargs="+",
        default=["off", "default"],
        choices=["off", "default"],
    )
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--model_type", default="baseline", choices=["baseline", "repa"])
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=WARMUP_STEPS_DEFAULT)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument(
        "--output_csv",
        type=str,
        default="evaluation/proteina/results/bench/sdpa.csv",
    )
    return p.parse_args()


def _sdpa_ctx(backend: str):
    """Return a context manager forcing the given SDPA backend, or nullcontext."""
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
    result_queue, seq_len, model_type, compile_mode, backend, batch_size, num_steps
):
    try:
        os.environ.setdefault("DATA_PATH", "/rds/user/sr2173/hpc-work/proteina/data")

        import proteinfoundation.repa.pyg_compat  # noqa: F401

        import lightning as L
        import torch
        from omegaconf import OmegaConf

        from proteinfoundation.proteinflow.proteina import Proteina
        from proteinfoundation.repa.proteina_repa import ProteinaREPA

        cfg = load_proteina_cfg("training_ca")
        cfg.hardware.ngpus_per_node_ = 1
        cfg.hardware.nnodes_ = 1
        cfg.run_name_ = f"bench_sdpa_{backend}_{compile_mode}"
        # Ensure SDPA path is taken (vs manual einsum)
        if "use_sdpa" in cfg.get("model", {}).get("nn", {}):
            cfg.model.nn.use_sdpa = True

        if model_type == "repa":
            repa_cfg = {
                "gearnet_ckpt_path": os.path.join(
                    os.environ["DATA_PATH"],
                    "metric_factory/model_weights/gearnet_ca.pth",
                ),
                "layers": [4],
                "lambda_repa": 0.5,
                "combination_mode": "additive",
                "similarity_type": "cosine",
                "averaging": "per_residue",
                "projector_hidden_dim": 512,
                "projector_num_layers": 3,
            }
            OmegaConf.update(cfg, "repa", repa_cfg)

        torch.set_float32_matmul_precision("medium")
        L.seed_everything(42)

        tmp_dir = tempfile.mkdtemp(prefix="bench_sdpa_")
        try:
            model_cls = ProteinaREPA if model_type == "repa" else Proteina
            model = model_cls(cfg, store_dir=tmp_dir)

            if compile_mode != "off":
                model.nn = torch.compile(model.nn, mode=compile_mode)

            # Wrap the model's forward in the SDPA-kernel context
            # so every attention op inside dispatches to the chosen backend.
            orig_forward = model.nn.forward
            ctx_factory = lambda: _sdpa_ctx(backend)  # noqa: E731

            def forward_with_sdpa(*a, **kw):
                with ctx_factory():
                    return orig_forward(*a, **kw)

            model.nn.forward = forward_with_sdpa

            datamodule = build_fake_pdb_datamodule(
                max_len=seq_len,
                batch_size=batch_size,
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
                precision="bf16-mixed",
                gradient_clip_algorithm="norm",
                gradient_clip_val=1.0,
            )

            torch.cuda.reset_peak_memory_stats()
            trainer.fit(model, datamodule)
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

            result_queue.put(
                {
                    "status": "ok",
                    "times_s": timer.times_s,
                    "peak_mem_gb": round(peak_mem_gb, 2),
                }
            )
        finally:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            result_queue.put({"status": "oom", "error": str(e)})
        elif "no available kernel" in msg or "not supported" in msg:
            # Backend rejected the inputs (e.g., FLASH + attn_mask on older torch)
            result_queue.put({"status": "unsupported", "error": str(e)})
        else:
            result_queue.put(
                {
                    "status": "error",
                    "error": f"{e}\n{traceback.format_exc()}",
                }
            )
    except Exception as e:
        result_queue.put(
            {
                "status": "error",
                "error": f"{e}\n{traceback.format_exc()}",
            }
        )


def main():
    args = parse_args()
    rows = []
    for mode in args.compile_modes:
        for backend in args.backends:
            print(
                f"\n>>> backend={backend} compile={mode} "
                f"bs={args.batch_size} seq={args.seq_len} steps={args.num_steps}"
            )
            clear_compile_cache()
            res = run_in_subprocess(
                _run_one,
                args=(
                    args.seq_len,
                    args.model_type,
                    mode,
                    backend,
                    args.batch_size,
                    args.num_steps,
                ),
                timeout=args.timeout,
            )
            row = {
                "seq_len": args.seq_len,
                "model_type": args.model_type,
                "sdpa_backend": backend,
                "compile_mode": mode,
                "batch_size": args.batch_size,
                "status": res["status"],
                "peak_mem_gb": res.get("peak_mem_gb"),
            }
            if res["status"] == "ok":
                stats = summarize(res["times_s"], args.warmup)
                row.update(stats.as_row())
                row["samples_per_sec"] = round(
                    args.batch_size * stats.steps_per_sec,
                    2,
                )
                print(
                    f"    median={stats.median_s:.3f}s/step  "
                    f"{stats.steps_per_sec:.2f} steps/s  "
                    f"peak={res.get('peak_mem_gb')} GB"
                )
            else:
                print(f"    {res['status'].upper()}: " f"{res.get('error', '')[:200]}")
            rows.append(row)

    print("\n" + "=" * 80)
    print("SDPA BACKEND BENCHMARK RESULTS")
    print("=" * 80)
    print_table(rows, drop_cols=["n_total", "n_warmup", "total_s", "p10_s", "p99_s"])

    write_csv(rows, args.output_csv)
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
