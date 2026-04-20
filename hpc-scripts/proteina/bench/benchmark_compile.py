"""Benchmark torch.compile on proteina training throughput.

Isolates the compile question: fake data, fixed batch size, only the
model changes. For each (compile_mode, seq_len, model_type) the script
spawns a fresh subprocess, trains N steps, and reports steady-state
steps/sec with the first `warmup` steps dropped (compile warmup skew).

Usage:
    python benchmark_compile.py \\
        --seq_lens 256 512 \\
        --model_types baseline \\
        --compile_modes off default \\
        --num_steps 100 --warmup 30 \\
        --output_csv evaluation/proteina/bench/results/compile.csv
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


DEFAULT_BS = {
    ("baseline", 128): 12,
    ("baseline", 256): 8,
    ("baseline", 512): 6,
    ("repa", 128): 8,
    ("repa", 256): 6,
    ("repa", 512): 4,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_lens", type=int, nargs="+", default=[256, 512])
    p.add_argument(
        "--model_types",
        nargs="+",
        default=["baseline"],
        choices=["baseline", "repa"],
    )
    p.add_argument(
        "--compile_modes",
        nargs="+",
        default=["off", "default"],
        choices=["off", "default", "reduce-overhead", "max-autotune"],
    )
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=WARMUP_STEPS_DEFAULT)
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override default batch size lookup",
    )
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument(
        "--output_csv",
        type=str,
        default="evaluation/proteina/bench/results/compile.csv",
    )
    return p.parse_args()


def _run_one(result_queue, seq_len, model_type, compile_mode, batch_size, num_steps):
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
        cfg.run_name_ = f"bench_compile_{seq_len}_{model_type}_{compile_mode}"

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

        tmp_dir = tempfile.mkdtemp(prefix="bench_compile_")
        try:
            model_cls = ProteinaREPA if model_type == "repa" else Proteina
            model = model_cls(cfg, store_dir=tmp_dir)

            if compile_mode != "off":
                model.nn = torch.compile(model.nn, mode=compile_mode)

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
        if "out of memory" in str(e).lower():
            result_queue.put({"status": "oom", "error": str(e)})
        else:
            result_queue.put(
                {"status": "error", "error": f"{e}\n{traceback.format_exc()}"}
            )
    except Exception as e:
        result_queue.put({"status": "error", "error": f"{e}\n{traceback.format_exc()}"})


def main():
    args = parse_args()
    rows = []
    for seq_len in args.seq_lens:
        for model_type in args.model_types:
            bs = args.batch_size or DEFAULT_BS.get((model_type, seq_len))
            if bs is None:
                print(
                    f"!! No default BS for ({model_type}, {seq_len}); pass --batch_size"
                )
                continue
            for mode in args.compile_modes:
                print(
                    f"\n>>> seq_len={seq_len} model={model_type} "
                    f"compile={mode} bs={bs} steps={args.num_steps}"
                )
                clear_compile_cache()
                res = run_in_subprocess(
                    _run_one,
                    args=(seq_len, model_type, mode, bs, args.num_steps),
                    timeout=args.timeout,
                )
                row = {
                    "seq_len": seq_len,
                    "model_type": model_type,
                    "compile_mode": mode,
                    "batch_size": bs,
                    "status": res["status"],
                    "peak_mem_gb": res.get("peak_mem_gb"),
                }
                if res["status"] == "ok":
                    stats = summarize(res["times_s"], args.warmup)
                    row.update(stats.as_row())
                    row["samples_per_sec"] = round(bs * stats.steps_per_sec, 2)
                    print(
                        f"    median={stats.median_s:.3f}s/step  "
                        f"{stats.steps_per_sec:.2f} steps/s  "
                        f"peak={res.get('peak_mem_gb')} GB"
                    )
                else:
                    print(f"    FAILED: {res.get('error', res['status'])[:200]}")
                rows.append(row)

    print("\n" + "=" * 80)
    print("COMPILE BENCHMARK RESULTS")
    print("=" * 80)
    print_table(rows, drop_cols=["n_total", "n_warmup", "total_s", "p10_s", "p99_s"])

    write_csv(rows, args.output_csv)
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
