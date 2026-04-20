"""Batch-size sweep for tabasco training.

Binary-searches the maximum batch size before OOM for each
(experiment, compile_mode, gc_layers) combination. Each trial runs in a
spawn subprocess so OOM kills the child cleanly.

Usage:
    python hpc-scripts/tabasco/bench/batch_size_sweep.py \\
        --experiments geom/mild geom/chemprop_tradeoff \\
        --min_bs 64 --max_bs 1024 \\
        --num_steps 20 --timeout 300 \\
        --output_csv evaluation/tabasco/bench/results/batch_size_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

from _bench_common import (  # noqa: E402
    GEOM_MAX_ATOMS,
    apply_gradient_checkpointing,
    build_fake_tabasco_datamodule,
    load_tabasco_cfg,
    run_in_subprocess,
)


IS_REPA = {
    "geom/mild": False,
    "geom/chemprop_tradeoff": True,
    "geom/mace_cached_tradeoff": True,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiments",
        nargs="+",
        default=["geom/mild"],
    )
    p.add_argument("--min_bs", type=int, default=64)
    p.add_argument("--max_bs", type=int, default=1024)
    p.add_argument("--num_atoms", type=int, default=GEOM_MAX_ATOMS)
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument(
        "--compile_modes",
        nargs="+",
        default=["off", "reduce-overhead"],
        choices=["off", "default", "reduce-overhead", "max-autotune"],
    )
    p.add_argument("--precision", default="16")
    p.add_argument(
        "--gc_variants",
        nargs="+",
        type=int,
        default=[0],
        help="Number of transformer blocks to gradient-checkpoint (last N). 0 = none.",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default="evaluation/tabasco/bench/results/batch_size_sweep.csv",
    )
    return p.parse_args()


def _run_one(
    result_queue,
    experiment,
    compile_mode,
    gc_layers,
    precision,
    batch_size,
    num_atoms,
    num_steps,
    warmup_steps,
):
    try:
        import hydra
        import lightning as L
        import torch

        # Load with compile=false so we can patch layers before compiling.
        # Compile is re-applied manually below after grad-ckpt is wired in.
        cfg = load_tabasco_cfg(experiment, extra_overrides=["model.compile=false"])

        torch.set_float32_matmul_precision("high")
        L.seed_everything(42)

        lightning_module = hydra.utils.instantiate(cfg.lightning_module)

        # Grad-ckpt must run BEFORE compile (OptimizedModule hides .layers).
        if gc_layers > 0:
            apply_gradient_checkpointing(lightning_module, gc_layers)

        if compile_mode != "off":
            inner = lightning_module.model.net
            lightning_module.model.net = torch.compile(inner, mode=compile_mode)

        class SteadyStateTimer(L.Callback):
            def __init__(self, warmup):
                self.warmup = warmup
                self.count = 0
                self.steady_start = None

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                self.count += 1
                if self.count == self.warmup:
                    torch.cuda.synchronize()
                    self.steady_start = time.time()

        datamodule = build_fake_tabasco_datamodule(
            batch_size=batch_size,
            num_atoms=num_atoms,
            with_smiles=IS_REPA.get(experiment, False),
            num_workers=0,
        )
        timer = SteadyStateTimer(warmup=min(warmup_steps, max(num_steps - 1, 0)))
        trainer = L.Trainer(
            max_steps=num_steps,
            accelerator="gpu",
            devices=1,
            num_nodes=1,
            callbacks=[timer],
            logger=False,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
            check_val_every_n_epoch=None,
            val_check_interval=999999,
            strategy="auto",
            precision=precision,
            gradient_clip_algorithm="norm",
            gradient_clip_val=0.5,
        )

        torch.cuda.reset_peak_memory_stats()
        total_start = time.time()
        trainer.fit(lightning_module, datamodule)
        torch.cuda.synchronize()
        total_elapsed = time.time() - total_start
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

        steady_steps = max(timer.count - timer.warmup, 0)
        if timer.steady_start is not None and steady_steps > 0:
            steady_elapsed = time.time() - timer.steady_start
            steps_per_sec = steady_steps / steady_elapsed
            time_per_step = steady_elapsed / steady_steps
        else:
            steps_per_sec = timer.count / max(total_elapsed, 1e-6)
            time_per_step = total_elapsed / max(timer.count, 1)

        result_queue.put(
            {
                "status": "ok",
                "peak_mem_gb": round(peak_mem_gb, 2),
                "steps_per_sec": round(steps_per_sec, 3),
                "time_per_step": round(time_per_step, 3),
                "steady_steps": steady_steps,
                "total_elapsed_s": round(total_elapsed, 2),
            }
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            result_queue.put({"status": "oom", "error": str(e)[:200]})
        else:
            result_queue.put(
                {"status": "error", "error": f"{e}\n{traceback.format_exc()}"}
            )
    except Exception as e:
        result_queue.put({"status": "error", "error": f"{e}\n{traceback.format_exc()}"})


def binary_search_max_bs(
    experiment,
    compile_mode,
    gc_layers,
    precision,
    num_atoms,
    lo,
    hi,
    num_steps,
    warmup_steps,
    timeout,
):
    best_bs = lo - 1
    best_result = None
    print(f"\n{'='*70}")
    print(
        f"  {experiment}  compile={compile_mode}  gc_layers={gc_layers}  precision={precision}"
    )
    print(f"  Searching BS in [{lo}, {hi}]")
    print(f"{'='*70}")

    while lo <= hi:
        mid = (lo + hi) // 2
        print(f"  Testing BS={mid}...", end=" ", flush=True)

        cache_dir = f"/tmp/torchinductor_{os.environ.get('USER', 'unknown')}"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)

        res = run_in_subprocess(
            _run_one,
            args=(
                experiment,
                compile_mode,
                gc_layers,
                precision,
                mid,
                num_atoms,
                num_steps,
                warmup_steps,
            ),
            timeout=timeout,
        )
        if res["status"] == "ok":
            print(
                f"OK  (peak={res['peak_mem_gb']:.1f}GB, "
                f"{res['steps_per_sec']:.2f} steps/s)"
            )
            best_bs = mid
            best_result = res
            lo = mid + 1
        else:
            print(f"FAIL ({res.get('status')}: {res.get('error', '')[:80]})")
            hi = mid - 1
    return best_bs, best_result


def _write_partial(rows, path):
    """Write CSV after each config so timeouts don't waste progress."""
    if not rows or not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = []
    for exp in args.experiments:
        for mode in args.compile_modes:
            for gc in args.gc_variants:
                max_bs, best = binary_search_max_bs(
                    exp,
                    mode,
                    gc,
                    args.precision,
                    args.num_atoms,
                    args.min_bs,
                    args.max_bs,
                    args.num_steps,
                    args.warmup_steps,
                    args.timeout,
                )
                row = {
                    "experiment": exp,
                    "compile_mode": mode,
                    "gc_layers": gc,
                    "precision": args.precision,
                    "num_atoms": args.num_atoms,
                    "max_bs": max_bs,
                    "peak_mem_gb": best["peak_mem_gb"] if best else None,
                    "steps_per_sec": best["steps_per_sec"] if best else None,
                    "time_per_step": best["time_per_step"] if best else None,
                    "accum_to_256": math.ceil(256 / max_bs) if max_bs > 0 else None,
                }
                rows.append(row)
                _write_partial(rows, args.output_csv)

    print("\n" + "=" * 100)
    print("  TABASCO BATCH-SIZE SWEEP RESULTS")
    print("=" * 100)
    print(
        f"{'experiment':>28} {'compile':>16} {'gc':>4} {'max_bs':>7} "
        f"{'peak_GB':>8} {'steps/s':>8} {'s/step':>8} {'accum→256':>10}"
    )
    print("-" * 100)
    for r in rows:
        peak = f"{r['peak_mem_gb']:.1f}" if r["peak_mem_gb"] else "N/A"
        sps = f"{r['steps_per_sec']:.2f}" if r["steps_per_sec"] else "N/A"
        tps = f"{r['time_per_step']:.3f}" if r["time_per_step"] else "N/A"
        accum = str(r["accum_to_256"]) if r["accum_to_256"] else "N/A"
        print(
            f"{r['experiment']:>28} {r['compile_mode']:>16} {r['gc_layers']:>4} "
            f"{r['max_bs']:>7} {peak:>8} {sps:>8} {tps:>8} {accum:>10}"
        )
    print("=" * 100)

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
