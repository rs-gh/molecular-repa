"""Benchmark torch.compile modes on tabasco training throughput.

Each (experiment, compile_mode, precision) runs in a fresh subprocess on a
fake datamodule (so we isolate the compile question from I/O). Peak GPU
memory, steady-state steps/sec, and median/p90 step time are reported.

Tabasco's production config already uses `compile=true` with
`compile_mode=reduce-overhead` (flow_model.yaml), but these choices were
never measured. This script establishes the actual delta.

Usage:
    python benchmark_compile.py \\
        --experiments geom/mild \\
        --compile_modes off default reduce-overhead max-autotune \\
        --precisions 16 bf16-mixed \\
        --num_steps 100 --warmup 30 \\
        --output_csv evaluation/tabasco/bench/results/compile.csv
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


# Default BS per experiment (current production defaults, see configs/experiment/geom/mild.yaml)
DEFAULT_BS = {
    "geom/mild": 256,
    "geom/chemprop_tradeoff": 256,
    "geom/mace_cached_tradeoff": 256,
}

# REPA experiments need SMILES + lmdb_key non-tensor fields. We skip REPA in
# compile benches by default because the cached encoders need real LMDB data
# and we want to isolate the compile question to the flow model itself.
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
    p.add_argument(
        "--compile_modes",
        nargs="+",
        default=["off", "default", "reduce-overhead"],
        choices=["off", "default", "reduce-overhead", "max-autotune"],
    )
    p.add_argument(
        "--precisions",
        nargs="+",
        default=["16", "bf16-mixed"],
        help="Lightning precision strings: 16 | bf16-mixed | 32-true",
    )
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=WARMUP_STEPS_DEFAULT)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_atoms", type=int, default=GEOM_MAX_ATOMS)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument(
        "--output_csv",
        type=str,
        default="evaluation/tabasco/bench/results/compile.csv",
    )
    return p.parse_args()


def _run_one(
    result_queue, experiment, compile_mode, precision, batch_size, num_atoms, num_steps
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

        datamodule = build_fake_tabasco_datamodule(
            batch_size=batch_size,
            num_atoms=num_atoms,
            with_smiles=IS_REPA.get(experiment, False),
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
    for exp in args.experiments:
        bs = args.batch_size or DEFAULT_BS.get(exp, 256)
        for mode in args.compile_modes:
            for prec in args.precisions:
                print(
                    f"\n>>> exp={exp} compile={mode} precision={prec} "
                    f"bs={bs} atoms={args.num_atoms} steps={args.num_steps}"
                )
                clear_compile_cache()
                res = run_in_subprocess(
                    _run_one,
                    args=(exp, mode, prec, bs, args.num_atoms, args.num_steps),
                    timeout=args.timeout,
                )
                row = {
                    "experiment": exp,
                    "compile_mode": mode,
                    "precision": prec,
                    "batch_size": bs,
                    "num_atoms": args.num_atoms,
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
    print("TABASCO COMPILE BENCHMARK RESULTS")
    print("=" * 80)
    print_table(rows, drop_cols=["n_total", "n_warmup", "total_s", "p10_s", "p99_s"])

    write_csv(rows, args.output_csv)
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
