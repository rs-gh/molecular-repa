"""End-to-end training benchmark: does the compile / I/O / grad-ckpt
story translate when the model meets real data?

Runs short training with the real LMDB dataloader. One subprocess per
(compile_mode, lmdb_source, gc_layers) variant. Reports steady-state
steps/sec after dropping warmup.

Usage:
    python benchmark_e2e.py \\
        --compile_modes off default \\
        --lmdb_sources lustre nvme \\
        --gc_layers 0 \\
        --num_steps 100 --warmup 30
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

from _bench_common import (  # noqa: E402
    DEFAULT_DATA_PATH,
    WARMUP_STEPS_DEFAULT,
    apply_gradient_checkpointing,
    build_step_timer_callback,
    clear_compile_cache,
    load_proteina_cfg,
    print_table,
    run_in_subprocess,
    summarize,
    write_csv,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--compile_modes",
        nargs="+",
        default=["off", "default"],
        choices=["off", "default", "reduce-overhead", "max-autotune"],
    )
    p.add_argument(
        "--lmdb_sources",
        nargs="+",
        default=["lustre", "nvme"],
        choices=["lustre", "nvme"],
    )
    p.add_argument("--gc_layers", type=int, nargs="+", default=[0])
    p.add_argument("--model_type", default="baseline", choices=["baseline", "repa"])
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--max_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=WARMUP_STEPS_DEFAULT)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument(
        "--lustre_lmdb_dir", type=str, default=f"{DEFAULT_DATA_PATH}/pdb_train/lmdb"
    )
    p.add_argument("--nvme_lmdb_dir", type=str, default="/tmp/proteina_pdb_lmdb")
    p.add_argument(
        "--output_csv",
        type=str,
        default="evaluation/proteina/bench/results/e2e.csv",
    )
    return p.parse_args()


def ensure_nvme_copy(lustre_dir: str, nvme_dir: str) -> str:
    if os.path.isdir(nvme_dir):
        existing = [f for f in os.listdir(nvme_dir) if f.endswith(".lmdb")]
        if existing:
            return nvme_dir
    size_kb = int(subprocess.check_output(["du", "-sk", lustre_dir]).split()[0])
    free_kb = int(
        subprocess.check_output(["df", "--output=avail", "/tmp"])
        .splitlines()[-1]
        .strip()
    )
    if free_kb <= size_kb + 1_048_576:
        raise RuntimeError(
            f"Not enough /tmp space: {size_kb//1024}MB needed, "
            f"{free_kb//1024}MB free"
        )
    os.makedirs(nvme_dir, exist_ok=True)
    for split in ["val", "test", "train"]:
        src = os.path.join(lustre_dir, f"{split}.lmdb")
        if os.path.isfile(src):
            dst = os.path.join(nvme_dir, f"{split}.lmdb")
            t0 = time.perf_counter()
            shutil.copy(src, dst)
            dt = time.perf_counter() - t0
            sz_gb = os.path.getsize(dst) / (1024**3)
            print(f"    Copied {split}: {sz_gb:.2f} GB in {dt:.1f}s")
    return nvme_dir


def _run_one(
    result_queue,
    lmdb_dir,
    compile_mode,
    gc_layers,
    model_type,
    batch_size,
    max_size,
    num_workers,
    num_steps,
):
    try:
        # spawn start method for worker-based dataloader
        import torch.multiprocessing as tmp

        tmp.set_start_method("spawn", force=True)

        os.environ["LMDB_DIR"] = lmdb_dir
        os.environ.setdefault("DATA_PATH", DEFAULT_DATA_PATH)

        import proteinfoundation.repa.pyg_compat  # noqa: F401

        import hydra
        import lightning as L
        import torch
        from omegaconf import OmegaConf

        from proteinfoundation.proteinflow.proteina import Proteina
        from proteinfoundation.repa.proteina_repa import ProteinaREPA

        # --- Model config ---
        cfg = load_proteina_cfg("training_ca")
        cfg.hardware.ngpus_per_node_ = 1
        cfg.hardware.nnodes_ = 1
        cfg.run_name_ = f"bench_e2e_{model_type}_{compile_mode}_gc{gc_layers}"

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

        # --- Data config (instantiate real LMDB datamodule via hydra) ---
        data_cfg_path = os.path.join(
            os.path.dirname(__file__),
            "../../../src/proteina/configs/datasets_config/pdb",
        )
        with hydra.initialize_config_dir(
            os.path.abspath(data_cfg_path),
            version_base=hydra.__version__,
        ):
            cfg_data = hydra.compose(config_name="pdb_lmdb")

        cfg_data.datamodule.batch_size = batch_size
        cfg_data.datamodule.num_workers = num_workers
        for t in cfg_data.datamodule.transforms:
            if t.get("_target_", "").endswith("PaddingTransform"):
                t["max_size"] = max_size

        torch.set_float32_matmul_precision("medium")
        L.seed_everything(42)

        datamodule = hydra.utils.instantiate(cfg_data.datamodule)

        tmp_dir = tempfile.mkdtemp(prefix="bench_e2e_")
        try:
            model_cls = ProteinaREPA if model_type == "repa" else Proteina
            model = model_cls(cfg, store_dir=tmp_dir)

            if gc_layers > 0:
                apply_gradient_checkpointing(model, gc_layers)

            if compile_mode != "off":
                model.nn = torch.compile(model.nn, mode=compile_mode)

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
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result_queue.put({"status": "oom", "error": str(e)})
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

    # Prepare NVMe source if requested. If copy fails (e.g., /tmp too small on
    # this node) drop `nvme` from the source list and continue with Lustre.
    source_paths = {}
    lmdb_sources = list(args.lmdb_sources)
    if "lustre" in lmdb_sources:
        source_paths["lustre"] = args.lustre_lmdb_dir
    if "nvme" in lmdb_sources:
        print(f"Preparing NVMe copy at {args.nvme_lmdb_dir}...")
        try:
            source_paths["nvme"] = ensure_nvme_copy(
                args.lustre_lmdb_dir,
                args.nvme_lmdb_dir,
            )
        except Exception as e:
            print(f"!! NVMe copy failed ({e}); skipping nvme source.")
            lmdb_sources = [s for s in lmdb_sources if s != "nvme"]
    args.lmdb_sources = lmdb_sources
    if not lmdb_sources:
        print("No sources available — aborting.")
        return

    rows = []
    for mode in args.compile_modes:
        for source in args.lmdb_sources:
            for gc in args.gc_layers:
                print(
                    f"\n>>> compile={mode} source={source} gc={gc} "
                    f"bs={args.batch_size} steps={args.num_steps}"
                )
                clear_compile_cache()
                res = run_in_subprocess(
                    _run_one,
                    args=(
                        source_paths[source],
                        mode,
                        gc,
                        args.model_type,
                        args.batch_size,
                        args.max_size,
                        args.num_workers,
                        args.num_steps,
                    ),
                    timeout=args.timeout,
                )
                row = {
                    "model_type": args.model_type,
                    "compile_mode": mode,
                    "lmdb_source": source,
                    "gc_layers": gc,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
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
                    print(f"    FAILED: {res.get('error', res['status'])[:200]}")
                rows.append(row)

    print("\n" + "=" * 80)
    print("E2E BENCHMARK RESULTS")
    print("=" * 80)
    print_table(rows, drop_cols=["n_total", "n_warmup", "total_s", "p10_s", "p99_s"])

    write_csv(rows, args.output_csv)
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
