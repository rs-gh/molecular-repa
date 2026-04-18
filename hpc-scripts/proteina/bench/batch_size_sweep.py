"""Batch size sweep for proteina training on A100 80GB.

Binary-searches for the maximum batch size before OOM for each
(seq_len, model_type, compile) combination. Each test runs in a
subprocess so OOM kills the child cleanly.

Usage:
    python hpc-scripts/proteina/bench/batch_size_sweep.py \
        --seq_lens 128 256 512 \
        --model_types baseline repa \
        --min_bs 4 --max_bs 128 \
        --num_steps 20 --timeout 300 \
        --output_csv batch_size_results.csv
"""

import argparse
import csv
import math
import multiprocessing as mp
import os
import shutil
import tempfile
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Batch size sweep for proteina")
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument(
        "--model_types",
        nargs="+",
        default=["baseline", "repa"],
        choices=["baseline", "repa"],
    )
    parser.add_argument("--min_bs", type=int, default=4)
    parser.add_argument("--max_bs", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=300, help="Per-test timeout (s)")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument(
        "--grad_checkpoint",
        action="store_true",
        default=False,
        help="Test with gradient checkpointing (recompute activations to save memory)",
    )
    parser.add_argument(
        "--gc_layers",
        type=int,
        default=None,
        help="Number of layers to checkpoint (default: all 10). "
        "Checkpoints the LAST N layers (deepest layers use most memory).",
    )
    parser.add_argument(
        "--all_variants",
        action="store_true",
        default=False,
        help="Test all combinations: compile × gc_layers variants",
    )
    parser.add_argument("--output_csv", type=str, default="batch_size_results.csv")
    return parser.parse_args()


def run_single_test(
    seq_len, batch_size, model_type, compile_flag, gc_layers, num_steps, result_queue
):
    """Run a single batch size test in a child process.

    Args:
        gc_layers: Number of transformer layers to gradient-checkpoint (0=none,
            10=all). Checkpoints the last N layers since deeper layers have
            larger activations.

    Imports are inside this function so each subprocess gets a fresh state.
    Results are sent back via multiprocessing.Queue.
    """
    try:
        os.environ.setdefault("DATA_PATH", "/rds/user/sr2173/hpc-work/proteina/data")

        import proteinfoundation.repa.pyg_compat  # noqa: F401

        import hydra
        import lightning as L
        import torch
        from omegaconf import OmegaConf, open_dict
        from torch_geometric.data import Data, Dataset

        from proteinfoundation.datasets.base_data import BaseLightningDataModule
        from proteinfoundation.proteinflow.proteina import Proteina
        from proteinfoundation.repa.proteina_repa import ProteinaREPA

        # --- Fake dataset (from smoke_test_fake_data.sh pattern) ---
        class FakePDBDataset(Dataset):
            def __init__(self, size, max_len, transform=None):
                super().__init__(transform=transform)
                self._size = size
                self._template = Data(
                    coords=torch.randn(max_len, 37, 3),
                    coord_mask=torch.ones(max_len, 37, dtype=torch.bool),
                    seq=torch.randint(0, 20, (max_len,)),
                    res_seq_pdb_idx=torch.arange(max_len, dtype=torch.long),
                    chain_break_per_res=torch.zeros(max_len, dtype=torch.long),
                    num_nodes=max_len,
                )

            def len(self):
                return self._size

            def get(self, idx):
                return self._template.clone()

        class FakePDBDataModule(BaseLightningDataModule):
            def __init__(self, max_len, batch_size, transforms=None):
                super().__init__(
                    batch_padding=True,
                    sampling_mode="random",
                    transforms=transforms,
                    batch_size=batch_size,
                    num_workers=0,
                    pin_memory=True,
                )
                self._max_len = max_len

            def _get_dataset(self, split):
                size = 6 if split == "val" else max(batch_size * 4, 100)
                return FakePDBDataset(size, self._max_len, transform=self.transform)

        # --- Load base config ---
        config_path = os.path.join(
            os.path.dirname(__file__),
            "../../../src/proteina/configs/experiment_config",
        )
        with hydra.initialize_config_dir(
            os.path.abspath(config_path), version_base=hydra.__version__
        ):
            cfg_exp = hydra.compose(config_name="training_ca")

        cfg_exp.hardware.ngpus_per_node_ = 1
        cfg_exp.hardware.nnodes_ = 1
        with open_dict(cfg_exp):
            cfg_exp.compile = compile_flag
        cfg_exp.run_name_ = f"bs_sweep_{seq_len}_{model_type}_{batch_size}"

        # --- Load transforms ---
        # Build PaddingTransform for the target seq_len
        from proteinfoundation.datasets.transforms import (
            ChainBreakPerResidueTransform,
            GlobalRotationTransform,
            PaddingTransform,
        )

        transforms = [
            GlobalRotationTransform(),
            ChainBreakPerResidueTransform(),
            PaddingTransform(max_size=seq_len),
        ]

        # --- Inject REPA config if needed ---
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
                "projector_num_layers": 2,
            }
            with open_dict(cfg_exp):
                OmegaConf.update(cfg_exp, "repa", repa_cfg, force_add=True)

        # --- Create model ---
        tmp_dir = tempfile.mkdtemp(prefix="bs_sweep_")
        try:
            torch.set_float32_matmul_precision("medium")
            L.seed_everything(42)

            if model_type == "repa":
                model = ProteinaREPA(cfg_exp, store_dir=tmp_dir)
            else:
                model = Proteina(cfg_exp, store_dir=tmp_dir)

            # --- Apply gradient checkpointing ---
            if gc_layers > 0 and hasattr(model, "nn") and hasattr(model.nn, "layers"):
                from torch.utils.checkpoint import checkpoint as ckpt_fn

                n_total = len(model.nn.layers)
                n_ckpt = min(gc_layers, n_total)
                # Checkpoint the LAST n_ckpt layers (deeper = larger activations)
                start_idx = n_total - n_ckpt

                for i in range(start_idx, n_total):
                    layer = model.nn.layers[i]
                    orig_forward = layer.forward

                    def make_ckpt_forward(fn):
                        def ckpt_forward(*args, **kwargs):
                            return ckpt_fn(fn, *args, use_reentrant=False, **kwargs)

                        return ckpt_forward

                    layer.forward = make_ckpt_forward(orig_forward)

                print(
                    f"    Gradient checkpointing: layers {start_idx}-{n_total-1} "
                    f"({n_ckpt}/{n_total})",
                    flush=True,
                )

            datamodule = FakePDBDataModule(
                max_len=seq_len,
                batch_size=batch_size,
                transforms=transforms,
            )

            trainer = L.Trainer(
                max_steps=num_steps,
                accelerator="gpu",
                devices=1,
                num_nodes=1,
                callbacks=[],
                logger=False,
                log_every_n_steps=1,
                enable_progress_bar=False,
                enable_checkpointing=False,
                check_val_every_n_epoch=None,
                val_check_interval=999999,
                strategy="auto",
                precision="bf16-mixed",
                gradient_clip_algorithm="norm",
                gradient_clip_val=1.0,
            )

            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            trainer.fit(model, datamodule)
            elapsed = time.time() - start

            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            steps_per_sec = num_steps / elapsed
            time_per_step = elapsed / num_steps

            result_queue.put(
                {
                    "status": "ok",
                    "peak_mem_gb": round(peak_mem_gb, 2),
                    "steps_per_sec": round(steps_per_sec, 3),
                    "time_per_step": round(time_per_step, 3),
                }
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            result_queue.put({"status": "oom"})
        else:
            result_queue.put({"status": "error", "error": str(e)})
    except Exception as e:
        result_queue.put({"status": "error", "error": str(e)})


def test_batch_size(
    seq_len, batch_size, model_type, compile_flag, gc_layers, num_steps, timeout
):
    """Run a single test in a subprocess, return result dict."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=run_single_test,
        args=(
            seq_len,
            batch_size,
            model_type,
            compile_flag,
            gc_layers,
            num_steps,
            result_queue,
        ),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"status": "timeout"}

    if proc.exitcode != 0:
        # Likely OOM killed by OS
        try:
            return result_queue.get_nowait()
        except Exception:
            return {"status": "oom"}

    try:
        return result_queue.get_nowait()
    except Exception:
        return {"status": "error", "error": "no result from subprocess"}


def binary_search_max_bs(
    seq_len, model_type, compile_flag, gc_layers, lo, hi, num_steps, timeout
):
    """Binary search for the maximum batch size that doesn't OOM."""
    best_result = None
    best_bs = lo - 1

    gc_str = f"  gc_layers={gc_layers}/10" if gc_layers > 0 else ""
    print(f"\n{'='*70}")
    print(f"  seq_len={seq_len}  model={model_type}  compile={compile_flag}{gc_str}")
    print(f"  Searching BS in [{lo}, {hi}]")
    print(f"{'='*70}")

    while lo <= hi:
        mid = (lo + hi) // 2
        print(f"\n  Testing BS={mid}...", end=" ", flush=True)

        # Clear torch compile cache between tests
        cache_dir = f"/tmp/torchinductor_{os.environ.get('USER', 'unknown')}"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)

        result = test_batch_size(
            seq_len, mid, model_type, compile_flag, gc_layers, num_steps, timeout
        )

        if result["status"] == "ok":
            print(
                f"OK  (peak={result['peak_mem_gb']:.1f}GB, "
                f"{result['steps_per_sec']:.2f} steps/s, "
                f"{result['time_per_step']:.3f}s/step)"
            )
            best_bs = mid
            best_result = result
            lo = mid + 1
        else:
            reason = result.get("error", result["status"])
            print(f"FAIL ({reason})")
            hi = mid - 1

    return best_bs, best_result


def main():
    args = parse_args()

    # Build list of (compile, gc_layers) variants to test
    # gc_layers=0 means no checkpointing, 5=half, 10=all
    if args.all_variants:
        variants = [
            (True, 0),  # compile, no checkpointing
            (True, 5),  # compile, checkpoint last 5 layers
            (True, 10),  # compile, checkpoint all 10 layers
        ]
    elif args.grad_checkpoint:
        gc = args.gc_layers if args.gc_layers is not None else 10
        variants = [(args.compile, gc)]
    else:
        variants = [(args.compile, 0)]

    print("=" * 70)
    print("  PROTEINA BATCH SIZE SWEEP")
    print("  GPU: A100 80GB")
    print(f"  Seq lengths: {args.seq_lens}")
    print(f"  Model types: {args.model_types}")
    print(f"  BS range: [{args.min_bs}, {args.max_bs}]")
    print(f"  Steps per test: {args.num_steps}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Variants: {variants}")
    print("=" * 70)

    results = []

    for seq_len in args.seq_lens:
        for model_type in args.model_types:
            for compile_flag, gc_layers in variants:
                max_bs, best = binary_search_max_bs(
                    seq_len,
                    model_type,
                    compile_flag,
                    gc_layers,
                    args.min_bs,
                    args.max_bs,
                    args.num_steps,
                    args.timeout,
                )

                row = {
                    "seq_len": seq_len,
                    "model_type": model_type,
                    "compile": compile_flag,
                    "gc_layers": gc_layers,
                    "max_bs": max_bs,
                    "peak_mem_gb": best["peak_mem_gb"] if best else None,
                    "steps_per_sec": best["steps_per_sec"] if best else None,
                    "time_per_step": best["time_per_step"] if best else None,
                    "eff_bs_256_accum": math.ceil(256 / max_bs) if max_bs > 0 else None,
                }
                results.append(row)

    # --- Print summary table ---
    print("\n\n" + "=" * 100)
    print("  RESULTS SUMMARY")
    print("=" * 100)
    header = (
        f"{'seq_len':>8} {'model':>10} {'compile':>8} {'gc_layers':>10} "
        f"{'max_bs':>7} {'peak_GB':>8} {'steps/s':>8} {'s/step':>8} {'accum→256':>10}"
    )
    print(header)
    print("-" * 100)
    for r in results:
        peak = f"{r['peak_mem_gb']:.1f}" if r["peak_mem_gb"] else "N/A"
        sps = f"{r['steps_per_sec']:.2f}" if r["steps_per_sec"] else "N/A"
        tps = f"{r['time_per_step']:.3f}" if r["time_per_step"] else "N/A"
        accum = str(r["eff_bs_256_accum"]) if r["eff_bs_256_accum"] else "N/A"
        gc = f"{r['gc_layers']}/10" if r["gc_layers"] > 0 else "none"
        print(
            f"{r['seq_len']:>8} {r['model_type']:>10} {str(r['compile']):>8} "
            f"{gc:>10} "
            f"{r['max_bs']:>7} {peak:>8} {sps:>8} {tps:>8} {accum:>10}"
        )
    print("=" * 100)

    # --- Write CSV ---
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
