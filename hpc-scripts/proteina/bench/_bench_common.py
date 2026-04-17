"""Shared helpers for proteina performance benchmarks.

Each benchmark script runs individual variants in a fresh subprocess so
torch.compile cache, CUDA state, and peak-memory stats start clean.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


PROTEINA_CONFIG_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../src/proteina/configs",
    )
)
WARMUP_STEPS_DEFAULT = 30
DEFAULT_DATA_PATH = "/rds/user/sr2173/hpc-work/proteina/data"


@dataclass
class StepStats:
    """Steady-state per-step timing, skipping warmup."""

    n_total: int
    n_warmup: int
    n_measured: int
    median_s: float
    p10_s: float
    p90_s: float
    p99_s: float
    steps_per_sec: float
    total_s: float
    all_times_s: list = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_warmup": self.n_warmup,
            "n_measured": self.n_measured,
            "median_s_per_step": round(self.median_s, 4),
            "p10_s": round(self.p10_s, 4),
            "p90_s": round(self.p90_s, 4),
            "p99_s": round(self.p99_s, 4),
            "steps_per_sec": round(self.steps_per_sec, 3),
            "total_s": round(self.total_s, 2),
        }


def summarize(times_s: list, n_warmup: int) -> StepStats:
    n_total = len(times_s)
    measured = times_s[n_warmup:] if n_total > n_warmup else times_s
    if not measured:
        raise ValueError(
            f"No steady-state samples (n_total={n_total}, warmup={n_warmup})"
        )
    sorted_m = sorted(measured)
    n = len(sorted_m)
    median = statistics.median(sorted_m)
    p10 = sorted_m[max(0, int(0.10 * n) - 1)]
    p90 = sorted_m[min(n - 1, int(0.90 * n))]
    p99 = sorted_m[min(n - 1, int(0.99 * n))]
    return StepStats(
        n_total=n_total,
        n_warmup=n_warmup,
        n_measured=n,
        median_s=median,
        p10_s=p10,
        p90_s=p90,
        p99_s=p99,
        steps_per_sec=1.0 / median,
        total_s=sum(times_s),
        all_times_s=times_s,
    )


def build_step_timer_callback():
    """Return a Lightning callback that records wall time per training step.

    Synchronizes CUDA at step boundaries so async kernel launches are not
    counted toward the next step.
    """
    import lightning as L
    import torch

    class StepTimer(L.Callback):
        def __init__(self):
            self.times_s: list = []
            self._start: Optional[float] = None

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._start = time.perf_counter()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if self._start is not None:
                self.times_s.append(time.perf_counter() - self._start)
                self._start = None

    return StepTimer()


def build_fake_pdb_datamodule(max_len: int, batch_size: int, num_workers: int = 0):
    """Fake-data datamodule matching the real proteina pipeline's tensor shapes."""
    import torch
    from torch_geometric.data import Data, Dataset

    from proteinfoundation.datasets.base_data import BaseLightningDataModule
    from proteinfoundation.datasets.transforms import (
        ChainBreakPerResidueTransform,
        GlobalRotationTransform,
        PaddingTransform,
    )

    class FakePDBDataset(Dataset):
        def __init__(self, size, transform=None):
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
        def __init__(self):
            super().__init__(
                batch_padding=True,
                sampling_mode="random",
                transforms=[
                    GlobalRotationTransform(),
                    ChainBreakPerResidueTransform(),
                    PaddingTransform(max_size=max_len),
                ],
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )

        def _get_dataset(self, split):
            size = 6 if split == "val" else max(batch_size * 8, 200)
            return FakePDBDataset(size, transform=self.transform)

    return FakePDBDataModule()


def load_proteina_cfg(config_name: str = "training_ca"):
    """Load a proteina experiment config via Hydra from the bench scaffold."""
    import hydra

    config_dir = os.path.join(PROTEINA_CONFIG_ROOT, "experiment_config")
    with hydra.initialize_config_dir(config_dir, version_base=hydra.__version__):
        cfg = hydra.compose(config_name=config_name)
    return cfg


def apply_gradient_checkpointing(model, gc_layers: int):
    """Checkpoint the last N transformer layers (deeper = larger activations)."""
    from torch.utils.checkpoint import checkpoint as ckpt_fn

    if gc_layers <= 0:
        return
    if not (hasattr(model, "nn") and hasattr(model.nn, "layers")):
        raise RuntimeError("Model has no nn.layers for gradient checkpointing")
    n_total = len(model.nn.layers)
    n_ckpt = min(gc_layers, n_total)
    start_idx = n_total - n_ckpt
    for i in range(start_idx, n_total):
        layer = model.nn.layers[i]
        orig_forward = layer.forward

        def make_ckpt_forward(fn):
            def ckpt_forward(*args, **kwargs):
                return ckpt_fn(fn, *args, use_reentrant=False, **kwargs)

            return ckpt_forward

        layer.forward = make_ckpt_forward(orig_forward)


def clear_compile_cache():
    """Wipe torchinductor cache so each compile run starts fresh."""
    import shutil

    cache_dir = f"/tmp/torchinductor_{os.environ.get('USER', 'unknown')}"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)


def run_in_subprocess(
    target: Callable[..., Any],
    args: tuple,
    timeout: int = 900,
) -> dict:
    """Run `target` in a spawn subprocess, return result dict from its queue.

    `target` must accept `(result_queue, *args)` and put one dict into the queue.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(result_queue, *args))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"status": "timeout"}
    try:
        return result_queue.get_nowait()
    except Exception:
        if proc.exitcode != 0:
            return {"status": "oom_or_crash", "exitcode": proc.exitcode}
        return {"status": "error", "error": "no result from subprocess"}


def write_csv(rows: list, path: str) -> None:
    import csv

    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys: list = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def print_table(rows: list, drop_cols: Optional[list] = None) -> None:
    if not rows:
        print("(no rows)")
        return
    drop = set(drop_cols or [])
    keys = [k for k in rows[0].keys() if k not in drop]
    widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    header = "  ".join(k.rjust(widths[k]) for k in keys)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(k, "")).rjust(widths[k]) for k in keys))
