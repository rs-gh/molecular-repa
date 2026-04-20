"""Shared helpers for tabasco performance benchmarks.

Mirrors the structure of hpc-scripts/proteina/bench/_bench_common.py: each
variant runs in a fresh spawn subprocess so torch.compile cache, CUDA
state, and peak-memory stats start clean. Warmup steps are dropped before
computing steady-state throughput.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


TABASCO_CONFIG_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../src/tabasco/configs",
    )
)
TABASCO_PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../src/tabasco",
    )
)
WARMUP_STEPS_DEFAULT = 30

# GEOM drugs tensor shapes (from src/tabasco/data/lmdb_geom/train_stats.yaml)
GEOM_MAX_ATOMS = 71
GEOM_ATOM_DIM = 9
GEOM_SPATIAL_DIM = 3


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
    """Lightning callback that records wall time per training step.

    Synchronises CUDA at step boundaries so async kernel launches on the
    current step are not charged to the next step.
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


def _ensure_project_root_env() -> None:
    """Hydra paths config reads PROJECT_ROOT via oc.env. Make sure it's set."""
    os.environ.setdefault("PROJECT_ROOT", TABASCO_PROJECT_ROOT)


def load_tabasco_cfg(experiment: str, extra_overrides: Optional[List[str]] = None):
    """Compose tabasco's Hydra train config with the given experiment override.

    Args:
        experiment: e.g. "geom/mild", "geom/chemprop_tradeoff", "geom/mace_cached_tradeoff".
        extra_overrides: extra hydra overrides, e.g. ["model.compile=false"].
    """
    import hydra

    _ensure_project_root_env()
    overrides = [f"experiment={experiment}"]
    if extra_overrides:
        overrides.extend(extra_overrides)
    with hydra.initialize_config_dir(TABASCO_CONFIG_ROOT, version_base="1.3"):
        cfg = hydra.compose(config_name="train", overrides=overrides)
    return cfg


def _make_fake_sample(num_atoms: int, real_atoms: int, smiles: Optional[str]):
    """Build a single TensorDict matching UnconditionalLMDBDataset output shape."""
    import torch
    from tensordict import TensorDict

    coords = torch.randn(num_atoms, GEOM_SPATIAL_DIM)
    atomics = torch.zeros(num_atoms, GEOM_ATOM_DIM)
    atom_types = torch.randint(0, GEOM_ATOM_DIM - 1, (real_atoms,))
    atomics[torch.arange(real_atoms), atom_types] = 1.0
    atomics[real_atoms:, GEOM_ATOM_DIM - 1] = 1.0  # "*" dummy
    padding_mask = torch.zeros(num_atoms, dtype=torch.bool)
    padding_mask[real_atoms:] = True

    td = TensorDict(
        {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
        batch_size=[],
    )
    if smiles is not None:
        td.set_non_tensor("smiles", smiles)
        td.set_non_tensor("lmdb_key", f"fake_{real_atoms}")
    return td


def build_fake_tabasco_datamodule(
    batch_size: int,
    num_atoms: int = GEOM_MAX_ATOMS,
    mean_real_atoms: int = 40,
    with_smiles: bool = False,
    num_workers: int = 0,
    dataset_size: Optional[int] = None,
):
    """Fake datamodule producing TensorDict batches matching the real LMDB output.

    Args:
        batch_size: samples per batch.
        num_atoms: padded sequence length (sets N in coords/atomics/padding_mask).
        mean_real_atoms: synthetic "real" atom count per sample (below padding).
        with_smiles: attach a dummy SMILES + lmdb_key non-tensor field per sample.
            Required when benchmarking REPA variants; harmless otherwise.
        num_workers: DataLoader worker count.
        dataset_size: optional override; defaults to ~batch_size * 8 steps of data.
    """
    import lightning as L
    from torch.utils.data import DataLoader, Dataset

    from tabasco.data.utils import TensorDictCollator

    class FakeTabascoDataset(Dataset):
        def __init__(self, size: int):
            self._size = size
            self._real_atoms = max(2, min(num_atoms - 1, mean_real_atoms))
            self._smiles = "CCO" if with_smiles else None

        def __len__(self):
            return self._size

        def __getitem__(self, idx):
            return _make_fake_sample(num_atoms, self._real_atoms, self._smiles)

    class FakeTabascoDataModule(L.LightningDataModule):
        def setup(self, stage: Optional[str] = None):
            size = dataset_size or max(batch_size * 8, 200)
            self.train_set = FakeTabascoDataset(size)
            self.val_set = FakeTabascoDataset(max(batch_size, 16))

        def train_dataloader(self):
            return DataLoader(
                self.train_set,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=TensorDictCollator(),
                shuffle=False,
            )

        def val_dataloader(self):
            return DataLoader(
                self.val_set,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=TensorDictCollator(),
                shuffle=False,
            )

    return FakeTabascoDataModule()


def apply_gradient_checkpointing(lightning_module, gc_layers: int) -> None:
    """Wrap the last `gc_layers` transformer blocks in torch.utils.checkpoint.

    Tabasco's transformer lives at:
        LightningTabasco.model           -> FlowMatchingModel
        FlowMatchingModel.net            -> TransformerModule (possibly torch.compile wrapped)
        TransformerModule.transformer    -> Transformer (reimplemented) or nn.TransformerEncoder
        .layers                          -> ModuleList of per-block modules

    Must be called BEFORE torch.compile to avoid compile wrapping hiding `.layers`.
    """
    from torch.utils.checkpoint import checkpoint as ckpt_fn

    if gc_layers <= 0:
        return
    model = (
        lightning_module.model
        if hasattr(lightning_module, "model")
        else lightning_module
    )
    net = model.net
    if hasattr(net, "_orig_mod"):
        net = net._orig_mod  # unwrap OptimizedModule if compiled
    transformer = net.transformer
    layers = (
        transformer.layers
    )  # both nn.TransformerEncoder and reimplemented.Transformer expose this
    n_total = len(layers)
    n_ckpt = min(gc_layers, n_total)
    start = n_total - n_ckpt
    for i in range(start, n_total):
        layer = layers[i]
        orig_forward = layer.forward

        def make_ckpt_forward(fn):
            def ckpt_forward(*args, **kwargs):
                return ckpt_fn(fn, *args, use_reentrant=False, **kwargs)

            return ckpt_forward

        layer.forward = make_ckpt_forward(orig_forward)


def clear_compile_cache() -> None:
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
    """Run `target` in a spawn subprocess, return result dict via its queue.

    `target` must accept `(result_queue, *args)` and `put` one dict.
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
