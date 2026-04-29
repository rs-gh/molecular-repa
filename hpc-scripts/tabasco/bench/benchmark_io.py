"""Benchmark tabasco LMDB dataloader throughput.

Pure I/O - no model, no GPU compute. Builds a real `LmdbDataModule` over
the tabasco LMDB and times successive `next()` calls on the train loader.

Tabasco's feedback memory says num_workers > 0 causes segfaults on this
cluster (see feedback_num_workers.md). This script **probes** that
assumption inside a subprocess: a crash is a valid, logged datapoint - not
a script-level failure.

Usage:
    python benchmark_io.py \\
        --dataset geom \\
        --num_workers 0 2 4 \\
        --pin_memory true false \\
        --num_batches 300 --warmup 20
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

from _bench_common import (  # noqa: E402
    TABASCO_PROJECT_ROOT,
    print_table,
    run_in_subprocess,
    write_csv,
)


DATASET_PATHS = {
    "geom": {
        "data_dir": f"{TABASCO_PROJECT_ROOT}/data/processed_geom_train.pt",
        "lmdb_dir": f"{TABASCO_PROJECT_ROOT}/data/lmdb_geom",
    },
    "qm9": {
        "data_dir": f"{TABASCO_PROJECT_ROOT}/data/processed_qm9_train.pt",
        "lmdb_dir": f"{TABASCO_PROJECT_ROOT}/data/lmdb_qm9",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="geom", choices=list(DATASET_PATHS.keys()))
    p.add_argument("--num_workers", type=int, nargs="+", default=[0])
    p.add_argument(
        "--pin_memory",
        nargs="+",
        default=["true", "false"],
        choices=["true", "false"],
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_batches", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument(
        "--output_csv",
        type=str,
        default="evaluation/tabasco/bench/results/io.csv",
    )
    return p.parse_args()


def _run_one(result_queue, dataset, batch_size, num_workers, pin_memory, num_batches):
    try:
        os.environ.setdefault("PROJECT_ROOT", TABASCO_PROJECT_ROOT)

        from torch.utils.data import DataLoader

        from tabasco.data.components.lmdb_unconditional import UnconditionalLMDBDataset
        from tabasco.data.utils import TensorDictCollator

        paths = DATASET_PATHS[dataset]

        ds = UnconditionalLMDBDataset(
            data_dir=paths["data_dir"],
            split="train",
            add_random_rotation=True,
            add_random_permutation=False,
            reorder_to_smiles_order=True,
            remove_hydrogens=True,
            lmdb_dir=paths["lmdb_dir"],
        )

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=TensorDictCollator(),
            pin_memory=pin_memory,
            shuffle=True,
        )

        times_s = []
        t_prev = time.perf_counter()
        it = iter(loader)
        for _ in range(num_batches):
            try:
                _ = next(it)
            except StopIteration:
                it = iter(loader)
                _ = next(it)
            t_now = time.perf_counter()
            times_s.append(t_now - t_prev)
            t_prev = t_now

        result_queue.put({"status": "ok", "times_s": times_s})
    except Exception as e:
        result_queue.put({"status": "error", "error": f"{e}\n{traceback.format_exc()}"})


def summarize_io(times_s, warmup, batch_size):
    measured = times_s[warmup:] if len(times_s) > warmup else times_s
    sm = sorted(measured)
    n = len(sm)
    median = statistics.median(sm)
    p90 = sm[min(n - 1, int(0.90 * n))]
    p99 = sm[min(n - 1, int(0.99 * n))]
    return {
        "n_total": len(times_s),
        "n_warmup": warmup,
        "n_measured": n,
        "median_s_per_batch": round(median, 4),
        "p90_s": round(p90, 4),
        "p99_s": round(p99, 4),
        "batches_per_sec": round(1.0 / median, 2),
        "samples_per_sec": round(batch_size / median, 1),
        "total_s": round(sum(times_s), 2),
    }


def main():
    args = parse_args()
    rows = []
    for nw in args.num_workers:
        for pm_str in args.pin_memory:
            pin_memory = pm_str == "true"
            print(
                f"\n>>> dataset={args.dataset} workers={nw} pin={pin_memory} "
                f"bs={args.batch_size} batches={args.num_batches}"
            )
            res = run_in_subprocess(
                _run_one,
                args=(
                    args.dataset,
                    args.batch_size,
                    nw,
                    pin_memory,
                    args.num_batches,
                ),
                timeout=args.timeout,
            )
            row = {
                "dataset": args.dataset,
                "num_workers": nw,
                "pin_memory": pin_memory,
                "batch_size": args.batch_size,
                "status": res["status"],
            }
            if res["status"] == "ok":
                row.update(summarize_io(res["times_s"], args.warmup, args.batch_size))
                print(
                    f"    median={row['median_s_per_batch']:.3f}s/batch  "
                    f"{row['samples_per_sec']:.1f} samples/s"
                )
            else:
                print(f"    {res['status'].upper()}: {res.get('error', '')[:200]}")
            rows.append(row)

    print("\n" + "=" * 80)
    print("TABASCO I/O BENCHMARK RESULTS")
    print("=" * 80)
    print_table(rows, drop_cols=["n_total", "n_warmup", "total_s"])

    write_csv(rows, args.output_csv)
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
