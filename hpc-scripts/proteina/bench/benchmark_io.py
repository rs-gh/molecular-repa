"""Benchmark LMDB DataLoader throughput: Lustre vs local NVMe.

Pure I/O - no model, no GPU compute. Builds the real
PDBLightningDataModule against a given LMDB directory, iterates
train_dataloader() for N batches, and reports per-batch latency.

The NVMe variant copies the LMDB to /tmp/proteina_pdb_lmdb first, matching
what hpc-scripts/proteina/training/pdb/train_baseline.sh does at job start.

Usage:
    python benchmark_io.py \\
        --sources lustre nvme \\
        --num_workers 0 2 4 8 \\
        --pin_memory true false \\
        --num_batches 300 --warmup 20
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

from _bench_common import (  # noqa: E402
    DEFAULT_DATA_PATH,
    print_table,
    run_in_subprocess,
    write_csv,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sources", nargs="+", default=["lustre", "nvme"], choices=["lustre", "nvme"]
    )
    p.add_argument("--num_workers", type=int, nargs="+", default=[0, 2, 4])
    p.add_argument(
        "--pin_memory", nargs="+", default=["true", "false"], choices=["true", "false"]
    )
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--max_size", type=int, default=512)
    p.add_argument("--num_batches", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument(
        "--lustre_lmdb_dir", type=str, default=f"{DEFAULT_DATA_PATH}/pdb_train/lmdb"
    )
    p.add_argument("--nvme_lmdb_dir", type=str, default="/tmp/proteina_pdb_lmdb")
    p.add_argument(
        "--output_csv",
        type=str,
        default="evaluation/proteina/bench/results/io.csv",
    )
    return p.parse_args()


def ensure_nvme_copy(lustre_dir: str, nvme_dir: str) -> str:
    """Copy LMDB splits from Lustre to NVMe. Returns the NVMe dir on success."""
    if not os.path.isdir(lustre_dir):
        raise FileNotFoundError(f"Lustre LMDB not found: {lustre_dir}")

    # Skip copy if already present and non-empty
    if os.path.isdir(nvme_dir):
        existing = [f for f in os.listdir(nvme_dir) if f.endswith(".lmdb")]
        if existing:
            print(f"    NVMe already populated at {nvme_dir}: {existing}")
            return nvme_dir

    size_kb = int(subprocess.check_output(["du", "-sk", lustre_dir]).split()[0])
    free_kb = int(
        subprocess.check_output(["df", "--output=avail", "/tmp"])
        .splitlines()[-1]
        .strip()
    )
    if free_kb <= size_kb + 1_048_576:
        raise RuntimeError(
            f"Not enough /tmp space: need {size_kb//1024}MB + 1GB, "
            f"have {free_kb//1024}MB"
        )
    os.makedirs(nvme_dir, exist_ok=True)
    for split in ["val", "test", "train"]:
        src = os.path.join(lustre_dir, f"{split}.lmdb")
        if os.path.isfile(src):
            dst = os.path.join(nvme_dir, f"{split}.lmdb")
            print(f"    Copying {split}.lmdb ...")
            t0 = time.perf_counter()
            shutil.copy(src, dst)
            dt = time.perf_counter() - t0
            sz_gb = os.path.getsize(dst) / (1024**3)
            print(f"    Done: {sz_gb:.2f} GB in {dt:.1f}s ({sz_gb / dt:.2f} GB/s)")
    return nvme_dir


def _run_one(
    result_queue, lmdb_dir, batch_size, max_size, num_workers, pin_memory, num_batches
):
    try:
        os.environ["LMDB_DIR"] = lmdb_dir
        os.environ.setdefault("DATA_PATH", DEFAULT_DATA_PATH)

        import proteinfoundation.repa.pyg_compat  # noqa: F401
        from proteinfoundation.datasets.pdb_data import PDBLightningDataModule
        from proteinfoundation.datasets.transforms import (
            ChainBreakPerResidueTransform,
            GlobalRotationTransform,
            PaddingTransform,
        )

        transforms = [
            GlobalRotationTransform(),
            ChainBreakPerResidueTransform(),
            PaddingTransform(max_size=max_size),
        ]

        dm = PDBLightningDataModule(
            data_dir=f"{os.environ['DATA_PATH']}/pdb_train/",
            lmdb_dir=lmdb_dir,
            batch_padding=True,
            sampling_mode="random",
            transforms=transforms,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        dm.setup("fit")
        loader = dm.train_dataloader()

        # Time wall-clock between successive batches arriving from the loader.
        times_s = []
        t_prev = time.perf_counter()
        it = iter(loader)
        # First __next__() spins up workers - count it in times but drop in warmup.
        for i in range(num_batches):
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
        result_queue.put(
            {
                "status": "error",
                "error": f"{e}\n{traceback.format_exc()}",
            }
        )


def summarize_io(times_s, warmup, batch_size):
    measured = times_s[warmup:] if len(times_s) > warmup else times_s
    sm = sorted(measured)
    n = len(sm)
    median = statistics.median(sm)
    p50 = median
    p90 = sm[min(n - 1, int(0.90 * n))]
    p99 = sm[min(n - 1, int(0.99 * n))]
    return {
        "n_total": len(times_s),
        "n_warmup": warmup,
        "n_measured": n,
        "median_s_per_batch": round(median, 4),
        "p50_s": round(p50, 4),
        "p90_s": round(p90, 4),
        "p99_s": round(p99, 4),
        "batches_per_sec": round(1.0 / median, 2),
        "samples_per_sec": round(batch_size / median, 1),
        "total_s": round(sum(times_s), 2),
    }


def main():
    args = parse_args()

    # Prepare NVMe source if requested. If copy fails (e.g., /tmp too small on
    # this node) drop `nvme` from the source list and continue with Lustre so
    # we still get useful results rather than dying outright.
    source_paths = {}
    sources = list(args.sources)
    if "lustre" in sources:
        source_paths["lustre"] = args.lustre_lmdb_dir
    if "nvme" in sources:
        print(f"Preparing NVMe copy at {args.nvme_lmdb_dir}...")
        try:
            source_paths["nvme"] = ensure_nvme_copy(
                args.lustre_lmdb_dir,
                args.nvme_lmdb_dir,
            )
        except Exception as e:
            print(f"!! NVMe copy failed ({e}); skipping nvme source.")
            sources = [s for s in sources if s != "nvme"]
    args.sources = sources
    if not sources:
        print("No sources available - aborting.")
        return

    rows = []
    for source in args.sources:
        lmdb_dir = source_paths[source]
        for nw in args.num_workers:
            for pm_str in args.pin_memory:
                pin_memory = pm_str == "true"
                print(
                    f"\n>>> source={source} workers={nw} pin={pin_memory} "
                    f"bs={args.batch_size} batches={args.num_batches}"
                )
                res = run_in_subprocess(
                    _run_one,
                    args=(
                        lmdb_dir,
                        args.batch_size,
                        args.max_size,
                        nw,
                        pin_memory,
                        args.num_batches,
                    ),
                    timeout=args.timeout,
                )
                row = {
                    "source": source,
                    "num_workers": nw,
                    "pin_memory": pin_memory,
                    "batch_size": args.batch_size,
                    "max_size": args.max_size,
                    "status": res["status"],
                }
                if res["status"] == "ok":
                    row.update(
                        summarize_io(
                            res["times_s"],
                            args.warmup,
                            args.batch_size,
                        )
                    )
                    print(
                        f"    median={row['median_s_per_batch']:.3f}s/batch  "
                        f"{row['samples_per_sec']:.1f} samples/s"
                    )
                else:
                    print(f"    FAILED: {res.get('error', res['status'])[:200]}")
                rows.append(row)

    print("\n" + "=" * 80)
    print("I/O BENCHMARK RESULTS")
    print("=" * 80)
    print_table(rows, drop_cols=["n_total", "n_warmup", "total_s", "p50_s", "max_size"])

    write_csv(rows, args.output_csv)
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
