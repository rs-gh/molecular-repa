"""Backfill ss_cache/ss_fractions.npz for eval dirs missing SS cache.

For each eval dir that has samples_fid/*_fid.pdb but no ss_cache/ss_fractions.npz,
runs P-SEA via the existing compute_per_sample_ss_fractions helper and writes
the cache. Idempotent — skips dirs whose cache already exists.

Usage:
    python backfill_ss_cache.py /rds/user/sr2173/hpc-work/proteina/eval_output
    python backfill_ss_cache.py /home/sr2173/git/molecular-repa/eval_output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the existing utils module is importable.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils.ss_metrics import compute_per_sample_ss_fractions  # noqa: E402


def find_samples(eval_dir: Path) -> list[str]:
    samples = sorted(eval_dir.glob("samples_fid/*_fid.pdb"))
    return [str(p) for p in samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_root", type=Path)
    ap.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Substring filter for eval dir basenames (e.g. '256').",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many eval dirs (smoke-test).",
    )
    args = ap.parse_args()

    eval_dirs = [p for p in sorted(args.eval_root.iterdir()) if p.is_dir()]
    if args.filter:
        eval_dirs = [p for p in eval_dirs if args.filter in p.name]

    todo = []
    for d in eval_dirs:
        cache = d / "ss_cache" / "ss_fractions.npz"
        if cache.exists():
            continue
        samples = find_samples(d)
        if not samples:
            continue
        todo.append((d, samples, cache))

    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(todo)} eval dirs need backfill")
    for i, (d, samples, cache) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {d.name}  ({len(samples)} PDBs)")
        cache.parent.mkdir(parents=True, exist_ok=True)
        fracs, kept = compute_per_sample_ss_fractions(samples, cache_path=cache)
        print(f"  → {len(kept)} kept, cached to {cache}")


if __name__ == "__main__":
    main()
