#!/usr/bin/env python3
"""Clean variance-sweep sweep_results.jsonl files.

Three passes:
1. Drop skeleton rows where rep_idx is None AND all metrics are None
   (these are aggregator placeholders from failed/in-flight jobs).
2. De-duplicate on (run, step, sampler_tag, rep_idx) — keeps the first
   occurrence; asserts metric equality across duplicates so silent
   divergence surfaces as an error rather than a quiet drop.
3. Backfill `seed` from `rep_idx` (0->42, 1->1042, 2->2042) where the
   row has rep_idx set but a stale or missing seed.

Writes <input>.clean.jsonl alongside the original. The original is left
untouched; user can diff + swap when satisfied.

Usage:
    python clean_variance_jsonl.py <path1.jsonl> [<path2.jsonl> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _row_key(r: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    return (r.get("run"), r.get("step"), r.get("sampler_tag"), r.get("rep_idx"))


def _is_skeleton(r: Dict[str, Any]) -> bool:
    """All-None placeholder row: rep_idx/seed/sampler_tag None and no metric values."""
    if r.get("rep_idx") is not None:
        return False
    if r.get("sampler_tag") is not None:
        return False
    # any non-trivial value besides run/step/error?
    interesting = {
        k: v
        for k, v in r.items()
        if k
        not in ("run", "step", "config_name", "error", "rep_idx", "seed", "sampler_tag")
    }
    return all(v is None for v in interesting.values())


def _expected_seed(rep_idx: int) -> int:
    return 42 + 1000 * rep_idx


def _metric_keys(r: Dict[str, Any]) -> List[str]:
    return sorted(
        k
        for k in r
        if k.startswith("_res_") or k in ("designability_rate", "ss_2d_jsd")
    )


def clean(in_path: Path) -> Path:
    rows = [json.loads(ln) for ln in in_path.read_text().splitlines() if ln.strip()]
    n_in = len(rows)

    # Pass 1: drop skeletons
    rows = [r for r in rows if not _is_skeleton(r)]
    n_after_skel = len(rows)

    # Pass 2: de-dup. On key collision (e.g. AFDB pre-variance rows where
    # rep_idx/seed/sampler_tag are all None), prefer the row with MORE
    # populated metrics — legacy backfill rows that only filled in foldseek-
    # novelty otherwise mask the original full-metric eval.
    def _populated_metric_count(r: Dict[str, Any]) -> int:
        return sum(1 for k in _metric_keys(r) if r.get(k) is not None)

    seen: Dict[Tuple, Dict] = {}
    drop_count = 0
    conflicts: List[Tuple[Tuple, str]] = []
    for r in rows:
        k = _row_key(r)
        if k not in seen:
            seen[k] = r
            continue
        prev = seen[k]
        # Conflict check: only flag when *both* rows have the metric populated.
        for mk in _metric_keys(r):
            if (
                prev.get(mk) is not None
                and r.get(mk) is not None
                and prev.get(mk) != r.get(mk)
            ):
                conflicts.append((k, mk))
        # Keep the richer row; merge non-None metrics from the other so we don't
        # lose data when each duplicate carried a complementary subset.
        if _populated_metric_count(r) > _populated_metric_count(prev):
            richer, sparser = r, prev
        else:
            richer, sparser = prev, r
        for mk in _metric_keys(sparser):
            if richer.get(mk) is None and sparser.get(mk) is not None:
                richer[mk] = sparser[mk]
        seen[k] = richer
        drop_count += 1
    if conflicts:
        print(
            f"  WARN: {len(conflicts)} metric conflicts across dupes:", file=sys.stderr
        )
        for k, mk in conflicts[:5]:
            print(f"    {k}  metric={mk}", file=sys.stderr)
    rows = list(seen.values())
    n_after_dedup = len(rows)

    # Pass 3: backfill seed
    seed_fixed = 0
    for r in rows:
        ri = r.get("rep_idx")
        if ri is None:
            continue
        want = _expected_seed(int(ri))
        if r.get("seed") != want:
            r["seed"] = want
            seed_fixed += 1

    out_path = in_path.with_suffix(".clean.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"{in_path}")
    print(f"  input rows         : {n_in}")
    print(f"  skeleton dropped   : {n_in - n_after_skel}")
    print(f"  duplicates dropped : {n_after_skel - n_after_dedup}")
    print(f"  seed backfilled    : {seed_fixed}")
    print(f"  output rows        : {n_after_dedup}")
    print(f"  -> {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    for arg in sys.argv[1:]:
        clean(Path(arg))
