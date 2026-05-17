"""Aggregate replicate runs from a variance/sampler sweep into mean ± sd.

Reads `sweep_results.jsonl` produced by `evaluation/proteina/generation/scripts/
run_sweep.py` (extended with sampler_tag + rep_idx fields), groups rows by
(run, step, sampler_tag), and emits mean / sd / N over the replicate axis for
every numeric `_res_*` metric. Sample sd (ddof=1) is used; N=1 groups emit NaN
for sd.

Output:
  - <output_dir>/sweep_results_agg.csv   one row per (run, step, sampler_tag),
                                          columns: <metric>_mean, <metric>_sd, n_reps
  - <output_dir>/sweep_results_agg.jsonl same content, one JSON object per row
  - <output_dir>/sweep_results_agg.md    glanceable summary table (mean ± sd)

Usage:
  python evaluation/proteina/joint/scripts/aggregate_variance.py \
    --jsonl evaluation/proteina/generation/results/variance/n128/sweep_results.jsonl

Legacy JSONLs without sampler_tag / rep_idx fields are aggregated with
sampler_tag=None and rep_idx=None — i.e. one "rep" per (run, step), so the mean
equals the value and sd is NaN. Useful for sanity-checking the aggregator
against existing point-estimate sweeps before any replicate has finished.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _is_metric(k: str) -> bool:
    return isinstance(k, str) and k.startswith("_res_")


def _coerce_float(v) -> Optional[float]:
    """Return v as float if numeric and finite; else None.

    Designed for `_res_*` columns. Strings that parse as numbers (legacy CSV
    backfill writes some numerics as strings) are also accepted.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, str):
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except ValueError:
            return None
    return None


def _load_rows(jsonl_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                continue
            if "run" not in r or "step" not in r:
                continue
            rows.append(r)
    return rows


def _group_key(r: Dict) -> Tuple[str, int, Optional[str]]:
    return (r["run"], int(r["step"]), r.get("sampler_tag"))


def _dedup_within_rep(group: List[Dict]) -> List[Dict]:
    """Keep the last row per rep_idx within a (run, step, sampler_tag) group.

    Reruns of the same (group key, rep) collapse to the most recent entry —
    matching run_sweep.consolidate()'s last-write-wins semantics.
    """
    by_rep: Dict[Optional[int], Dict] = {}
    for r in group:
        rep = r.get("rep_idx")
        if rep is not None:
            rep = int(rep)
        by_rep[rep] = r
    return list(by_rep.values())


def aggregate(rows: List[Dict]) -> List[Dict]:
    """Return one aggregated row per (run, step, sampler_tag) group.

    `_res_*` columns become `<metric>_mean` and `<metric>_sd`. `n_reps` is the
    count of replicate rows in the group. Sample sd (ddof=1); NaN for N<=1.
    """
    grouped: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        grouped[_group_key(r)].append(r)

    out: List[Dict] = []

    # Sort with a None-safe key: legacy rows (sampler_tag=None) sort to the top.
    def _sort_key(item):
        (run, step, tag), _ = item
        return (run, step, tag is not None, tag or "")

    for (run, step, sampler_tag), group in sorted(grouped.items(), key=_sort_key):
        group = _dedup_within_rep(group)
        metric_keys = sorted({k for r in group for k in r.keys() if _is_metric(k)})
        agg_row: Dict = {
            "run": run,
            "step": step,
            "sampler_tag": sampler_tag,
            "n_reps": len(group),
            "rep_seeds": sorted(
                int(r["seed"]) for r in group if r.get("seed") is not None
            ),
        }
        for m in metric_keys:
            vals = [_coerce_float(r.get(m)) for r in group]
            vals = [v for v in vals if v is not None]
            if not vals:
                agg_row[f"{m}_mean"] = None
                agg_row[f"{m}_sd"] = None
                continue
            n = len(vals)
            mean = sum(vals) / n
            if n >= 2:
                ss = sum((v - mean) ** 2 for v in vals)
                sd = math.sqrt(ss / (n - 1))  # ddof=1
            else:
                sd = float("nan")
            agg_row[f"{m}_mean"] = mean
            agg_row[f"{m}_sd"] = sd
        out.append(agg_row)
    return out


def write_csv(agg_rows: List[Dict], path: Path) -> None:
    import csv

    if not agg_rows:
        path.write_text("")
        return
    cols: List[str] = []
    seen: set = set()
    for r in agg_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                cols.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in agg_rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def write_jsonl(agg_rows: List[Dict], path: Path) -> None:
    with path.open("w") as f:
        for r in agg_rows:
            f.write(json.dumps(r, default=str) + "\n")


# Subset surfaced in the glanceable MD; full numbers live in CSV/JSONL.
_MD_METRICS = [
    "_res_PDB_FID",
    "_res_AFDB_FID",
    "_res_designability_rate",
    "_res_diversity_clusters_mean",
    "_res_novelty_rate",
]


def _fmt_mean_sd(mean, sd) -> str:
    if mean is None:
        return ""
    if sd is None or (isinstance(sd, float) and math.isnan(sd)):
        return f"{mean:.4f} (n=1)"
    return f"{mean:.4f} ± {sd:.4f}"


def write_md(agg_rows: List[Dict], path: Path) -> None:
    if not agg_rows:
        path.write_text("# No rows.\n")
        return
    cols_present = [m for m in _MD_METRICS if any(f"{m}_mean" in r for r in agg_rows)]
    header = ["run", "step", "sampler_tag", "n_reps", *cols_present]
    lines = [
        f"# Variance aggregation — {len(agg_rows)} groups",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in agg_rows:
        vals = [
            str(r["run"]),
            str(r["step"]),
            str(r.get("sampler_tag") or ""),
            str(r.get("n_reps", 0)),
        ]
        for m in cols_present:
            vals.append(_fmt_mean_sd(r.get(f"{m}_mean"), r.get(f"{m}_sd")))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to sweep_results.jsonl from a variance / sampler sweep.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write the aggregated outputs (default: alongside the JSONL).",
    )
    args = p.parse_args()

    jsonl_path = Path(args.jsonl).resolve()
    if not jsonl_path.exists():
        raise SystemExit(f"JSONL not found: {jsonl_path}")
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else jsonl_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(jsonl_path)
    agg_rows = aggregate(rows)

    csv_path = output_dir / "sweep_results_agg.csv"
    jsonl_agg_path = output_dir / "sweep_results_agg.jsonl"
    md_path = output_dir / "sweep_results_agg.md"
    write_csv(agg_rows, csv_path)
    write_jsonl(agg_rows, jsonl_agg_path)
    write_md(agg_rows, md_path)

    print(f"Aggregated {len(rows)} input rows -> {len(agg_rows)} groups")
    print(f"  csv:   {csv_path}")
    print(f"  jsonl: {jsonl_agg_path}")
    print(f"  md:    {md_path}")


if __name__ == "__main__":
    main()
