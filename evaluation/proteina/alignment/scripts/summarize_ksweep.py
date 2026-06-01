"""Summarize the CKNNA k-sensitivity sweep.

Reads every cknna_matrix_per_{residue,protein}{,_k*}.jsonl in results/ and
reports, per k, the peak REPA-vs-baseline separation. This answers the
question the report flags: is the small absolute CKNNA (~0.02-0.05 at k=10)
an artifact of the small mutual-kNN neighbourhood, or robust as k -> N
(where CKNNA -> global CKA)?

For each (k, mode) it prints:
  - peak REPA CKNNA   (max point estimate over REPA rows x layers x encoders)
  - matched baseline  (baseline CKNNA at that same layer x encoder cell)
  - the baseline's own peak (the "random floor" ceiling)

Run:
    python evaluation/proteina/alignment/scripts/summarize_ksweep.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
BASELINE = "baseline"


def _k_from_name(name: str) -> int:
    m = re.search(r"_k(\d+)\.jsonl$", name)
    return int(m.group(1)) if m else 10


def _load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    files = sorted(RESULTS.glob("cknna_matrix_per_residue*.jsonl")) + sorted(
        RESULTS.glob("cknna_matrix_per_protein*.jsonl")
    )
    if not files:
        print(f"No matrices found in {RESULTS}")
        return

    rows = []
    for path in files:
        mode = "per_residue" if "per_residue" in path.name else "per_protein"
        k = _k_from_name(path.name)
        recs = _load(path)
        cells = {(r["model"], r["layer"], r["encoder"]): r["cknna"] for r in recs}

        repa = [r for r in recs if r["model"] != BASELINE]
        if not repa:
            continue
        best = max(repa, key=lambda r: r["cknna"])
        matched_base = cells.get((BASELINE, best["layer"], best["encoder"]))
        base_peak = max(
            (r["cknna"] for r in recs if r["model"] == BASELINE), default=float("nan")
        )
        rows.append(
            {
                "mode": mode,
                "k": k,
                "n": recs[0]["n_samples"],
                "repa_peak": best["cknna"],
                "at": f"{best['model']} L{best['layer']}x{best['encoder']}",
                "base_matched": matched_base,
                "base_peak": base_peak,
            }
        )

    rows.sort(key=lambda r: (r["mode"], r["k"]))
    hdr = f"{'mode':12} {'k':>4} {'N':>6} {'REPA peak':>10} {'base@cell':>10} {'base peak':>10}  where"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        bm = "n/a" if r["base_matched"] is None else f"{r['base_matched']:.4f}"
        print(
            f"{r['mode']:12} {r['k']:>4} {r['n']:>6} {r['repa_peak']:>10.4f} "
            f"{bm:>10} {r['base_peak']:>10.4f}  {r['at']}"
        )


if __name__ == "__main__":
    main()
