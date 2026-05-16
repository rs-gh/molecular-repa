"""Dump per-set paper-sweep jsonls directly to a flat TSV.

The jsonl files at `results/paper/n{N}_paper_*/sweep_results.jsonl` are the
single source of truth for sweep metrics. The TSVs under
`figures/paper/n{N}_paper/n{N}_paper_tables.tsv` are the human/spreadsheet
view of that data. Earlier those TSVs were produced by hand-curating a
markdown file and converting via `md_tables_to_tsv.py`, which drifts the
moment the jsonls change.

This script renders the TSV directly from the jsonls: one row per
checkpoint, columns are `profile` + `run` + `step` + every `_res_` metric
that appears anywhere in the set (union, sorted alphabetically; missing
cells are empty). Run after any backfill / sweep / metric rerun that
touches the jsonls so the TSV doesn't fall behind.

Usage:
    python jsonl_to_tsv.py n128
    python jsonl_to_tsv.py n256
    python jsonl_to_tsv.py n512
    python jsonl_to_tsv.py all   # all three
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent.parent.parent
RESULTS_ROOT = REPO_ROOT / "evaluation/proteina/generation/results/paper"
FIGURES_ROOT = REPO_ROOT / "evaluation/proteina/generation/figures/paper"

# Identifier columns (always present, in this order). Everything else (the
# `_res_*` metric columns) is appended alphabetically.
ID_COLUMNS = ["profile", "run", "step", "config_name", "ckpt_path"]


def dump_set(set_name: str) -> Path:
    """Render one TSV for `set_name` (e.g. 'n128') from its jsonls.

    Returns the path of the TSV written.
    """
    profile_pattern = f"{set_name}_paper_*"
    jsonl_paths = sorted(RESULTS_ROOT.glob(f"{profile_pattern}/sweep_results.jsonl"))
    if not jsonl_paths:
        raise SystemExit(
            f"No sweep_results.jsonl under {RESULTS_ROOT}/{profile_pattern}/"
        )

    rows: list[dict] = []
    metric_cols: set[str] = set()
    for jp in jsonl_paths:
        profile = jp.parent.name
        for ln in jp.read_text().splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            r["profile"] = profile
            rows.append(r)
            metric_cols.update(k for k in r if k.startswith("_res_"))

    columns = ID_COLUMNS + sorted(metric_cols)
    out_dir = FIGURES_ROOT / f"{set_name}_paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{set_name}_paper_tables.tsv"

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})

    print(
        f"Wrote {out}  ({len(rows)} rows × {len(columns)} columns "
        f"across {len(jsonl_paths)} profile(s))"
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "set",
        choices=["n128", "n256", "n512", "all"],
        help="Which paper set to render. 'all' renders n128, n256, n512.",
    )
    args = ap.parse_args()
    sets = ["n128", "n256", "n512"] if args.set == "all" else [args.set]
    for s in sets:
        try:
            dump_set(s)
        except SystemExit as exc:
            print(f"  skipped {s}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
