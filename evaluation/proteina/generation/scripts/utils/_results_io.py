"""Shared loaders for sweep_results.jsonl.

Plot scripts read directly from the per-(run, step) JSONL emitted by
run_sweep.py — no intermediate CSV. JSONL is append-only with last-wins
dedup applied here, mirroring run_sweep.consolidate()'s logic.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_sweep_rows(jsonl_path: Path) -> list[dict]:
    """Read sweep_results.jsonl and return deduplicated, error-free rows.

    For each (run, step) pair the last non-error row in the file wins —
    matches the dedup logic in run_sweep.consolidate(). Returns [] if the
    file doesn't exist.
    """
    if not jsonl_path.exists():
        return []
    rows: list[dict] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    seen: dict[tuple, int] = {}
    for i, r in enumerate(rows):
        if "error" not in r and "run" in r and "step" in r:
            seen[(r["run"], int(r["step"]))] = i
    keep = set(seen.values())
    return [r for i, r in enumerate(rows) if i in keep]
