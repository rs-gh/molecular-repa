"""Append SS columns to hand-curated paper tsv files (n256, n512).

These tsvs are produced by `md_tables_to_tsv.py` from hand-curated .md files
whose row labels are display names ("REPA L4 ep22") rather than canonical
run keys ("repa_l4_256_ep22"). To avoid touching the .md, we post-process
the .tsv directly: parse each data row, match (display_label, step) →
canonical (run, step) heuristically against the paper sweep JSONLs, and
append four SS columns: %H, %E, JSD (PDB), JSD (AFDB).

Rows we cannot match get blank SS cells. Header rows (first column only,
trailing empty cells) and section dividers are passed through with empty
SS cells.

Idempotent: detects "SS %H" already in header and rewrites tail rather
than re-appending.

Usage:
    python evaluation/proteina/generation/scripts/paper/append_ss_to_tsv.py \\
        evaluation/proteina/generation/figures/paper/n256_paper/n256_paper_tables.tsv \\
        evaluation/proteina/generation/figures/paper/n512_paper/n512_paper_tables.tsv
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
RESULTS_GLOBS = [
    str(
        REPO_ROOT / "evaluation/proteina/generation/results/paper/*/sweep_results.jsonl"
    ),
    str(
        REPO_ROOT / "evaluation/proteina/generation/results/lite/*/sweep_results.jsonl"
    ),
]

SS_HEADERS = [
    "SS %H ↑",
    "SS %E ↑",
    "SS JSD (PDB) ↓",
    "SS JSD (AFDB) ↓",
    "SS %H des. ↑",
    "SS %E des. ↑",
    "SS JSD des. (PDB) ↓",
    "SS JSD des. (AFDB) ↓",
    "SS JSD 2D (PDB) ↓",
    "SS JSD 2D (AFDB) ↓",
    "SS JSD 2D des. (PDB) ↓",
    "SS JSD 2D des. (AFDB) ↓",
]
SS_KEYS = [
    "_res_ss_frac_H",
    "_res_ss_frac_E",
    "_res_ss_jsd_pdb",
    "_res_ss_jsd_afdb",
    "_res_ss_frac_H_designable",
    "_res_ss_frac_E_designable",
    "_res_ss_jsd_pdb_designable",
    "_res_ss_jsd_afdb_designable",
    "_res_ss_jsd_pdb_2d",
    "_res_ss_jsd_afdb_2d",
    "_res_ss_jsd_pdb_designable_2d",
    "_res_ss_jsd_afdb_designable_2d",
]


def _load_ss_index() -> dict[tuple[str, int], dict]:
    """Return {(canonical_run, step_int): row_dict} from all paper sweep JSONLs."""
    out: dict[tuple[str, int], dict] = {}
    paths: list[str] = []
    for g in RESULTS_GLOBS:
        paths.extend(glob.glob(g))
    for jsonl in paths:
        for line in Path(jsonl).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (
                "error" in r
                or "_res_ss_n" not in r
                or "run" not in r
                or "step" not in r
            ):
                continue
            out[(str(r["run"]), int(r["step"]))] = r
    return out


_STEP_RE = re.compile(r"^([\d.]+)\s*([kKmM]?)$")


def _parse_step(step_str: str) -> int | None:
    s = step_str.strip().replace(",", "").lstrip("~")
    if not s or s == "—":
        return None
    if s.isdigit():
        return int(s)
    m = _STEP_RE.match(s)
    if not m:
        return None
    val = float(m.group(1))
    suffix = m.group(2).lower()
    if suffix == "k":
        val *= 1_000
    elif suffix == "m":
        val *= 1_000_000
    return int(round(val))


def _label_tokens(label: str) -> list[str]:
    """Lowercase tokens from a display label, keeping epXX/stepXX/lambda etc."""
    s = label.lower()
    s = s.replace("(", " ").replace(")", " ").replace("/", " ").replace("-", " ")
    s = s.replace("=", " ")
    s = s.replace("λ", "lambda")
    return [t for t in re.split(r"[\s,]+", s) if t]


def _match_run(label: str, step: int | None, ss_index: dict) -> dict | None:
    """Best-effort match of display label+step to a canonical SS row."""
    if step is None:
        return None
    tokens = set(_label_tokens(label))
    candidates: list[tuple[int, dict]] = []
    for (run, run_step), row in ss_index.items():
        if abs(run_step - step) > 1500:  # tolerate slight Lightning-step drift
            continue
        run_tokens = set(re.split(r"[_\-]", run.lower()))
        # Score: matched tokens beyond size markers.
        size_neutral_run = run_tokens - {"128", "256", "512", "sm"}
        size_neutral_lbl = tokens - {"128", "256", "512", "sm"}
        overlap = len(size_neutral_run & size_neutral_lbl)
        if overlap == 0:
            # Fallback: substring containment (≥4 chars) catches display-label
            # variants like "proteinmpnn" ⊃ "mpnn" or "lambda" ⊂ "lambda1".
            for lt in size_neutral_lbl:
                if len(lt) < 4:
                    continue
                for rt in size_neutral_run:
                    if len(rt) < 4:
                        continue
                    if lt in rt or rt in lt:
                        overlap += 1
                        break
            if overlap == 0:
                continue
        candidates.append((overlap, row))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def _is_separator_row(cells: list[str]) -> bool:
    """Section header / blank row: only cell 0 has content."""
    return bool(cells[0]) and all(not c for c in cells[1:])


def _is_blank_row(cells: list[str]) -> bool:
    return all(not c for c in cells)


def append_ss_columns(tsv_path: Path) -> int:
    text = tsv_path.read_text()
    lines = text.splitlines()
    if not lines:
        return 0

    ss_index = _load_ss_index()
    new_lines: list[str] = []
    n_matched = n_data = 0
    in_table = False
    header_run_idx = None
    header_step_idx = None
    notes_idx: int | None = None  # last column index ("Notes") if present

    def _trim_existing_ss(cells: list[str], header_cells: list[str]) -> list[str]:
        """Drop any pre-existing SS columns so we can reappend cleanly.

        Repeats the trim while SS columns remain in `header_cells` to handle
        prior runs that double-appended (each pass strips one SS block).
        Accepts both header conventions (n256 "SS %H ↑" and n128 "SS %H (↑)")
        by matching on `startswith("SS %H")`.
        """

        def _ss_block_start(h: list[str]) -> int | None:
            for i, x in enumerate(h):
                if x.strip().startswith("SS %H"):
                    return i
            return None

        result = list(cells)
        h = list(header_cells)
        while True:
            ss_start = _ss_block_start(h)
            if ss_start is None:
                break
            # Find contiguous SS run starting at ss_start.
            ss_end = ss_start
            while ss_end < len(h) and h[ss_end].strip().startswith("SS "):
                ss_end += 1
            # Drop cells[ss_start:ss_end] (only when row is wide enough).
            if ss_end <= len(result):
                result = result[:ss_start] + result[ss_end:]
            h = h[:ss_start] + h[ss_end:]
        return result

    header_cells_raw: list[str] = []

    for line in lines:
        if not line.strip():
            new_lines.append(line)
            in_table = False
            header_run_idx = header_step_idx = notes_idx = None
            header_cells_raw = []
            continue
        cells = line.split("\t")
        # Detect a header row: first cell == "Run".
        if cells[0].strip() == "Run":
            # Keep the ORIGINAL header (incl. any duplicate SS blocks) for
            # data-row trim alignment — we strip those same indices below.
            header_cells_raw = list(cells)
            cells = _trim_existing_ss(cells, header_cells_raw)
            # Insert SS headers before "Notes" if present, else at end.
            notes_idx = (
                len(cells) - 1
                if cells and cells[-1].strip().lower() == "notes"
                else None
            )
            insert_at = notes_idx if notes_idx is not None else len(cells)
            new_header = cells[:insert_at] + SS_HEADERS + cells[insert_at:]
            new_lines.append("\t".join(new_header))
            in_table = True
            header_run_idx = 0
            header_step_idx = 1
            continue

        if not in_table:
            new_lines.append(line)
            continue

        # Trim any old SS values then re-insert in the right slot.
        cells = _trim_existing_ss(cells, header_cells_raw)
        insert_at = notes_idx if notes_idx is not None else len(cells)

        if _is_separator_row(cells) or _is_blank_row(cells):
            ss_cells = [""] * len(SS_HEADERS)
        else:
            n_data += 1
            label = cells[header_run_idx] if header_run_idx is not None else cells[0]
            step_str = (
                cells[header_step_idx]
                if header_step_idx is not None and len(cells) > header_step_idx
                else ""
            )
            step = _parse_step(step_str)
            row = _match_run(label, step, ss_index)
            if row is None:
                ss_cells = [""] * len(SS_HEADERS)
            else:
                n_matched += 1
                vals = []
                for k in SS_KEYS:
                    v = row.get(k)
                    if v is None:
                        vals.append("")
                    elif "frac" in k:
                        vals.append(f"{v:.3f}")
                    else:
                        vals.append(f"{v:.4f}")
                ss_cells = vals

        new_cells = cells[:insert_at] + ss_cells + cells[insert_at:]
        new_lines.append("\t".join(new_cells))

    tsv_path.write_text("\n".join(new_lines) + "\n")
    print(f"  {tsv_path.name}: matched {n_matched}/{n_data} data rows")
    return n_matched


def main(argv: list[str]) -> None:
    if not argv:
        print("usage: append_ss_to_tsv.py <tsv> [<tsv> ...]", file=sys.stderr)
        sys.exit(2)
    for path in argv:
        append_ss_columns(Path(path))


if __name__ == "__main__":
    main(sys.argv[1:])
