"""Convert GFM markdown tables in a .md file to TSV for spreadsheet paste.

Finds every GFM table block in the input markdown and writes them — in order,
with a blank separator row between tables — to a single TSV. Pasting the TSV
into Google Sheets or Excel yields one cell per column with no quoting drama.

Inline markdown inside cells is stripped: `**bold**`, `_underscore italic_`,
`*star italic*`, and backtick `code` keep their text and lose the markup.
Section-header rows (those rendered with the label in the first column and
empty trailing cells) survive as a row with the label in column A and tabs
after — they reappear as visible group separators in the spreadsheet.

CLI:
    python md_tables_to_tsv.py <input.md> [<output.tsv>]

Library:
    from md_tables_to_tsv import md_to_tsv
    md_to_tsv(Path("foo.md"), Path("foo.tsv"))
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# `**bold**` → `bold`. Non-greedy so adjacent bold spans don't merge.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
# `` `code` `` → `code`.
_CODE_RE = re.compile(r"`([^`]+)`")
# `*ital*` and `_ital_` → `ital`. Conservative: require non-word boundary on
# both sides so we don't mangle `*` glob patterns or snake_case identifiers.
_STAR_ITAL_RE = re.compile(r"(?<![\w*])\*(?!\*)([^*\n]+?)\*(?!\*)(?![\w*])")
_UND_ITAL_RE = re.compile(r"(?<![\w_])_(?!_)([^_\n]+?)_(?!_)(?![\w_])")


def _strip_inline_md(s: str) -> str:
    s = _BOLD_RE.sub(r"\1", s)
    s = _CODE_RE.sub(r"\1", s)
    s = _STAR_ITAL_RE.sub(r"\1", s)
    s = _UND_ITAL_RE.sub(r"\1", s)
    # Tabs inside cells would corrupt the TSV; collapse to a single space.
    return s.replace("\t", " ").strip()


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    # Split on pipes that aren't escaped with a backslash.
    parts = re.split(r"(?<!\\)\|", s)
    return [_strip_inline_md(p.replace("\\|", "|")) for p in parts]


# A GFM separator row: `| :--- | ---: | --- |` with at least one column.
_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def extract_tables(md_text: str) -> list[list[list[str]]]:
    """Return list of tables; each table is `list[rows]`; row is `list[cells]`."""
    lines = md_text.splitlines()
    tables: list[list[list[str]]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            line.lstrip().startswith("|")
            and i + 1 < len(lines)
            and _SEP_RE.match(lines[i + 1])
        ):
            header = _split_row(line)
            i += 2  # skip header + separator
            rows: list[list[str]] = [header]
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            tables.append(rows)
        else:
            i += 1
    return tables


def md_to_tsv(md_path: Path, tsv_path: Path) -> int:
    """Read markdown at `md_path`, write TSV at `tsv_path`. Returns table count."""
    text = md_path.read_text()
    tables = extract_tables(text)
    out_lines: list[str] = []
    for ti, tab in enumerate(tables):
        if ti > 0:
            out_lines.append("")  # blank separator row between tables
        for row in tab:
            out_lines.append("\t".join(row))
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    tsv_path.write_text("\n".join(out_lines) + "\n")
    return len(tables)


def main(argv: list[str]) -> None:
    if len(argv) < 1 or len(argv) > 2:
        print("usage: md_tables_to_tsv.py <input.md> [<output.tsv>]", file=sys.stderr)
        sys.exit(2)
    in_path = Path(argv[0])
    out_path = Path(argv[1]) if len(argv) == 2 else in_path.with_suffix(".tsv")
    n = md_to_tsv(in_path, out_path)
    print(f"Wrote {out_path} ({n} table{'s' if n != 1 else ''})")


if __name__ == "__main__":
    main(sys.argv[1:])
