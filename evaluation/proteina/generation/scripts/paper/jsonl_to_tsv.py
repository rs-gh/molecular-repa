"""Dump per-set paper-sweep jsonls directly to a flat TSV.

The jsonl files at `results/paper/n{N}_paper_*/sweep_results.jsonl` are the
single source of truth for sweep metrics. The TSVs under
`figures/paper/n{N}_paper/n{N}_paper_tables.tsv` are the human/spreadsheet
view of that data.

Row grouping and order are driven by `results/paper/ablation_blocks.yaml`,
which maps each ablation block (verbatim from
`docs/research/proteina_ablation_checkpoints.md`) to an ordered list of
(profile, run, step) rows — baseline first within each block, anchor rows
repeated across blocks where the checkpoints doc lists them in multiple
ablations.

Output columns:
    ablation, ablation_id, profile, run, step, config_name, ckpt_path,
    <every `_res_*` metric column, sorted alphabetically>

Rerun after any backfill / sweep / metric rerun that touches the jsonls, OR
after editing ablation_blocks.yaml.

Usage:
    python jsonl_to_tsv.py n128
    python jsonl_to_tsv.py n256
    python jsonl_to_tsv.py n512
    python jsonl_to_tsv.py all   # all three
    python jsonl_to_tsv.py n128 --allow-missing   # don't error on yaml rows
                                                  # whose jsonl entry is absent
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent.parent.parent
RESULTS_ROOT = REPO_ROOT / "evaluation/proteina/generation/results/paper"
FIGURES_ROOT = REPO_ROOT / "evaluation/proteina/generation/figures/paper"
ABLATION_BLOCKS_YAML = RESULTS_ROOT / "ablation_blocks.yaml"

# Column order (per user schema 2026-05-16):
#   ablation_id, run, step,                                # row identity
#   <at-a-glance band — distribution / designability / diversity / novelty / SS>,
#   config_name, ckpt_path,                                # provenance
#   <remaining metric `_res_*` cols, alphabetically>,
#   H/E,                                                   # leftover derived
#   ablation                                               # block title from yaml
#
# The at-a-glance band groups the 11 columns we look at first for headline
# generation quality. The remaining _res_* columns are kept alphabetically
# sorted at the back for completeness.
LEADING_ID_COLUMNS = ["ablation_id", "run", "step"]
# At-a-glance order: distribution-match → designability → diversity → novelty → SS.
AT_A_GLANCE_COLUMNS = [
    "_res_PDB_FID",
    "_res_AFDB_FID",
    "_res_fS_T",
    "_res_designability_rate",
    "_res_scRMSD_mean",
    "_res_diversity_clusters_total",
    "_res_diversity_pairwise_tm_mean",
    "_res_novelty_foldseek_pdb_rate",
    "_res_novelty_foldseek_afdb_swissprot_rate",
    "H/E des",
    "_res_ss_jsd_pdb_2d",
    "_res_ss_jsd_afdb_2d",
]
PROVENANCE_COLUMNS = ["config_name", "ckpt_path"]
DERIVED_COLUMNS = [
    "H/E",
    "H/E des",
]  # computed in dump_set; H/E des is in the at-a-glance band, H/E global is appended after the alphabetic block
TRAILING_ID_COLUMNS = ["ablation"]


def load_blocks(set_name: str) -> list[dict]:
    blocks = yaml.safe_load(ABLATION_BLOCKS_YAML.read_text())
    return [b for b in blocks if b["set"] == set_name]


def build_index(set_name: str) -> tuple[dict[tuple[str, str, str], dict], set[str]]:
    """Index (profile, run, str(step)) → row across all profiles for this set."""
    profile_pattern = f"{set_name}_paper_*"
    jsonl_paths = sorted(RESULTS_ROOT.glob(f"{profile_pattern}/sweep_results.jsonl"))
    if not jsonl_paths:
        raise SystemExit(
            f"No sweep_results.jsonl under {RESULTS_ROOT}/{profile_pattern}/"
        )

    index: dict[tuple[str, str, str], dict] = {}
    metric_cols: set[str] = set()
    for jp in jsonl_paths:
        profile = jp.parent.name
        for ln in jp.read_text().splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            r["profile"] = profile
            key = (profile, str(r.get("run", "")), str(r.get("step", "")))
            index[key] = r
            metric_cols.update(k for k in r if k.startswith("_res_"))
    return index, metric_cols


def dump_set(set_name: str, allow_missing: bool = False) -> Path:
    """Render one TSV for `set_name` (e.g. 'n128') from its jsonls + yaml."""
    blocks = load_blocks(set_name)
    if not blocks:
        raise SystemExit(
            f"No ablation blocks for set={set_name} in {ABLATION_BLOCKS_YAML}"
        )

    index, metric_cols = build_index(set_name)
    # Build the column list explicitly. _res_* metric cols already placed in
    # the at-a-glance band are stripped from the alphabetic block so they
    # don't appear twice.
    at_a_glance_res = {c for c in AT_A_GLANCE_COLUMNS if c.startswith("_res_")}
    remaining_metrics = sorted(c for c in metric_cols if c not in at_a_glance_res)
    columns = (
        LEADING_ID_COLUMNS
        + AT_A_GLANCE_COLUMNS
        + PROVENANCE_COLUMNS
        + remaining_metrics
        + ["H/E"]
        + TRAILING_ID_COLUMNS
    )

    out_dir = FIGURES_ROOT / f"{set_name}_paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{set_name}_paper_tables.tsv"

    rows_emitted = 0
    missing: list[tuple[str, str, str, str]] = []  # (block_id, profile, run, step)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for block in blocks:
            for spec in block["rows"]:
                key = (spec["profile"], str(spec["run"]), str(spec["step"]))
                r = index.get(key)
                if r is None:
                    missing.append((block["id"], *key))
                    continue
                out_row = {c: r.get(c, "") for c in columns}
                h = r.get("_res_ss_frac_H")
                e = r.get("_res_ss_frac_E")
                out_row["H/E"] = (
                    (h / e) if (h is not None and e not in (None, 0)) else ""
                )
                h_d = r.get("_res_ss_frac_H_designable")
                e_d = r.get("_res_ss_frac_E_designable")
                out_row["H/E des"] = (
                    (h_d / e_d) if (h_d is not None and e_d not in (None, 0)) else ""
                )
                out_row["ablation"] = block["title"]
                out_row["ablation_id"] = block["id"]
                w.writerow(out_row)
                rows_emitted += 1

    print(
        f"Wrote {out}  ({rows_emitted} rows × {len(columns)} columns "
        f"across {len(blocks)} ablation block(s))"
    )
    if missing:
        msg = "\n".join(f"  {bid}: {p} / {r} / step={s}" for bid, p, r, s in missing)
        full = f"Missing jsonl entries for {len(missing)} yaml row(s):\n{msg}"
        if allow_missing:
            print(full, file=sys.stderr)
        else:
            raise SystemExit(
                full + "\nRerun with --allow-missing to ignore, or fix the yaml."
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "set",
        choices=["n128", "n256", "n512", "all"],
        help="Which paper set to render. 'all' renders n128, n256, n512.",
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="Don't error when a yaml row has no matching jsonl entry.",
    )
    args = ap.parse_args()
    sets = ["n128", "n256", "n512"] if args.set == "all" else [args.set]
    for s in sets:
        try:
            dump_set(s, allow_missing=args.allow_missing)
        except SystemExit as exc:
            print(f"  skipped {s}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
