# ruff: noqa: E402, E702
"""Second pass: add CATH probe results to an existing sweep_results.jsonl.

The first sweep ran `CATHLabelTransform` was NOT applied in the data pipeline,
so `g.cath_code` was empty/None on every protein and CATH probes yielded
`cath.level == "none"`. This script:

  1. Re-loads the same batch as the sweep (same n_proteins, max_size, lmdb).
  2. Applies CATHLabelTransform at runtime using the pre-downloaded
     CATH source files under `${DATA_PATH}/pdb_train/`.
  3. Aborts with diagnostics if labels still don't populate (e.g. wrong
     `graph.id` format).
  4. For each (run, step) group in the existing JSONL, reloads the
     checkpoint once, re-extracts hidden states at all cached layers,
     re-runs the CATH probe, and updates the row's `cath` field in place.
     Contact probe data is preserved unchanged.
  5. Writes the consolidated CSV/MD fresh from the updated JSONL.

Expected wall-clock: 1 checkpoint-load per (run, step) × ~47 → ~25 min.

Usage:
  sbatch hpc-scripts/proteina/evaluation/run_probes.sh --cath_patch
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

HERE = Path(__file__).resolve().parent
# HERE        = .../representation/scripts (for utils/contact/cath shims + run_sweep)
# HERE.parent = .../representation         (so `from lib import ...` works)
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from utils import (
    RUN_SCHEDULES,
    _default_device,
    extract_gearnet_embeddings,
    extract_model_hidden_states_multilayer,
    find_checkpoint_path,
    load_checkpoint_by_path,
    load_proteina_batch,
)
from cath import run_cath_probe
from run_sweep import consolidate, _load_gearnet


def _apply_cath_labels(raw, data_path: str) -> Tuple[int, int, List]:
    """Apply CATHLabelTransform in-place. Returns (n_labelled, n_total, sample_ids)."""
    from proteinfoundation.datasets.transforms import CATHLabelTransform

    # CATH source files live at $DATA_PATH/pdb_train/ on this cluster (not cathdata/).
    cath_root = Path(data_path) / "pdb_train"
    tf = CATHLabelTransform(root_dir=str(cath_root))

    labelled = 0
    sample_ids = []
    for g in raw:
        try:
            tf.forward(g)
        except Exception as e:
            print(f"[warn] CATH transform failed on {getattr(g, 'id', '?')}: {e}")
        cc = getattr(g, "cath_code", None)
        if cc:  # non-empty list
            labelled += 1
        if len(sample_ids) < 3:
            sample_ids.append((getattr(g, "id", "?"), cc))
    return labelled, len(raw), sample_ids


def _rerun_cath(reps: torch.Tensor, mask, raw, level: str):
    res = run_cath_probe(reps, mask, raw, preferred_level=level)
    return asdict(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_proteins", type=int, default=200)
    ap.add_argument("--max_size", type=int, default=256)
    ap.add_argument("--cath_level", type=str, default="T", choices=["C", "A", "T"])
    # Default writes to .../representation/results/ (sibling of scripts/).
    ap.add_argument("--output_dir", type=str, default=str(HERE.parent / "results"))
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    jsonl_path = outdir / "sweep_results.jsonl"
    if not jsonl_path.exists():
        sys.exit(f"No existing JSONL at {jsonl_path}. Run run_sweep.py first.")

    rows: List[Dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(rows)} existing rows from {jsonl_path.name}")

    device = _default_device()

    # --- Reload shared batch with same parameters as the sweep ---
    print(f"Reloading {args.n_proteins} proteins (≤ {args.max_size} residues)...")
    batch, raw = load_proteina_batch(
        n=args.n_proteins, max_size=args.max_size, device=device
    )
    print(f"  Loaded {len(raw)} proteins")

    # --- Sanity check: CATH labels must populate ---
    data_path = os.environ.get("DATA_PATH", "/rds/user/sr2173/hpc-work/proteina/data")
    n_lab, n_tot, sample_ids = _apply_cath_labels(raw, data_path)
    print(f"  CATH labels populated on {n_lab}/{n_tot} proteins")
    print(f"  Sample graph.id → cath_code: {sample_ids}")
    if n_lab < max(10, int(0.05 * n_tot)):
        sys.exit(
            f"FATAL: only {n_lab}/{n_tot} proteins got CATH labels — "
            f"graph.id format likely doesn't match pdb_chain_cath_uniprot.tsv. "
            f"See sample IDs above and adjust CATHLabelTransform or lookup key."
        )

    # --- Group rows by (run, step) → list of layer-rows to update ---
    groups: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        groups[(r["run"], int(r["step"]))].append(r)
    print(f"Grouped into {len(groups)} (run, step) pairs")

    # --- Update each group: extract reps, rerun CATH, update row in place ---
    updated = 0
    for (run, step), group in sorted(groups.items()):
        layers_needed = sorted({int(r["layer"]) for r in group})

        # Gearnet special: one encoder pass, one "layer" = -1
        if run == "gearnet":
            print(f"\n=== gearnet @ step {step} ===")
            try:
                enc = _load_gearnet()
                reps = extract_gearnet_embeddings(enc, batch)
                new_cath = _rerun_cath(reps, batch["mask"], raw, args.cath_level)
                for r in group:
                    r["cath"] = new_cath
                    updated += 1
                print(
                    f"  cath-{new_cath['level']} acc={new_cath['accuracy']:.3f} "
                    f"classes={new_cath['n_classes']}"
                )
                del enc
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"  ❌ {e}")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            continue

        # Regular run: find checkpoint, extract reps at needed layers, rerun CATH.
        if run not in RUN_SCHEDULES:
            print(f"\n[skip] unknown run {run}")
            continue
        run_dir, is_repa, _, _ = RUN_SCHEDULES[run]
        ckpt = find_checkpoint_path(run_dir, step, prefer_ema=True)
        if ckpt is None:
            print(f"\n[skip] no ckpt for {run}@{step}")
            continue

        print(f"\n=== {run} @ step {step} (layers {layers_needed}) ===")
        try:
            model = load_checkpoint_by_path(str(ckpt), is_repa=is_repa, device=device)
            reps_by_layer = extract_model_hidden_states_multilayer(
                model, batch, layers_needed
            )
            for r in group:
                L = int(r["layer"])
                if L not in reps_by_layer:
                    continue
                new_cath = _rerun_cath(
                    reps_by_layer[L], batch["mask"], raw, args.cath_level
                )
                r["cath"] = new_cath
                updated += 1
                print(
                    f"  L{L:2d}: cath-{new_cath['level']} acc={new_cath['accuracy']:.3f} "
                    f"classes={new_cath['n_classes']}"
                )
            del model, reps_by_layer
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  ❌ {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Rewrite JSONL atomically (temp → rename) ---
    tmp_path = jsonl_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    tmp_path.replace(jsonl_path)

    consolidate(outdir)
    print(f"\nPatched {updated} rows. Consolidated CSV/MD written to {outdir}.")


if __name__ == "__main__":
    main()
