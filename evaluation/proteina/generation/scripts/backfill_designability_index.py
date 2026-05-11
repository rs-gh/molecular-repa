"""Reconstruct `designability_index.csv` for past eval runs whose
`tmp_designability/` artefacts are still on disk.

The expensive steps of designability (ProteinMPNN sequence generation +
ESMFold refolding) write their outputs to
`<eval_root>/tmp_designability/<pdb_name>/{seqs/, esm_*.pdb_esm}`. As long
as those survive, we can re-run the cheap finishing pass — RMSD + TM-score
between the original and each refold, with pLDDT recovered from the refold's
B-factor column — without touching a GPU.

Output schema matches what `evaluate.py:_write_designability_index` writes
on a fresh run, so downstream consumers (`backfill_ss.py`, future plotting)
can treat the two interchangeably.

Usage (single eval root):
    python evaluation/proteina/generation/scripts/backfill_designability_index.py \\
        --eval_root eval_output/inference_paper_inference_fid_60m_n128_paper_sweep_baseline_128_bs24_step400k_step_400000

Usage (sweep over many roots):
    python evaluation/proteina/generation/scripts/backfill_designability_index.py \\
        --glob 'eval_output/inference_paper_inference_fid_60m_*'

Idempotent: skips any eval root that already has `designability_index.csv`
unless `--force` is passed.
"""

from __future__ import annotations

import argparse
import glob as _glob
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `utils.*` resolves if needed later
# Make proteina src importable for load_pdb / rmsd_metric / compute_tm_score.
REPO_ROOT = HERE.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "proteina"))


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--eval_root", type=str, help="Single eval_output/<run> dir")
    g.add_argument(
        "--glob", type=str, help="Glob pattern matching multiple eval_output dirs"
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing designability_index.csv",
    )
    p.add_argument("--designable_threshold", type=float, default=2.0)
    return p.parse_args()


def _esm_plddt(esm_pdb_path: Path) -> float:
    """Recover mean pLDDT from a refold PDB's CA B-factor column.

    ESMFold writes per-atom pLDDT into b_factors (see
    designability.convert_outputs_to_pdb). Averaging over CA atoms reproduces
    the per-protein pLDDT that batch_designability records.
    """
    bs = []
    with open(esm_pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    bs.append(float(line[60:66]))
                except ValueError:
                    continue
    return float(sum(bs) / len(bs)) if bs else float("nan")


def _build_length_lookup(eval_root: Path) -> dict:
    """Inline reimplementation of evaluate.build_length_to_pdbs.

    Importing `evaluate` directly pulls torch_scatter (broken on some login
    nodes due to glibc CXXABI mismatch); the length parsing only needs torch
    + the batch_*.pt sidecars, so we duplicate the small loop here.
    """
    import re  # noqa: PLC0415
    import torch  # noqa: PLC0415

    tensors_dir = eval_root / "tensors"
    samples_dir = eval_root / "samples_fid"
    if not tensors_dir.is_dir() or not samples_dir.is_dir():
        return {}

    out: dict = {}
    pdb_idx = 0
    for tf in sorted(tensors_dir.glob("batch_*.pt")):
        m = re.search(r"_n(\d+)_x(\d+)\.pt$", tf.name)
        if not m:
            continue
        nres = int(m.group(1))
        try:
            saved = torch.load(tf, map_location="cpu", weights_only=False)
        except Exception:
            continue
        n_in_batch = saved.shape[0]
        for _ in range(n_in_batch):
            pdb_path = samples_dir / f"{pdb_idx}_fid.pdb"
            if pdb_path.exists():
                out[str(pdb_path)] = nres
            pdb_idx += 1
    return out


def _process_eval_root(eval_root: Path, force: bool, threshold: float) -> bool:
    samples_dir = eval_root / "samples_fid"
    tmp_dir = eval_root / "tmp_designability"
    out_csv = eval_root / "designability_index.csv"

    if not samples_dir.is_dir():
        print(f"  SKIP {eval_root.name}: no samples_fid/")
        return False
    if not tmp_dir.is_dir():
        print(f"  SKIP {eval_root.name}: no tmp_designability/")
        return False
    # Empty tmp_designability/ means designability was never run on this root
    # (or its artefacts were cleaned). Writing an all-NaN index would mislead
    # downstream consumers, so bail before doing any work.
    if not any(tmp_dir.iterdir()):
        print(f"  SKIP {eval_root.name}: tmp_designability/ is empty")
        return False
    if out_csv.exists() and not force:
        print(
            f"  SKIP {eval_root.name}: index already exists (use --force to overwrite)"
        )
        return False

    # Imports deferred so a missing optional dep on one root doesn't kill the loop.
    import torch  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from proteinfoundation.metrics.designability import load_pdb, rmsd_metric  # noqa: PLC0415
    from proteinfoundation.metrics.tm_score import compute_tm_score  # noqa: PLC0415

    length_lookup = _build_length_lookup(eval_root)
    all_pdbs = sorted(
        samples_dir.glob("*_fid.pdb"), key=lambda p: int(p.stem.split("_")[0])
    )
    if not all_pdbs:
        # samples_fid/ exists but is empty — placeholder /home dirs for runs
        # whose canonical artefacts live elsewhere (e.g. /rds counterpart).
        # Writing N=0 indices here is just clutter that confuses consumers.
        print(f"  SKIP {eval_root.name}: samples_fid/ has no *_fid.pdb")
        return False
    eval_subdirs = {d.name: d for d in tmp_dir.iterdir() if d.is_dir()}

    rows = []
    n_eval = n_des = n_failed = 0
    for i, pdb_path in enumerate(all_pdbs):
        rel = os.path.relpath(pdb_path, eval_root)
        length = length_lookup.get(str(pdb_path), "")
        name = pdb_path.stem  # e.g. "100_fid"
        sub = eval_subdirs.get(name)

        if sub is None:
            rows.append(
                {
                    "pdb_path": rel,
                    "length": length,
                    "evaluated": False,
                    "scRMSD": float("nan"),
                    "tm_score": float("nan"),
                    "plddt": float("nan"),
                    "designable": False,
                }
            )
            continue

        esm_paths = sorted(sub.glob("esm_*.pdb_esm"), key=lambda p: p.name)
        if not esm_paths:
            # MPNN/ESMFold raised before any refold landed → counts as failure
            rows.append(
                {
                    "pdb_path": rel,
                    "length": length,
                    "evaluated": True,
                    "scRMSD": float("nan"),
                    "tm_score": float("nan"),
                    "plddt": float("nan"),
                    "designable": False,
                }
            )
            n_eval += 1
            n_failed += 1
            continue

        try:
            gen_prot = load_pdb(str(pdb_path))
            gen_coors = torch.Tensor(gen_prot.atom_positions)
            best_rmsd, best_tm, best_plddt = math.inf, 0.0, float("nan")
            for ep in esm_paths:
                rec_prot = load_pdb(str(ep))
                rec_coors = torch.Tensor(rec_prot.atom_positions)
                n_common = min(len(gen_coors), len(rec_coors))
                rmsd = float(rmsd_metric(gen_coors[:n_common], rec_coors[:n_common]))
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_plddt = _esm_plddt(ep)
                tm = float(
                    compute_tm_score(
                        gen_coors[:n_common].numpy(), rec_coors[:n_common].numpy()
                    )
                )
                best_tm = max(best_tm, tm)
        except Exception as exc:
            print(f"    {name}: failed ({type(exc).__name__}: {exc})")
            rows.append(
                {
                    "pdb_path": rel,
                    "length": length,
                    "evaluated": True,
                    "scRMSD": float("nan"),
                    "tm_score": float("nan"),
                    "plddt": float("nan"),
                    "designable": False,
                }
            )
            n_eval += 1
            n_failed += 1
            continue

        is_des = best_rmsd < threshold
        rows.append(
            {
                "pdb_path": rel,
                "length": length,
                "evaluated": True,
                "scRMSD": best_rmsd,
                "tm_score": best_tm,
                "plddt": best_plddt,
                "designable": bool(is_des),
            }
        )
        n_eval += 1
        if is_des:
            n_des += 1
        if (i + 1) % 100 == 0:
            print(
                f"    {eval_root.name}: processed {i+1}/{len(all_pdbs)} (eval={n_eval} des={n_des})"
            )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(
        f"  WROTE {out_csv.name}: N={len(rows)}, evaluated={n_eval}, "
        f"designable={n_des}, failed={n_failed}"
    )
    return True


def main():
    args = parse_args()
    if args.eval_root:
        roots = [Path(args.eval_root).resolve()]
    else:
        roots = [Path(p).resolve() for p in sorted(_glob.glob(args.glob))]

    if not roots:
        print("No eval roots found.", file=sys.stderr)
        sys.exit(1)

    n_done = 0
    for root in roots:
        print(f"=== {root} ===")
        if _process_eval_root(root, args.force, args.designable_threshold):
            n_done += 1

    print(f"\nDone: {n_done}/{len(roots)} indices written.")


if __name__ == "__main__":
    main()
