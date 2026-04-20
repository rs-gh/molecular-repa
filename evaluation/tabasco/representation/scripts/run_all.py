# ruff: noqa: E402
"""Run all tabasco representation-quality probes across all sources.

Sources (13 total):
  Frozen encoders (3):
    - chemeleon (ChemPropEncoder pretrained='chemeleon')
    - mace      (MACEEncoder model_name='small')
    - dummy     (coord-MLP baseline, untrained)
  Trained checkpoints (7, at evaluation/tabasco/checkpoints/geom/*.ckpt):
    - baseline (no REPA)
    - chemeleon_additive_same
    - chemeleon_tradeoff_same
    - chemeleon_additive_fused
    - chemeleon_tradeoff_fused
    - mace_additive
    - mace_tradeoff

For each source we compute:
  P3 — atom-type classification (per-atom linear probe)
  P4 — RDKit-descriptor regression (per-molecule ridge probe, 4 targets)

Writes:
  evaluation/tabasco/representation/results/results.json     — full results
  evaluation/tabasco/representation/results/results.md       — pretty markdown table

Usage:
  source .venv/bin/activate
  export PROJECT_ROOT=$(pwd)/src/tabasco
  python playground/tabasco/probes/run_all.py --n_mols 1000
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List

import torch

HERE = Path(__file__).resolve().parent
# HERE        = .../representation/scripts
# HERE.parent = .../representation (contains the lib/ module dir)
# Add lib/ to sys.path so the flat `from utils/atom_type/descriptors import X`
# form (same as the legacy playground layout) still works.
sys.path.insert(0, str(HERE.parent / "lib"))

from utils import (
    DESCRIPTOR_NAMES,
    PROJECT_ROOT,
    _default_device,
    build_batch,
    extract_encoder_embeddings,
    extract_hidden_states,
    load_molecules,
)
from atom_type import run_atom_type_probe
from descriptors import run_descriptor_probe

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


# --------------------------------------------------------------------------- #
# Source registry
# --------------------------------------------------------------------------- #


def _build_chemeleon_encoder():
    from tabasco.models.components.encoders import ChemPropEncoder

    enc = ChemPropEncoder(pretrained="chemeleon")
    enc.eval()
    return enc.to(_default_device())


def _build_mace_encoder():
    from tabasco.models.components.encoders import MACEEncoder

    enc = MACEEncoder(model_name="small")
    enc.eval()
    return enc.to(_default_device())


def _build_dummy_encoder():
    from tabasco.models.components.encoders import DummyEncoder

    enc = DummyEncoder(encoder_dim=256)
    enc.eval()
    return enc.to(_default_device())


def _load_tabasco_checkpoint(path: str):
    # Reuse the evaluation loader so we match production inference exactly.
    # parents[4] = repo root (scripts → representation → tabasco → evaluation
    #              → molecular-repa). Points at the generation-side eval
    #              scripts (moved in the parallel generation commit).
    eval_scripts = (
        Path(__file__).resolve().parents[4] / "evaluation" / "tabasco" / "scripts"
    )
    sys.path.insert(0, str(eval_scripts))
    from evaluate import load_checkpoint  # noqa: E402

    model, _, meta = load_checkpoint(path, device=_default_device())
    model.eval()
    return model, meta


ENCODER_SOURCES = {
    "chemeleon_frozen": _build_chemeleon_encoder,
    "mace_frozen": _build_mace_encoder,
    "dummy_frozen": _build_dummy_encoder,
}

CKPT_DIR = (
    # parents[4] = repo root; evaluation/tabasco/checkpoints/geom/ still holds
    # the model checkpoints (caches, not reorganized by this PR).
    Path(__file__).resolve().parents[4]
    / "evaluation"
    / "tabasco"
    / "checkpoints"
    / "geom"
)
CHECKPOINT_SOURCES = [
    "baseline",
    "chemeleon_additive_same",
    "chemeleon_tradeoff_same",
    "chemeleon_additive_fused",
    "chemeleon_tradeoff_fused",
    "mace_additive",
    "mace_tradeoff",
]


# --------------------------------------------------------------------------- #
# Per-source probing
# --------------------------------------------------------------------------- #


def probe_source(
    name: str,
    rep_fn: Callable[[], torch.Tensor],
    mols,
    padding_mask: torch.Tensor,
    max_atoms: int,
) -> Dict:
    print(f"\n=== Source: {name} ===")
    t0 = time.time()
    reps = rep_fn()  # [B, N, D]
    t_extract = time.time() - t0
    D = reps.shape[-1]
    print(f"  reps shape={tuple(reps.shape)}  extract={t_extract:.1f}s")

    t0 = time.time()
    atom = run_atom_type_probe(reps, mols, max_atoms=max_atoms)
    print(f"  atom-type: acc={atom.accuracy:.4f}  f1={atom.macro_f1:.4f}")

    desc = run_descriptor_probe(reps, mols, padding_mask)
    for r in desc:
        print(f"  {r.target:20s}: r={r.pearson:+.3f}  R²={r.r2:+.3f}")
    t_probe = time.time() - t0
    print(f"  probes: {t_probe:.1f}s")

    return {
        "source": name,
        "dim": int(D),
        "atom_type": asdict(atom),
        "descriptors": [asdict(r) for r in desc],
        "extract_seconds": t_extract,
        "probe_seconds": t_probe,
    }


# --------------------------------------------------------------------------- #
# Pretty output
# --------------------------------------------------------------------------- #


def _markdown_table(results: List[Dict]) -> str:
    lines = []
    lines.append("# Tabasco Representation Quality Probes\n")
    lines.append(
        "Per-atom atom-type classification (P3) and per-molecule descriptor regression (P4).\n"
    )
    lines.append("## P3 — Atom-type classification\n")
    lines.append("| source | dim | accuracy | macro-F1 | n_train | n_test | classes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        a = r["atom_type"]
        lines.append(
            f"| {r['source']} | {r['dim']} | {a['accuracy']:.4f} | {a['macro_f1']:.4f} | "
            f"{a['n_train']} | {a['n_test']} | {a['n_classes']} |"
        )

    lines.append("\n## P4 — RDKit descriptor regression (Pearson r / R²)\n")
    header_cols = "| source |" + " |".join(f" {d} " for d in DESCRIPTOR_NAMES) + " |"
    sep_cols = "|---|" + "|".join(["---:"] * len(DESCRIPTOR_NAMES)) + "|"
    lines.append(header_cols)
    lines.append(sep_cols)
    for r in results:
        cells = []
        for rr in r["descriptors"]:
            cells.append(f"{rr['pearson']:+.3f}/{rr['r2']:+.3f}")
        lines.append(f"| {r['source']} | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_mols", type=int, default=1000)
    ap.add_argument("--dataset", type=str, default="geom", choices=["geom", "qm9"])
    ap.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"]
    )
    ap.add_argument(
        "--lmdb_path",
        type=str,
        default=None,
        help="Explicit LMDB path (bypasses Lustre path probing).",
    )
    ap.add_argument("--skip_encoders", action="store_true")
    ap.add_argument("--skip_checkpoints", action="store_true")
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of sources to run (subset).",
    )
    # Default writes to .../representation/results/ (sibling of scripts/).
    ap.add_argument("--output_dir", type=str, default=str(HERE.parent / "results"))
    args = ap.parse_args()

    # --- Load one shared batch ---
    mols = load_molecules(
        args.n_mols,
        dataset=args.dataset,
        split=args.split,
        lmdb_path=args.lmdb_path,
    )
    max_atoms = max(m.GetNumAtoms() for m in mols)
    print(f"Loaded {len(mols)} mols. max_atoms={max_atoms}")
    batch = build_batch(mols, max_atoms=max_atoms, device=_default_device())
    pad_cpu = batch["padding_mask"].cpu()

    # --- Build source list ---
    only = set(args.only.split(",")) if args.only else None
    sources = []
    if not args.skip_encoders:
        for name, builder in ENCODER_SOURCES.items():
            if only and name not in only:
                continue
            sources.append(("encoder", name, builder))
    if not args.skip_checkpoints:
        for name in CHECKPOINT_SOURCES:
            if only and name not in only:
                continue
            sources.append(("checkpoint", name, CKPT_DIR / f"{name}.ckpt"))

    # --- Probe each ---
    results = []
    for kind, name, payload in sources:
        try:
            if kind == "encoder":
                enc = payload()

                def rep_fn(enc=enc, batch=batch):
                    return extract_encoder_embeddings(enc, batch)

                out = probe_source(name, rep_fn, mols, pad_cpu, max_atoms)
                del enc
            else:
                ckpt_path = payload
                if not ckpt_path.exists():
                    print(f"  ⚠ skip {name}: {ckpt_path} not found")
                    continue
                model, meta = _load_tabasco_checkpoint(str(ckpt_path))

                def rep_fn(model=model, batch=batch):
                    return extract_hidden_states(model, batch)

                out = probe_source(f"ckpt:{name}", rep_fn, mols, pad_cpu, max_atoms)
                out["meta"] = meta
                del model
            results.append(out)
        except Exception as e:
            print(f"  ❌ {name} failed: {type(e).__name__}: {e}")
            results.append({"source": name, "error": f"{type(e).__name__}: {e}"})
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Write outputs ---
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "results.json"
    md_path = outdir / "results.md"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    md_path.write_text(_markdown_table([r for r in results if "error" not in r]))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
