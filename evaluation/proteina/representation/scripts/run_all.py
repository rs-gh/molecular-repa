# ruff: noqa: E402
"""Run all proteina representation-quality probes across all sources.

Sources:
  Frozen encoder:
    - gearnet      (NoTrainCAGearNet, per-residue 512-d)
  Trained checkpoints:
    - baseline              (no REPA)
    - repa_l0_per_residue   (REPA at layer 0)
    - repa_l4_per_residue   (REPA at layer 4)
    - repa_l9_per_residue   (REPA at layer 9)

Probes per source:
  P1 — contact prediction P@L/5 (long-range, 8 Å, |i-j|≥24)
  P2 — CATH fold classification (T-level, fallback A/C if too sparse)

Usage:
  source .venv/bin/activate
  export PROJECT_ROOT=$(pwd)/src/proteina
  export DATA_PATH=/rds/user/sr2173/hpc-work/proteina/data
  python playground/proteina/probes/run_all.py --n_proteins 200
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
from typing import Dict, List

import torch

HERE = Path(__file__).resolve().parent
# HERE        = .../representation/scripts (for utils/contact/cath shims)
# HERE.parent = .../representation         (so `from lib import ...` works)
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from utils import (
    CHECKPOINT_REGISTRY,
    _default_device,
    extract_gearnet_embeddings,
    extract_model_hidden_states,
    load_checkpoint,
    load_proteina_batch,
)
from contact import run_contact_probe
from cath import run_cath_probe


GEARNET_CKPT = os.environ.get(
    "GEARNET_CKPT_PATH",
    "/rds/user/sr2173/hpc-work/proteina/data/metric_factory/model_weights/gearnet_ca.pth",
)


# --------------------------------------------------------------------------- #
# Source builders
# --------------------------------------------------------------------------- #


def _load_gearnet():
    from proteinfoundation.repa.gearnet_encoder import GearNetPerResidueEncoder

    enc = GearNetPerResidueEncoder(ckpt_path=GEARNET_CKPT)
    enc.eval()
    return enc.to(_default_device())


# --------------------------------------------------------------------------- #
# Per-source probing
# --------------------------------------------------------------------------- #


def probe_source(
    name: str, reps: torch.Tensor, batch: Dict, raw, probe_cath_level: str
) -> Dict:
    print(f"\n=== Source: {name} ===  reps={tuple(reps.shape)}")
    t0 = time.time()
    contact = run_contact_probe(reps, batch)
    print(
        f"  P1 contact: P@L={contact.p_at_L:.3f}  P@L/2={contact.p_at_L_2:.3f}  "
        f"P@L/5={contact.p_at_L_5:.3f}  (n_test={contact.n_proteins_test})"
    )
    cath = run_cath_probe(reps, batch["mask"], raw, preferred_level=probe_cath_level)
    print(
        f"  P2 CATH-{cath.level}: acc={cath.accuracy:.3f}  f1={cath.macro_f1:.3f}  "
        f"classes={cath.n_classes}  (n_train={cath.n_train}, n_test={cath.n_test})"
    )
    dt = time.time() - t0
    print(f"  probes: {dt:.1f}s")
    return {
        "source": name,
        "dim": int(reps.shape[-1]),
        "contact": asdict(contact),
        "cath": asdict(cath),
    }


def _markdown_table(results: List[Dict]) -> str:
    lines = []
    lines.append("# Proteina Representation Quality Probes\n")
    lines.append(
        "Long-range contact prediction (P1) and CATH fold classification (P2).\n"
    )
    lines.append("## P1 — Contact prediction (P@L/k)\n")
    lines.append("| source | dim | P@L | P@L/2 | P@L/5 | n_test |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        c = r["contact"]
        lines.append(
            f"| {r['source']} | {r['dim']} | {c['p_at_L']:.3f} | {c['p_at_L_2']:.3f} | "
            f"{c['p_at_L_5']:.3f} | {c['n_proteins_test']} |"
        )
    lines.append("\n## P2 — CATH classification\n")
    lines.append(
        "| source | level | classes | accuracy | macro-F1 | n_train | n_test |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        c = r["cath"]
        lines.append(
            f"| {r['source']} | {c['level']} | {c['n_classes']} | {c['accuracy']:.3f} | "
            f"{c['macro_f1']:.3f} | {c['n_train']} | {c['n_test']} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_proteins", type=int, default=200)
    ap.add_argument("--max_size", type=int, default=256)
    ap.add_argument(
        "--only", type=str, default=None, help="Comma-separated list of sources to run."
    )
    ap.add_argument("--skip_gearnet", action="store_true")
    ap.add_argument("--skip_checkpoints", action="store_true")
    ap.add_argument("--cath_level", type=str, default="T", choices=["C", "A", "T"])
    # Default writes to .../representation/results/ (sibling of scripts/).
    ap.add_argument("--output_dir", type=str, default=str(HERE.parent / "results"))
    args = ap.parse_args()

    device = _default_device()

    # --- Load shared batch ---
    print(f"Loading {args.n_proteins} proteins (≤ {args.max_size} residues)...")
    batch, raw = load_proteina_batch(
        n=args.n_proteins, max_size=args.max_size, device=device
    )
    print(f"  Loaded {len(raw)} proteins, mask sum = {int(batch['mask'].sum().item())}")

    only = set(args.only.split(",")) if args.only else None

    sources = []
    if not args.skip_gearnet and (only is None or "gearnet" in only):
        sources.append(("encoder", "gearnet"))
    if not args.skip_checkpoints:
        for name in CHECKPOINT_REGISTRY:
            if only is None or name in only:
                sources.append(("checkpoint", name))

    results = []
    for kind, name in sources:
        try:
            if kind == "encoder":
                enc = _load_gearnet()
                reps = extract_gearnet_embeddings(enc, batch)
                out = probe_source(name, reps, batch, raw, args.cath_level)
                del enc
            else:
                model, meta = load_checkpoint(name, device=device)
                layer = meta["repa_layers"][0] if meta["repa_layers"] else 4
                reps = extract_model_hidden_states(model, batch, layer=layer)
                out = probe_source(f"ckpt:{name}", reps, batch, raw, args.cath_level)
                out["meta"] = {
                    "layer": layer,
                    **{k: v for k, v in meta.items() if k != "repa_layers"},
                }
                del model
            results.append(out)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  ❌ {name} failed: {type(e).__name__}: {e}")
            results.append({"source": name, "error": f"{type(e).__name__}: {e}"})
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
