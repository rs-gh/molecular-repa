"""Compute two CKNNA matrices from cached features: per-residue and per-protein.

For each (model row, model layer, encoder column) cell, computes CKNNA in
both modes on the same 10,000 samples (residues or proteins respectively).

Writes:
    cknna_matrix_per_residue.jsonl
    cknna_matrix_per_protein.jsonl

Each line: {model, layer, encoder, cknna, lo5, hi95, median, n, mode, ...}

Run:
    source .venv/bin/activate
    export PROJECT_ROOT=$(pwd)/src/proteina
    python evaluation/proteina/alignment/scripts/run_cknna.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ALIGN_ROOT = HERE.parent
LIB_ROOT = (
    ALIGN_ROOT  # so `from lib.cknna import ...` resolves to alignment/lib/cknna.py
)
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))

from lib.cknna import cknna_bootstrap  # noqa: E402

OUT_DIR = ALIGN_ROOT / "results"
MODEL_OUT = OUT_DIR / "model_features"
ENC_OUT = OUT_DIR / "encoder_features"

K_NEIGHBORS = int(os.environ.get("CKNNA_K", 10))
N_BOOT = int(os.environ.get("CKNNA_N_BOOT", 50))

# k=10 keeps the canonical (untagged) filenames the plotting scripts read.
# Any other k writes a k-tagged sidecar so a sweep never clobbers the default.
_K_TAG = "" if K_NEIGHBORS == 10 else f"_k{K_NEIGHBORS}"
MATRIX_PATHS = {
    "per_residue": OUT_DIR / f"cknna_matrix_per_residue{_K_TAG}.jsonl",
    "per_protein": OUT_DIR / f"cknna_matrix_per_protein{_K_TAG}.jsonl",
}
MODEL_ROWS = [
    "baseline",
    "repa_gearnet_l4",
    "repa_gearnet_l9",
    "repa_mpnn_l4",
    "repa_mpnn_l9",
]
ENCODER_COLS = ["gearnet", "mpnn", "esm2"]


def _compute_matrix(mode: str, device: str) -> None:
    """Compute the per-residue OR per-protein CKNNA matrix and write JSONL."""
    suffix = f"_{mode}.pt"
    print(f"\n#### CKNNA matrix: {mode} ####")

    # Load encoder features for this mode.
    encoders: dict = {}
    for enc in ENCODER_COLS:
        path = ENC_OUT / f"{enc}{suffix}"
        if not path.exists():
            raise FileNotFoundError(f"Missing encoder features: {path}")
        encoders[enc] = torch.load(path, map_location="cpu", weights_only=False)
        print(f"  {enc}: features {tuple(encoders[enc]['features'].shape)}")

    n_samples = encoders[ENCODER_COLS[0]]["features"].shape[0]
    for enc in ENCODER_COLS:
        if encoders[enc]["features"].shape[0] != n_samples:
            raise RuntimeError(
                f"{enc} row-count mismatch in {mode}: "
                f"{encoders[enc]['features'].shape[0]} vs expected {n_samples}"
            )

    matrix_path = MATRIX_PATHS[mode]
    with open(matrix_path, "w") as f:
        for row in MODEL_ROWS:
            path = MODEL_OUT / f"{row}{suffix}"
            if not path.exists():
                print(f"  [warn] missing {path}, skip row")
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            per_layer = payload["per_layer"]
            if list(per_layer.values())[0].shape[0] != n_samples:
                raise RuntimeError(
                    f"{row} row-count mismatch in {mode}: "
                    f"{list(per_layer.values())[0].shape[0]} vs expected {n_samples}"
                )
            print(f"  === {row}  (step={payload['step']}, t={payload['t_value']}) ===")
            for layer in sorted(per_layer.keys()):
                phi = per_layer[layer].to(device)
                for enc in ENCODER_COLS:
                    psi = encoders[enc]["features"].to(device)
                    t0 = time.time()
                    out = cknna_bootstrap(
                        phi, psi, k=K_NEIGHBORS, n_boot=N_BOOT, seed=0
                    )
                    dt = time.time() - t0
                    rec = {
                        "mode": mode,
                        "model": row,
                        "layer": int(layer),
                        "encoder": enc,
                        "cknna": out["point"],
                        "lo5": out["lo5"],
                        "hi95": out["hi95"],
                        "median": out["median"],
                        "n_samples": int(n_samples),
                        "k": K_NEIGHBORS,
                        "n_boot": N_BOOT,
                        "t_value": payload["t_value"],
                        "step": payload["step"],
                    }
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(
                        f"    L{layer:2d} × {enc:7s}: CKNNA={out['point']:.4f}  "
                        f"[{out['lo5']:.4f}, {out['hi95']:.4f}]  ({dt:.1f}s)"
                    )
            del per_layer, payload
    print(f"  wrote {matrix_path}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"k={K_NEIGHBORS}, n_boot={N_BOOT}")
    print(
        f"Output: {MATRIX_PATHS['per_residue'].name}, {MATRIX_PATHS['per_protein'].name}"
    )
    _compute_matrix("per_protein", device)  # cheaper (smaller N at sample level)
    _compute_matrix("per_residue", device)


if __name__ == "__main__":
    main()
