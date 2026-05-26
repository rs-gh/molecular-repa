"""Compute the CKNNA matrix from cached features.

For each (model row, model layer, encoder column) cell:
  cknna(model_features[layer], encoder_features) on the frozen 10k residues.

Writes one row per cell to ``cknna_matrix.jsonl``:
  {model, layer, encoder, cknna, lo5, hi95, median, n}

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
MATRIX_PATH = OUT_DIR / "cknna_matrix.jsonl"

K_NEIGHBORS = int(os.environ.get("CKNNA_K", 10))
N_BOOT = int(os.environ.get("CKNNA_N_BOOT", 50))
MODEL_ROWS = [
    "baseline",
    "repa_gearnet_l4",
    "repa_gearnet_l9",
    "repa_mpnn_l4",
    "repa_mpnn_l9",
]
ENCODER_COLS = ["gearnet", "mpnn", "esm2"]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load encoders
    encoders: dict = {}
    for enc_name in ENCODER_COLS:
        path = ENC_OUT / f"{enc_name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing encoder features: {path}")
        encoders[enc_name] = torch.load(path, map_location="cpu", weights_only=False)
        print(
            f"Loaded {enc_name}: features {tuple(encoders[enc_name]['features'].shape)}"
        )

    n_real = encoders[ENCODER_COLS[0]]["features"].shape[0]
    for enc_name in ENCODER_COLS:
        if encoders[enc_name]["features"].shape[0] != n_real:
            raise RuntimeError(
                f"Row-count mismatch: {enc_name} has "
                f"{encoders[enc_name]['features'].shape[0]} rows, expected {n_real}"
            )

    # Compute the matrix
    with open(MATRIX_PATH, "w") as f:
        for row in MODEL_ROWS:
            path = MODEL_OUT / f"{row}.pt"
            if not path.exists():
                print(f"[warn] missing {path}, skip row")
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            per_layer = payload["per_layer"]
            if list(per_layer.values())[0].shape[0] != n_real:
                raise RuntimeError(
                    f"{row} row-count mismatch: got {list(per_layer.values())[0].shape[0]}, "
                    f"expected {n_real}"
                )
            print(f"\n=== {row}  (step={payload['step']}, t={payload['t_value']}) ===")
            for layer in sorted(per_layer.keys()):
                phi = per_layer[layer].to(device)
                for enc_name in ENCODER_COLS:
                    psi = encoders[enc_name]["features"].to(device)
                    t0 = time.time()
                    out = cknna_bootstrap(
                        phi, psi, k=K_NEIGHBORS, n_boot=N_BOOT, seed=0
                    )
                    dt = time.time() - t0
                    rec = {
                        "model": row,
                        "layer": int(layer),
                        "encoder": enc_name,
                        "cknna": out["point"],
                        "lo5": out["lo5"],
                        "hi95": out["hi95"],
                        "median": out["median"],
                        "n_residues": int(n_real),
                        "k": K_NEIGHBORS,
                        "n_boot": N_BOOT,
                        "t_value": payload["t_value"],
                        "step": payload["step"],
                    }
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(
                        f"  L{layer:2d} × {enc_name:7s}: "
                        f"CKNNA={out['point']:.4f}  "
                        f"[{out['lo5']:.4f}, {out['hi95']:.4f}]  "
                        f"({dt:.1f}s)"
                    )
            # free phi between rows
            del per_layer, payload

    print(f"\nWrote {MATRIX_PATH}")


if __name__ == "__main__":
    main()
