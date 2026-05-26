"""Extract per-residue features for every model row and encoder column.

Reads the frozen batch from ``build_batch.py`` and dumps:
  - 3 model feature files (baseline, repa_l4, repa_l9), each a dict
    {layer_idx: Tensor[n_real, 512]} over 10 transformer layers.
  - 3 encoder feature files (gearnet, mpnn, esm2), each a Tensor[n_real, D].

All features are flattened to per-residue rows using the SAME mask-derived
ordering captured in ``frozen_batch_n256.pt``, so any pair of files can be
compared directly via CKNNA.

Idempotent: skips any output that already exists. Re-run after a partial
crash and it will pick up where it left off.

Run:
    source .venv/bin/activate
    export PROJECT_ROOT=$(pwd)/src/proteina
    python evaluation/proteina/alignment/scripts/extract_features.py
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ALIGN_ROOT = HERE.parent
REP_ROOT = ALIGN_ROOT.parent / "representation"
PROTEINA_ROOT = Path("/home/sr2173/git/molecular-repa/src/proteina")

# `from lib import ...` resolves against representation/, matching the
# existing run_sweep.py convention. That `lib` package re-exports
# checkpoint helpers + extract utilities + the data loader.
for p in (str(REP_ROOT), str(PROTEINA_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Existing helpers from the representation pipeline — read-only reuse.
from lib import (  # noqa: E402
    _default_device,
    enable_hidden_states,
    extract_gearnet_embeddings,
    extract_model_hidden_states_multilayer,
    find_checkpoint_path,
    load_checkpoint_by_path,
    model_num_layers,
)

# ── Config ──────────────────────────────────────────────────────────────────

T_VALUE = 1.0  # CLEAN structure (proteina convention: t=1.0=clean, t=0.0=noise);
# matches the representation-probe pipeline default in extract.py
CHUNK_SIZE = 16

# Model rows: (display_name, run_dir, step, is_repa). All from the n=256 PDB
# convergence sweep. step=1000k is the latest common snapshot where every row
# has an EMA checkpoint on disk (limited by repa_gearnet_l4 which maxes at 1000k).
MODEL_ROWS = [
    (
        "baseline",
        "proteina_60m_baseline_256_bs24_2gpu",
        1000000,
        False,
    ),
    (
        "repa_gearnet_l4",
        "proteina_60m_repa_l4_256_per_residue_bs24_2gpu",
        1000000,
        True,
    ),
    (
        "repa_gearnet_l9",
        "proteina_60m_repa_l9_256_per_residue_bs24_2gpu",
        1000000,
        True,
    ),
    (
        "repa_mpnn_l4",
        "proteina_60m_repa_mpnn_l4_256_per_residue",
        1000000,
        True,
    ),
    (
        "repa_mpnn_l9",
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        1000000,
        True,
    ),
]

# Encoder columns: name → builder. All return frozen, eval-mode encoders on CUDA.
GEARNET_CKPT = os.environ.get(
    "GEARNET_CKPT_PATH",
    "/rds/user/sr2173/hpc-work/proteina/data/metric_factory/model_weights/gearnet_ca.pth",
)
MPNN_CKPT = os.environ.get(
    "MPNN_CKPT_PATH",
    "/rds/user/sr2173/hpc-work/proteina/ProteinMPNN/ca_model_weights/v_48_020.pt",
)
ESM_MODEL_ID = os.environ.get("ESM_MODEL_ID", "facebook/esm2_t33_650M_UR50D")

OUT_DIR = ALIGN_ROOT / "results"
BATCH_PATH = OUT_DIR / "frozen_batch_n256.pt"
MODEL_OUT = OUT_DIR / "model_features"
ENC_OUT = OUT_DIR / "encoder_features"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _flatten_with_mask(x: torch.Tensor, residue_index: torch.Tensor) -> torch.Tensor:
    """Flatten a [B, N, D] tensor to [n_real, D] using a pre-computed [n_real, 2] index.

    Using a precomputed index (instead of x[mask]) makes the row order match
    bit-exactly across all extraction runs without re-deriving it from mask.
    """
    bi, rj = residue_index[:, 0], residue_index[:, 1]
    return x[bi, rj].contiguous()


def _move_batch(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def extract_one_model(
    row_name: str,
    run_dir: str,
    step: int,
    is_repa: bool,
    batch: dict,
    residue_index: torch.Tensor,
    device: str,
) -> None:
    out_path = MODEL_OUT / f"{row_name}.pt"
    if out_path.exists():
        print(f"[{row_name}] cached at {out_path}, skip")
        return

    ckpt_path = find_checkpoint_path(run_dir, step, prefer_ema=True)
    if ckpt_path is None:
        raise FileNotFoundError(f"No EMA checkpoint found for {run_dir} @ step={step}")
    print(f"[{row_name}] loading {ckpt_path}")
    model = load_checkpoint_by_path(str(ckpt_path), is_repa=is_repa, device=device)

    n_layers = model_num_layers(model)
    layers = list(range(n_layers))
    enable_hidden_states(model, layers)
    print(f"[{row_name}] running forward (t={T_VALUE}, layers={layers})")
    per_layer = extract_model_hidden_states_multilayer(
        model, batch, layers, chunk_size=CHUNK_SIZE, t_value=T_VALUE
    )

    flat: dict = {}
    for lyr, h in per_layer.items():
        # h: [B, N, D] on CPU
        flat[lyr] = _flatten_with_mask(h, residue_index).cpu()
        print(
            f"[{row_name}] layer {lyr}: {tuple(flat[lyr].shape)}  "
            f"(mean={flat[lyr].mean():.3f}, std={flat[lyr].std():.3f})"
        )

    payload = {
        "row_name": row_name,
        "run_dir": run_dir,
        "step": step,
        "is_repa": is_repa,
        "ckpt_path": str(ckpt_path),
        "t_value": T_VALUE,
        "per_layer": flat,
    }
    torch.save(payload, out_path)
    print(f"[{row_name}] wrote {out_path}")

    del model, per_layer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_gearnet(batch: dict, residue_index: torch.Tensor, device: str) -> None:
    out_path = ENC_OUT / "gearnet.pt"
    if out_path.exists():
        print(f"[gearnet] cached at {out_path}, skip")
        return
    from proteinfoundation.repa.gearnet_encoder import GearNetPerResidueEncoder

    print(f"[gearnet] loading from {GEARNET_CKPT}")
    enc = GearNetPerResidueEncoder(ckpt_path=GEARNET_CKPT)
    enc.eval().to(device)

    print("[gearnet] forwarding")
    reps = extract_gearnet_embeddings(enc, batch, chunk_size=CHUNK_SIZE)  # [B, N, 512]
    flat = _flatten_with_mask(reps, residue_index).cpu()
    print(
        f"[gearnet] flat shape {tuple(flat.shape)}  (mean={flat.mean():.3f}, std={flat.std():.3f})"
    )

    torch.save(
        {
            "encoder": "gearnet",
            "ckpt_path": GEARNET_CKPT,
            "features": flat,
            "dim": int(flat.shape[-1]),
        },
        out_path,
    )
    print(f"[gearnet] wrote {out_path}")

    del enc, reps
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_mpnn(batch: dict, residue_index: torch.Tensor, device: str) -> None:
    out_path = ENC_OUT / "mpnn.pt"
    if out_path.exists():
        print(f"[mpnn] cached at {out_path}, skip")
        return
    from proteinfoundation.repa.mpnn_encoder import ProteinMPNNPerResidueEncoder
    from proteinfoundation.utils.coors_utils import ang_to_nm

    print(f"[mpnn] loading from {MPNN_CKPT}")
    enc = ProteinMPNNPerResidueEncoder(ckpt_path=MPNN_CKPT)
    enc.eval().to(device)

    # MPNN encoder signature: forward(ca_coords_nm, mask, residue_type=None)
    print("[mpnn] forwarding")
    outs = []
    B = batch["coords"].shape[0]
    for s in range(0, B, CHUNK_SIZE):
        e = min(s + CHUNK_SIZE, B)
        ca_nm = ang_to_nm(batch["coords"][s:e, :, 1, :].to(device))
        mask = batch["mask"][s:e].to(device).bool()
        out = enc(ca_nm, mask)
        outs.append(out.detach().cpu())
    reps = torch.cat(outs, dim=0)  # [B, N, 128]
    flat = _flatten_with_mask(reps, residue_index).cpu()
    print(
        f"[mpnn] flat shape {tuple(flat.shape)}  (mean={flat.mean():.3f}, std={flat.std():.3f})"
    )

    torch.save(
        {
            "encoder": "mpnn",
            "ckpt_path": MPNN_CKPT,
            "features": flat,
            "dim": int(flat.shape[-1]),
        },
        out_path,
    )
    print(f"[mpnn] wrote {out_path}")

    del enc, reps
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_esm(batch: dict, residue_index: torch.Tensor, device: str) -> None:
    out_path = ENC_OUT / "esm2.pt"
    if out_path.exists():
        print(f"[esm2] cached at {out_path}, skip")
        return
    from proteinfoundation.repa.esm_encoder import ESMPerResidueEncoder

    print(f"[esm2] loading {ESM_MODEL_ID}")
    enc = ESMPerResidueEncoder(model_id=ESM_MODEL_ID)
    enc.eval().to(device)

    # ESM signature: forward(ca_coords_nm, mask, residue_type=None) — requires residue_type
    if "residue_type" not in batch:
        raise RuntimeError(
            "Batch is missing 'residue_type'; ESM encoder needs sequence input."
        )
    print("[esm2] forwarding")
    outs = []
    B = batch["coords"].shape[0]
    # ESM-650M on a single batch of 64×256 may OOM — chunk smaller.
    esm_chunk = 4
    for s in range(0, B, esm_chunk):
        e = min(s + esm_chunk, B)
        ca_nm = batch["coords"][s:e, :, 1, :].to(
            device
        )  # nm conversion not used by ESM
        mask = batch["mask"][s:e].to(device).bool()
        res_type = batch["residue_type"][s:e].to(device).long()
        out = enc(ca_nm, mask, residue_type=res_type)  # [b, n, hidden_size]
        outs.append(out.detach().cpu())
    reps = torch.cat(outs, dim=0)
    flat = _flatten_with_mask(reps, residue_index).cpu()
    print(
        f"[esm2] flat shape {tuple(flat.shape)}  (mean={flat.mean():.3f}, std={flat.std():.3f})"
    )

    torch.save(
        {
            "encoder": "esm2",
            "model_id": ESM_MODEL_ID,
            "features": flat,
            "dim": int(flat.shape[-1]),
        },
        out_path,
    )
    print(f"[esm2] wrote {out_path}")

    del enc, reps
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    ENC_OUT.mkdir(parents=True, exist_ok=True)

    if not BATCH_PATH.exists():
        raise FileNotFoundError(f"Run build_batch.py first; missing {BATCH_PATH}")
    print(f"Loading frozen batch from {BATCH_PATH}")
    payload = torch.load(BATCH_PATH, map_location="cpu", weights_only=False)
    batch_cpu = payload["batch"]
    residue_index = payload["residue_index"]
    n_real = payload["n_real_residues"]
    print(f"  n_real_residues={n_real}, n_proteins={payload['n_proteins']}")

    device = _default_device()
    batch = _move_batch(batch_cpu, device)

    # Encoders first (smaller memory footprint, fast fail if a checkpoint is missing).
    extract_gearnet(batch, residue_index, device)
    extract_mpnn(batch, residue_index, device)
    extract_esm(batch, residue_index, device)

    # Proteina models last (largest GPU footprint, slowest to load).
    for row_name, run_dir, step, is_repa in MODEL_ROWS:
        extract_one_model(
            row_name, run_dir, step, is_repa, batch, residue_index, device
        )

    print("\nDone. Feature files:")
    for p in sorted((MODEL_OUT.iterdir())):
        print(f"  {p}")
    for p in sorted((ENC_OUT.iterdir())):
        print(f"  {p}")


if __name__ == "__main__":
    main()
