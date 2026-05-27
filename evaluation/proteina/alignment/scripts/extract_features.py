"""Extract per-residue + per-protein features for every model row and encoder.

Reads the frozen-batch manifest (LMDB keys + per-residue subsample) from
``build_batch.py`` and dumps, for each source:

  ``{name}_per_protein.pt``  — mean-pooled over real residues, shape (10000, D).
                                Used for the CATH-like per-protein CKNNA matrix.
  ``{name}_per_residue.pt``  — features for the 10,000 uniformly-sampled
                                residues. Used for the REPA-paper-style
                                per-residue CKNNA matrix.

Memory strategy: never holds a full [10000, 256, D] padded tensor. Loads
proteins in shards of SHARD_SIZE (default 500), forwards through the model,
immediately reduces to (a) mean-pool per protein and (b) the residue rows
that fall inside this shard. Drops the padded shard tensor before moving on.

Idempotent: skips any output that already exists.

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
from typing import Dict, List

import torch

HERE = Path(__file__).resolve().parent
ALIGN_ROOT = HERE.parent
REP_ROOT = ALIGN_ROOT.parent / "representation"
PROTEINA_ROOT = Path("/home/sr2173/git/molecular-repa/src/proteina")

# `from lib import ...` resolves against representation/, matching the
# existing run_sweep.py convention.
for p in (str(REP_ROOT), str(PROTEINA_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib import (  # noqa: E402
    _default_device,
    enable_hidden_states,
    extract_gearnet_embeddings,
    extract_model_hidden_states_multilayer,
    find_checkpoint_path,
    load_checkpoint_by_path,
    model_num_layers,
)
from lib.manifest import load_proteina_batch_from_manifest  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────

T_VALUE = 1.0  # CLEAN structure (proteina convention: t=1.0=clean, t=0.0=noise)
SHARD_SIZE = 500  # proteins loaded per shard; reduces peak RAM
PROTEINA_CHUNK = 32  # inner forward chunk for proteina (was 16)
ESM_CHUNK = 32  # inner forward chunk for ESM2 (was 4)
ENCODER_CHUNK = 32  # inner forward chunk for GearNet / MPNN (was 16)

# Model rows: (display_name, run_dir, step, is_repa). All from the n=256 PDB
# convergence sweep. step=1000k is the latest common snapshot across all five
# rows (repa_gearnet_l4_bs24_2gpu maxes at 1000k).
MODEL_ROWS = [
    ("baseline", "proteina_60m_baseline_256_bs24_2gpu", 1_000_000, False),
    (
        "repa_gearnet_l4",
        "proteina_60m_repa_l4_256_per_residue_bs24_2gpu",
        1_000_000,
        True,
    ),
    (
        "repa_gearnet_l9",
        "proteina_60m_repa_l9_256_per_residue_bs24_2gpu",
        1_000_000,
        True,
    ),
    ("repa_mpnn_l4", "proteina_60m_repa_mpnn_l4_256_per_residue", 1_000_000, True),
    ("repa_mpnn_l9", "proteina_60m_repa_mpnn_l9_256_per_residue", 1_000_000, True),
]

GEARNET_CKPT = os.environ.get(
    "GEARNET_CKPT_PATH",
    "/rds/user/sr2173/hpc-work/proteina/data/metric_factory/model_weights/gearnet_ca.pth",
)
MPNN_CKPT = os.environ.get(
    "MPNN_CKPT_PATH",
    "/rds/user/sr2173/hpc-work/proteina/ProteinMPNN/ca_model_weights/v_48_020.pt",
)
# ESM2-150M default (was 650M). ~4× faster, retains the cross-encoder ranking
# signal. Override with ESM_MODEL_ID=facebook/esm2_t33_650M_UR50D if needed.
ESM_MODEL_ID = os.environ.get("ESM_MODEL_ID", "facebook/esm2_t30_150M_UR50D")

OUT_DIR = ALIGN_ROOT / "results"
MANIFEST_PATH = OUT_DIR / "frozen_batch_n10k.json"
RESIDUE_INDEX_PATH = OUT_DIR / "frozen_batch_n10k_residues.pt"
MODEL_OUT = OUT_DIR / "model_features"
ENC_OUT = OUT_DIR / "encoder_features"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _mean_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool a [B, N, D] feature tensor over real residues to [B, D].

    Mask is [B, N] bool, True = real residue. Uses sum/clamp_min(1) to be
    safe against any all-padding rows.
    """
    h = h * mask[..., None].to(h.dtype)
    s = h.sum(dim=1)
    n = mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
    return s / n  # [B, D]


def _shard_residue_index(
    residue_index: torch.Tensor, shard_start: int, shard_end: int
) -> torch.Tensor:
    """Filter the global [n_resid, 2] (protein_i, residue_j) index to a shard.

    Returns a [n_in_shard, 3] tensor with columns
    (global_resid_row, local_protein_i, residue_j) — where
    ``global_resid_row`` is the original row in residue_index (so the caller
    can write into the correct slot of the global output) and
    ``local_protein_i`` is ``global_protein_i - shard_start`` for indexing
    into shard tensors.
    """
    pi = residue_index[:, 0]
    in_shard = (pi >= shard_start) & (pi < shard_end)
    rows = in_shard.nonzero(as_tuple=True)[0]
    local_pi = pi[rows] - shard_start
    rj = residue_index[rows, 1]
    return torch.stack([rows, local_pi, rj], dim=1).long()


def _shard_manifest(manifest: Dict, start: int, end: int) -> Dict:
    """Return a copy of ``manifest`` restricted to keys[start:end]."""
    return {
        **manifest,
        "keys": manifest["keys"][start:end],
        "lengths": manifest["lengths"][start:end],
        "n_proteins": end - start,
    }


def _alloc_per_protein(n_proteins: int, dim: int) -> torch.Tensor:
    return torch.zeros((n_proteins, dim), dtype=torch.float32)


def _alloc_per_residue(n_resid: int, dim: int) -> torch.Tensor:
    return torch.zeros((n_resid, dim), dtype=torch.float32)


# ── Proteina-model extraction (sharded, per-layer reduce-in-loop) ───────────


def extract_one_model(
    row_name: str,
    run_dir: str,
    step: int,
    is_repa: bool,
    manifest: Dict,
    residue_index: torch.Tensor,
    n_proteins: int,
    n_residues: int,
    device: str,
) -> None:
    pp_path = MODEL_OUT / f"{row_name}_per_protein.pt"
    pr_path = MODEL_OUT / f"{row_name}_per_residue.pt"
    if pp_path.exists() and pr_path.exists():
        print(f"[{row_name}] both outputs cached, skip")
        return

    ckpt_path = find_checkpoint_path(run_dir, step, prefer_ema=True)
    if ckpt_path is None:
        raise FileNotFoundError(f"No EMA checkpoint for {run_dir} @ step={step}")
    print(f"[{row_name}] loading {ckpt_path.name}")
    model = load_checkpoint_by_path(str(ckpt_path), is_repa=is_repa, device=device)
    n_layers = model_num_layers(model)
    layers = list(range(n_layers))
    enable_hidden_states(model, layers)

    # Allocate global per-protein + per-residue buffers (one per layer).
    per_protein: Dict[int, torch.Tensor] = {}
    per_residue: Dict[int, torch.Tensor] = {}
    dim: int = -1

    for shard_start in range(0, n_proteins, SHARD_SIZE):
        shard_end = min(shard_start + SHARD_SIZE, n_proteins)
        shard_manifest = _shard_manifest(manifest, shard_start, shard_end)
        shard_batch, _ = load_proteina_batch_from_manifest(
            shard_manifest, device=device
        )
        # Per-layer reduce: extract_model_hidden_states_multilayer returns
        # {layer: [B_shard, N, D]} on CPU. We immediately reduce per shard.
        per_layer = extract_model_hidden_states_multilayer(
            model, shard_batch, layers, chunk_size=PROTEINA_CHUNK, t_value=T_VALUE
        )
        mask_cpu = shard_batch["mask"].detach().cpu().bool()  # [B_shard, N]

        # Compute per-residue selector for this shard.
        shard_idx = _shard_residue_index(residue_index, shard_start, shard_end)
        global_rows = shard_idx[:, 0]
        local_pi = shard_idx[:, 1]
        local_rj = shard_idx[:, 2]

        for lyr, h in per_layer.items():
            # h: [B_shard, N, D] CPU
            if dim < 0:
                dim = h.shape[-1]
                for L in layers:
                    per_protein[L] = _alloc_per_protein(n_proteins, dim)
                    per_residue[L] = _alloc_per_residue(n_residues, dim)
            pp = _mean_pool(h, mask_cpu)  # [B_shard, D]
            per_protein[lyr][shard_start:shard_end] = pp
            if global_rows.numel():
                per_residue[lyr][global_rows] = h[local_pi, local_rj]

        del per_layer, shard_batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[{row_name}] shard {shard_start:>5d}..{shard_end:>5d} "
            f"({global_rows.numel()} residues kept)"
        )

    torch.save(
        {
            "row_name": row_name,
            "run_dir": run_dir,
            "step": step,
            "is_repa": is_repa,
            "ckpt_path": str(ckpt_path),
            "t_value": T_VALUE,
            "per_layer": per_protein,  # {layer: [n_proteins, D]}
            "mode": "per_protein",
        },
        pp_path,
    )
    torch.save(
        {
            "row_name": row_name,
            "run_dir": run_dir,
            "step": step,
            "is_repa": is_repa,
            "ckpt_path": str(ckpt_path),
            "t_value": T_VALUE,
            "per_layer": per_residue,  # {layer: [n_residues, D]}
            "mode": "per_residue",
        },
        pr_path,
    )
    print(f"[{row_name}] wrote {pp_path.name} + {pr_path.name}")

    del model, per_protein, per_residue
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Encoder extraction (sharded, mean-pool + selected residues) ─────────────


def _extract_encoder(
    name: str,
    forward_fn,  # (shard_batch, device) -> [B_shard, N, D]
    manifest: Dict,
    residue_index: torch.Tensor,
    n_proteins: int,
    n_residues: int,
    device: str,
) -> None:
    pp_path = ENC_OUT / f"{name}_per_protein.pt"
    pr_path = ENC_OUT / f"{name}_per_residue.pt"
    if pp_path.exists() and pr_path.exists():
        print(f"[{name}] both outputs cached, skip")
        return

    per_protein: torch.Tensor = None
    per_residue: torch.Tensor = None
    dim = -1

    for shard_start in range(0, n_proteins, SHARD_SIZE):
        shard_end = min(shard_start + SHARD_SIZE, n_proteins)
        shard_manifest = _shard_manifest(manifest, shard_start, shard_end)
        shard_batch, _ = load_proteina_batch_from_manifest(
            shard_manifest, device=device
        )
        h = forward_fn(shard_batch, device).detach().cpu()  # [B_shard, N, D]
        mask_cpu = shard_batch["mask"].detach().cpu().bool()  # [B_shard, N]
        if dim < 0:
            dim = h.shape[-1]
            per_protein = _alloc_per_protein(n_proteins, dim)
            per_residue = _alloc_per_residue(n_residues, dim)

        pp = _mean_pool(h, mask_cpu)
        per_protein[shard_start:shard_end] = pp

        shard_idx = _shard_residue_index(residue_index, shard_start, shard_end)
        if shard_idx.numel():
            global_rows = shard_idx[:, 0]
            local_pi = shard_idx[:, 1]
            local_rj = shard_idx[:, 2]
            per_residue[global_rows] = h[local_pi, local_rj]

        del shard_batch, h
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{name}] shard {shard_start:>5d}..{shard_end:>5d}")

    torch.save({"encoder": name, "features": per_protein, "dim": dim}, pp_path)
    torch.save({"encoder": name, "features": per_residue, "dim": dim}, pr_path)
    print(f"[{name}] wrote {pp_path.name} + {pr_path.name}")


def extract_gearnet(manifest, residue_index, n_proteins, n_residues, device):
    if (ENC_OUT / "gearnet_per_protein.pt").exists() and (
        ENC_OUT / "gearnet_per_residue.pt"
    ).exists():
        print("[gearnet] cached, skip")
        return
    from proteinfoundation.repa.gearnet_encoder import GearNetPerResidueEncoder

    print(f"[gearnet] loading {GEARNET_CKPT}")
    enc = GearNetPerResidueEncoder(ckpt_path=GEARNET_CKPT)
    enc.eval().to(device)

    def fwd(batch, device):
        return extract_gearnet_embeddings(enc, batch, chunk_size=ENCODER_CHUNK)

    _extract_encoder(
        "gearnet", fwd, manifest, residue_index, n_proteins, n_residues, device
    )
    # enc dropped on function return; ruff F821-flags a del-before-closure here.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_mpnn(manifest, residue_index, n_proteins, n_residues, device):
    if (ENC_OUT / "mpnn_per_protein.pt").exists() and (
        ENC_OUT / "mpnn_per_residue.pt"
    ).exists():
        print("[mpnn] cached, skip")
        return
    from proteinfoundation.repa.mpnn_encoder import ProteinMPNNPerResidueEncoder
    from proteinfoundation.utils.coors_utils import ang_to_nm

    print(f"[mpnn] loading {MPNN_CKPT}")
    enc = ProteinMPNNPerResidueEncoder(ckpt_path=MPNN_CKPT)
    enc.eval().to(device)

    def fwd(batch, device):
        outs: List[torch.Tensor] = []
        B = batch["coords"].shape[0]
        for s in range(0, B, ENCODER_CHUNK):
            e = min(s + ENCODER_CHUNK, B)
            ca_nm = ang_to_nm(batch["coords"][s:e, :, 1, :].to(device))
            mask = batch["mask"][s:e].to(device).bool()
            outs.append(enc(ca_nm, mask).detach())
        return torch.cat(outs, dim=0)

    _extract_encoder(
        "mpnn", fwd, manifest, residue_index, n_proteins, n_residues, device
    )
    # enc dropped on function return; ruff F821-flags a del-before-closure here.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_esm(manifest, residue_index, n_proteins, n_residues, device):
    if (ENC_OUT / "esm2_per_protein.pt").exists() and (
        ENC_OUT / "esm2_per_residue.pt"
    ).exists():
        print("[esm2] cached, skip")
        return
    from proteinfoundation.repa.esm_encoder import ESMPerResidueEncoder

    print(f"[esm2] loading {ESM_MODEL_ID}")
    enc = ESMPerResidueEncoder(model_id=ESM_MODEL_ID)
    enc.eval().to(device)

    def fwd(batch, device):
        outs: List[torch.Tensor] = []
        B = batch["coords"].shape[0]
        for s in range(0, B, ESM_CHUNK):
            e = min(s + ESM_CHUNK, B)
            ca_nm = batch["coords"][s:e, :, 1, :].to(device)
            mask = batch["mask"][s:e].to(device).bool()
            res_type = batch["residue_type"][s:e].to(device).long()
            outs.append(enc(ca_nm, mask, residue_type=res_type).detach())
        return torch.cat(outs, dim=0)

    _extract_encoder(
        "esm2", fwd, manifest, residue_index, n_proteins, n_residues, device
    )
    # enc dropped on function return; ruff F821-flags a del-before-closure here.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    ENC_OUT.mkdir(parents=True, exist_ok=True)

    if not (MANIFEST_PATH.exists() and RESIDUE_INDEX_PATH.exists()):
        raise FileNotFoundError(
            f"Run build_batch.py first; missing {MANIFEST_PATH} or {RESIDUE_INDEX_PATH}"
        )
    import json as _json

    with open(MANIFEST_PATH) as f:
        manifest = _json.load(f)
    res_payload = torch.load(RESIDUE_INDEX_PATH, map_location="cpu", weights_only=False)
    residue_index: torch.Tensor = res_payload["residue_index"]
    n_proteins = res_payload["n_proteins"]
    n_residues = res_payload["n_residues_subsample"]
    print(
        f"Loaded manifest: n_proteins={n_proteins}, "
        f"per-residue subsample={n_residues}"
    )

    device = _default_device()

    # Encoders first (smaller GPU footprint; fail fast if a ckpt is missing).
    extract_gearnet(manifest, residue_index, n_proteins, n_residues, device)
    extract_mpnn(manifest, residue_index, n_proteins, n_residues, device)
    extract_esm(manifest, residue_index, n_proteins, n_residues, device)

    # Proteina checkpoints (largest GPU footprint; slowest to load).
    for row_name, run_dir, step, is_repa in MODEL_ROWS:
        extract_one_model(
            row_name,
            run_dir,
            step,
            is_repa,
            manifest,
            residue_index,
            n_proteins,
            n_residues,
            device,
        )

    print("\nDone. Feature files:")
    for p in sorted(MODEL_OUT.iterdir()):
        print(f"  {p}")
    for p in sorted(ENC_OUT.iterdir()):
        print(f"  {p}")


if __name__ == "__main__":
    main()
