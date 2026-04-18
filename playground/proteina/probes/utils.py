# ruff: noqa: F841
"""Shared utilities for proteina representation-quality probes.

Provides:
  - `load_proteina_batch(n, max_size)`  : read PyG Data objects from LMDB, pad + batch
  - `load_checkpoint(name)`             : load baseline/REPA Proteina LightningModule
  - `enable_hidden_states(model, layers)` : cast model.nn to ProteinTransformerAF3WithHiddenStates
  - `extract_model_hidden_states(...)`  : forward pass with x_t=x_1_clean, t=1 → per-residue reps
  - `extract_gearnet_embeddings(...)`   : frozen GearNet encoder over CA coords
  - `contact_labels(ca_coords_ang, mask)` : [B, N, N] long-range contact labels (8 Å, |i-j|≥24)
  - `cath_labels_from_batch(cath_codes, level)` : coarse CATH class label per protein
  - `topk_precision_per_protein(...)`   : P@L/k for ranked contact predictions

Design notes:
  - Probes operate on CA-only coords in nm (ang_to_nm applied).
  - `x_t = x_1` + `t = 1.0` means we probe the clean-data endpoint; this is
    the setting most analogous to ImageNet linear probes on clean images.
  - For baseline checkpoints (Proteina, not ProteinaREPA), we swap
    `model.nn.__class__` in-place so the same forward path captures hidden
    states. Weights are untouched.
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: F401

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

PROTEINA_ROOT = Path("/home/sr2173/git/molecular-repa/src/proteina")
sys.path.insert(0, str(PROTEINA_ROOT))

LMDB_PATH = os.environ.get(
    "PROBES_LMDB_PATH",
    "/rds/user/sr2173/hpc-work/proteina/data/pdb_train/lmdb/val.lmdb",
)
STORE_ROOT = Path("/rds/user/sr2173/hpc-work/proteina/store")

# Log-spaced step schedule mirroring hpc-scripts/proteina/evaluation/eval_fid_lite_sweep.sh.
# Uses EMA checkpoints, matching the FID-sweep convention.
BASELINE_STEPS = [
    10000,
    20000,
    40000,
    80000,
    150000,
    250000,
    350000,
    450000,
    550000,
    650000,
    740000,
]
REPA_STEPS = [
    10000,
    20000,
    40000,
    80000,
    150000,
    250000,
    350000,
    450000,
    550000,
    650000,
    750000,
    840000,
]
REPA_L0_STEPS = [
    10000,
    20000,
    40000,
    80000,
    150000,
    250000,
    350000,
    450000,
    550000,
    650000,
    750000,
    830000,
]
REPA_L9_STEPS = REPA_STEPS  # matches the FID sweep — same schedule as layer-4 default

RUN_SCHEDULES = {
    # name: (run_dir, is_repa, repa_layer, step_list)
    "baseline": (
        "proteina_60m_baseline_v2",
        False,
        4,
        BASELINE_STEPS,
    ),  # probed at layer 4 (midpoint)
    "repa_l4": (
        "proteina_60m_repa_v2",
        True,
        4,
        REPA_STEPS,
    ),  # default REPA trains layer 4
    "repa_l0": ("proteina_60m_repa_layer0_v2", True, 0, REPA_L0_STEPS),
    "repa_l9": ("proteina_60m_repa_layer9_v2", True, 9, REPA_L9_STEPS),
}

# Keep the old registry for run_all.py (last.ckpt single-point mode); this only maps to _v2 runs now.
CHECKPOINT_REGISTRY = {
    "baseline": (RUN_SCHEDULES["baseline"][0], False, None),
    "repa_l0": (RUN_SCHEDULES["repa_l0"][0], True, [0]),
    "repa_l4": (RUN_SCHEDULES["repa_l4"][0], True, [4]),
    "repa_l9": (RUN_SCHEDULES["repa_l9"][0], True, [9]),
}


def find_checkpoint_path(
    run_dir: str, step: int, prefer_ema: bool = True
) -> Optional[Path]:
    """Locate the checkpoint file for a given (run_dir, step) matching the naming
    convention `chk_epoch=*_step=<12-digit>.ckpt` (or `-EMA` variant).

    Args:
        run_dir: Run directory name under STORE_ROOT (e.g. `proteina_60m_baseline_v2`).
        step: Global step number.
        prefer_ema: If True, return the EMA variant (matches FID sweep convention).

    Returns:
        Path to the checkpoint or None if not found.
    """
    ckpt_dir = STORE_ROOT / run_dir / "checkpoints"
    padded = f"{step:012d}"
    suffix = "-EMA.ckpt" if prefer_ema else ".ckpt"
    # Dir listing is unavoidable; do it once per lookup.
    for entry in os.listdir(ckpt_dir):
        if f"step={padded}" in entry and entry.endswith(suffix):
            # Reject the non-EMA when we want EMA (both can share "step=...ckpt" prefix).
            if prefer_ema and "-EMA" not in entry:
                continue
            if not prefer_ema and "-EMA" in entry:
                continue
            return ckpt_dir / entry
    return None


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_proteina_batch(
    n: int,
    max_size: int = 256,
    lmdb_path: str = LMDB_PATH,
    device: str = None,
) -> Tuple[Dict, List[Data]]:
    """Read `n` PyG Data objects from the LMDB, filter to length ≤ max_size,
    pad all tensors to max_size, stack into a batched dict compatible with
    model.nn.forward.

    Returns:
        batch: dict with tensor keys (coords, mask, residue_type, chain_break, ...) on device
        raw: list of the original Data objects (useful for CATH labels etc.)
    """
    if device is None:
        device = _default_device()

    db = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        subdir=False,
        readahead=False,
        meminit=False,
    )
    raw = []
    with db.begin() as txn:
        cursor = txn.cursor()
        for _, v in cursor:
            g = pickle.loads(v)
            n_res = g.coords.shape[0] if hasattr(g, "coords") else g.num_nodes
            if n_res <= max_size:
                raw.append(g)
            if len(raw) >= n:
                break
    db.close()

    if not raw:
        raise RuntimeError(f"No proteins ≤ {max_size} residues found in {lmdb_path}")

    B = len(raw)
    keys_tensor = set()
    for g in raw:
        for k, v in g:
            if isinstance(v, torch.Tensor):
                keys_tensor.add(k)

    batch: Dict[str, torch.Tensor] = {}
    for k in keys_tensor:
        stacked = []
        ok = True
        for g in raw:
            v = g[k]
            if v.dim() == 0:
                # Scalar per-protein attribute — stack as-is.
                stacked.append(v)
                continue
            pad_size = max_size - v.shape[0]
            if pad_size > 0:
                # F.pad order is right-to-left over dims; set last entry to pad dim 0 after.
                pad_tuple = [0] * (2 * v.dim())
                pad_tuple[-1] = pad_size
                v = F.pad(v, tuple(pad_tuple), mode="constant", value=0)
            elif pad_size < 0:
                v = v[:max_size]
            stacked.append(v)
        try:
            batch[k] = torch.stack(stacked, dim=0).to(device)
        except Exception as e:
            # Shapes don't line up across proteins for this key — skip it.
            print(f"  [warn] skipping key '{k}' in batch (stack failed: {e})")
            ok = False
        if not ok:
            continue

    # Build residue mask: True where a real residue lives. coords is [b, n, 37, 2]-style in
    # Proteina's pipeline; coords[..., 0, 0] is the mask for the backbone N atom, good enough
    # as a global residue mask (matches extract_clean_sample logic).
    if "mask_dict" in keys_tensor:
        # if mask_dict came through, let model extract as usual
        pass
    # Build an explicit [B, N] BOOL mask based on original lengths. The model's
    # cond_factory does `~mask`, which only works on bool/int tensors.
    lengths = torch.tensor([g.coords.shape[0] for g in raw], device=device)
    arange = torch.arange(max_size, device=device).unsqueeze(0).expand(B, -1)
    mask = arange < lengths.unsqueeze(1)  # [B, N] bool
    batch["mask"] = mask
    batch["lengths"] = lengths

    return batch, raw


# --------------------------------------------------------------------------- #
# Model loading / hidden-state extraction
# --------------------------------------------------------------------------- #


def load_checkpoint_by_path(ckpt_path: str, is_repa: bool, device: str = None):
    """Load a Proteina or ProteinaREPA LightningModule from an explicit ckpt path."""
    if device is None:
        device = _default_device()

    print(f"Loading {'ProteinaREPA' if is_repa else 'Proteina'} from {ckpt_path}")

    if is_repa:
        from proteinfoundation.repa.proteina_repa import ProteinaREPA

        model = ProteinaREPA.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    else:
        from proteinfoundation.proteinflow.proteina import Proteina

        model = Proteina.load_from_checkpoint(str(ckpt_path), map_location="cpu")

    model.eval()
    model.to(device)
    return model


def load_checkpoint(name: str, device: str = None):
    """Load a last.ckpt for a registry entry (used by run_all.py)."""
    if name not in CHECKPOINT_REGISTRY:
        raise KeyError(
            f"Unknown checkpoint: {name}. Registry: {list(CHECKPOINT_REGISTRY)}"
        )
    run_dir, is_repa, repa_layers = CHECKPOINT_REGISTRY[name]
    ckpt_path = STORE_ROOT / run_dir / "checkpoints" / "last.ckpt"
    model = load_checkpoint_by_path(ckpt_path, is_repa, device=device)
    return model, {"is_repa": is_repa, "repa_layers": repa_layers, "run_dir": run_dir}


def enable_hidden_states(model, layers: List[int]):
    """Ensure model.nn returns hidden states at `layers`. Works for both
    Proteina (plain ProteinTransformerAF3) and ProteinaREPA (already a
    ProteinTransformerAF3WithHiddenStates).
    """
    from proteinfoundation.repa.protein_transformer_repa import (
        ProteinTransformerAF3WithHiddenStates,
    )

    if not isinstance(model.nn, ProteinTransformerAF3WithHiddenStates):
        # Subclass of ProteinTransformerAF3 — safe to swap __class__ in-place.
        model.nn.__class__ = ProteinTransformerAF3WithHiddenStates
    model.nn.repa_layers = list(layers)
    return model


@torch.no_grad()
def extract_model_hidden_states_multilayer(
    model,
    batch: Dict,
    layers: List[int],
    chunk_size: int = 8,
) -> Dict[int, torch.Tensor]:
    """Run a clean forward pass (x_t = x_1, t = 1.0) and return one [B, N, D] per layer.

    Extracting all layers in one pass is nearly free: the transformer runs the
    same forward regardless of how many layers we capture, so we amortize the
    expensive checkpoint-load over all probe layers.

    Returns:
        Dict keyed by layer index → [B, N, D] hidden state on CPU.
    """
    from proteinfoundation.utils.coors_utils import ang_to_nm

    enable_hidden_states(model, layers)
    device = next(model.parameters()).device
    model.eval()

    B = batch["coords"].shape[0]
    per_layer: Dict[int, List[torch.Tensor]] = {lyr: [] for lyr in layers}
    for s in range(0, B, chunk_size):
        e = min(s + chunk_size, B)
        sub = {k: v[s:e] for k, v in batch.items() if isinstance(v, torch.Tensor)}
        sub = {k: v.to(device) for k, v in sub.items()}

        ca = sub["coords"][:, :, 1, :]  # [b, n, 3] in ang
        x_1_nm = ang_to_nm(ca)
        sub["x_t"] = x_1_nm
        sub["t"] = torch.ones(e - s, device=device, dtype=x_1_nm.dtype)

        nn_out = model.nn(sub, return_hidden_states=True)
        hs = nn_out["hidden_states"]  # ordered list matching `layers`
        for lyr, h in zip(layers, hs):
            per_layer[lyr].append(h.detach().cpu())

    return {lyr: torch.cat(v, dim=0) for lyr, v in per_layer.items()}


# Backward-compat shim used by run_all.py.
@torch.no_grad()
def extract_model_hidden_states(
    model, batch: Dict, layer: int, chunk_size: int = 8
) -> torch.Tensor:
    out = extract_model_hidden_states_multilayer(
        model, batch, [layer], chunk_size=chunk_size
    )
    return out[layer]


def model_num_layers(model) -> int:
    """Return the transformer's trunk-layer count."""
    return int(model.nn.nlayers)


@torch.no_grad()
def extract_gearnet_embeddings(
    encoder, batch: Dict, chunk_size: int = 8
) -> torch.Tensor:
    """Run frozen GearNet encoder over CA coords. Returns [B, N, 512] on CPU."""
    from proteinfoundation.utils.coors_utils import ang_to_nm

    encoder.eval()
    device = next(encoder.parameters()).device

    B = batch["coords"].shape[0]
    N = batch["coords"].shape[1]
    outs = []
    for s in range(0, B, chunk_size):
        e = min(s + chunk_size, B)
        ca = batch["coords"][s:e, :, 1, :].to(device)
        mask_bool = batch["mask"][s:e].bool().to(device)
        ca_nm = ang_to_nm(ca)
        # GearNet encoder signature: (ca_coords_nm [b, n, 3], mask [b, n] bool)
        out = encoder(ca_nm, mask_bool)
        outs.append(out.detach().cpu())
    return torch.cat(outs, dim=0)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def contact_labels(
    ca_coords_ang: torch.Tensor,  # [B, N, 3]
    mask: torch.Tensor,  # [B, N]
    threshold_ang: float = 8.0,
    min_seq_sep: int = 24,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build [B, N, N] long-range contact labels + pair mask. Pair mask is False
    where either residue is padding or |i-j| < min_seq_sep.
    """
    B, N, _ = ca_coords_ang.shape
    dist = torch.cdist(ca_coords_ang, ca_coords_ang)  # [B, N, N]
    idx = torch.arange(N, device=ca_coords_ang.device)
    seqsep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # [N, N]
    long_range = seqsep >= min_seq_sep
    real = mask.bool().unsqueeze(2) & mask.bool().unsqueeze(1)  # [B, N, N]
    pair_mask = real & long_range.unsqueeze(0)
    labels = (dist < threshold_ang).float() * pair_mask.float()
    return labels, pair_mask


def cath_labels_from_raw(
    raw: List[Data],
    level: str = "T",
) -> Tuple[np.ndarray, List[str]]:
    """Extract one coarse CATH label per protein (first domain, masked to `level`).

    Level meaning: "C" = class only, "A" = class.arch, "T" = class.arch.topology.
    Returns (int labels, list of unique label strings).

    Handles several shapes defensively:
      - g.cath_code = "1.10.10.10"              (plain string)
      - g.cath_code = ["1.10.10.10", ...]        (list of domain strings — expected)
      - g.cath_code = [["1.10.10.10"], ...]      (nested list, seen on some builds)
      - g.cath_code = None / []                  (unlabelled)
    """
    mapping = {"C": 0, "A": 1, "T": 2, "H": 3}
    k = mapping[level]
    strs = []
    n_missing = 0
    for g in raw:
        cc = getattr(g, "cath_code", None)
        if cc is None or (hasattr(cc, "__len__") and len(cc) == 0):
            strs.append(None)
            n_missing += 1
            continue
        # Unwrap nested list / find first string.
        first = cc
        for _ in range(3):  # unwrap up to 3 levels
            if isinstance(first, (list, tuple)):
                if not first:
                    first = None
                    break
                first = first[0]
            else:
                break
        if not isinstance(first, str):
            strs.append(None)
            n_missing += 1
            continue
        parts = first.split(".")
        key = ".".join(parts[: k + 1]) if len(parts) > k else None
        strs.append(key)

    if n_missing == len(raw):
        # No CATH-labelled proteins at all — caller will handle downgrade/skip.
        pass
    uniq = sorted(set(s for s in strs if s is not None))
    idx_of = {s: i for i, s in enumerate(uniq)}
    labels = np.asarray([idx_of.get(s, -1) for s in strs], dtype=np.int64)
    return labels, uniq


def mean_pool_by_mask(reps: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Mean-pool [B, N, D] reps with a [B, N] real-residue mask (True = real)."""
    real = mask.float().unsqueeze(-1).cpu()
    summed = (reps * real).sum(dim=1)
    n = real.sum(dim=1).clamp(min=1.0)
    return (summed / n).float().numpy()


# --------------------------------------------------------------------------- #
# Contact probe (Top-L/k precision)
# --------------------------------------------------------------------------- #


@dataclass
class ContactResult:
    p_at_L: float  # mean precision @ L predictions
    p_at_L_5: float  # mean precision @ L/5 predictions (headline metric)
    p_at_L_2: float  # mean precision @ L/2 predictions
    n_proteins_test: int


def linear_probe_contacts(
    reps: torch.Tensor,  # [B, N, D] on CPU
    ca_coords_ang: torch.Tensor,  # [B, N, 3] on CPU (for labels)
    mask: torch.Tensor,  # [B, N] on CPU
    lengths: torch.Tensor,  # [B] on CPU (unmasked lengths)
    test_frac: float = 0.2,
    seed: int = 42,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 4,
) -> ContactResult:
    """Train a small MLP on pair features h_i||h_j||abs(h_i-h_j) → P(contact),
    sampling positive and negative pairs per protein. Evaluate P@L/k on held-out
    proteins by ranking predicted scores over long-range pairs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    B, N, D = reps.shape
    labels, pair_mask = contact_labels(ca_coords_ang, mask)

    perm = torch.randperm(B)
    n_test = max(1, int(B * test_frac))
    test_idx = perm[:n_test].tolist()
    train_idx = perm[n_test:].tolist()

    # Pair feature dim = 3D (concat h_i, h_j, |h_i - h_j|)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = torch.nn.Sequential(
        torch.nn.Linear(3 * D, 256),
        torch.nn.SiLU(),
        torch.nn.Linear(256, 1),
    ).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    def _balanced_pairs(
        h: torch.Tensor, lab: torch.Tensor, pmask: torch.Tensor, n_samples: int
    ):
        """Sample n_samples positive + n_samples negative pairs from one protein."""
        pos = (lab == 1) & pmask
        neg = (lab == 0) & pmask
        pos_ij = pos.nonzero(as_tuple=False)
        neg_ij = neg.nonzero(as_tuple=False)
        if pos_ij.numel() == 0 or neg_ij.numel() == 0:
            return None
        npos = min(len(pos_ij), n_samples)
        nneg = min(len(neg_ij), n_samples)
        pos_sel = pos_ij[torch.randperm(len(pos_ij))[:npos]]
        neg_sel = neg_ij[torch.randperm(len(neg_ij))[:nneg]]
        ij = torch.cat([pos_sel, neg_sel], dim=0)
        y = torch.cat([torch.ones(npos), torch.zeros(nneg)])
        hi = h[ij[:, 0]]
        hj = h[ij[:, 1]]
        feats = torch.cat([hi, hj, (hi - hj).abs()], dim=-1)
        return feats, y

    # --- Train ---
    mlp.train()
    for epoch in range(epochs):
        total = 0.0
        np.random.shuffle(train_idx)
        for b_start in range(0, len(train_idx), batch_size):
            sub = train_idx[b_start : b_start + batch_size]
            feats_all, y_all = [], []
            for i in sub:
                out = _balanced_pairs(reps[i], labels[i], pair_mask[i], n_samples=200)
                if out is None:
                    continue
                f, y = out
                feats_all.append(f)
                y_all.append(y)
            if not feats_all:
                continue
            f = torch.cat(feats_all, dim=0).to(device)
            y = torch.cat(y_all, dim=0).to(device)
            logits = mlp(f).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)

    # --- Eval: P@L/k per protein, averaged ---
    mlp.eval()
    p_L, p_L2, p_L5 = [], [], []
    with torch.no_grad():
        for i in test_idx:
            L = int(lengths[i].item())
            if L < 50:
                continue
            h = reps[i, :L]  # [L, D]
            pm = pair_mask[i, :L, :L]
            lab = labels[i, :L, :L]
            # Score upper triangle only
            iu = torch.triu_indices(L, L, offset=1)
            keep = pm[iu[0], iu[1]]
            ii, jj = iu[0][keep], iu[1][keep]
            hi = h[ii]
            hj = h[jj]
            feats = torch.cat([hi, hj, (hi - hj).abs()], dim=-1).to(device)
            scores = mlp(feats).squeeze(-1).cpu()
            y = lab[ii, jj]
            if scores.numel() == 0:
                continue
            order = torch.argsort(scores, descending=True)
            for topk, target in [(L, p_L), (L // 2, p_L2), (L // 5, p_L5)]:
                k = max(1, min(topk, scores.numel()))
                hits = y[order[:k]].sum().item()
                target.append(hits / k)

    return ContactResult(
        p_at_L=float(np.mean(p_L)) if p_L else 0.0,
        p_at_L_2=float(np.mean(p_L2)) if p_L2 else 0.0,
        p_at_L_5=float(np.mean(p_L5)) if p_L5 else 0.0,
        n_proteins_test=len(p_L5),
    )
