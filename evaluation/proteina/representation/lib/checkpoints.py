"""Run schedules, checkpoint path resolution, LightningModule loading."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch

PROTEINA_ROOT = Path("/home/sr2173/git/molecular-repa/src/proteina")
if str(PROTEINA_ROOT) not in sys.path:
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

# External / pretrained checkpoints — static files at known paths rather than
# sweep dirs under STORE_ROOT. Used to probe NVIDIA's released Proteina weights
# at all transformer layers, mirroring REPA Fig. 3a (layer-wise representation
# quality of the unconditional generative model).
#
# Shape: name -> (absolute_path, is_repa, expected_nlayers)
PRETRAINED_CHECKPOINTS = {
    # 58.93M params, ProteinTransformerAF3 with nlayers=12
    # (distinct from our 10-layer in-house 60M runs).
    "pretrained_dfs_60m": (
        "/home/sr2173/git/molecular-repa/.local_ckpts/proteina_v1.3_DFS_60M_notri.ckpt",
        False,
        12,
    ),
}


# Flat last.ckpt registry used by run_all.py (single-point mode).
CHECKPOINT_REGISTRY = {
    "baseline": (RUN_SCHEDULES["baseline"][0], False, None),
    "repa_l0": (RUN_SCHEDULES["repa_l0"][0], True, [0]),
    "repa_l4": (RUN_SCHEDULES["repa_l4"][0], True, [4]),
    "repa_l9": (RUN_SCHEDULES["repa_l9"][0], True, [9]),
}


def find_checkpoint_path(
    run_dir: str, step: int, prefer_ema: bool = True
) -> Optional[Path]:
    """Locate the checkpoint file for a given (run_dir, step).

    Matches the naming convention ``chk_epoch=*_step=<12-digit>.ckpt``
    (or the ``-EMA`` variant).

    Args:
        run_dir: Run directory name under STORE_ROOT (e.g. ``proteina_60m_baseline_v2``).
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
            # Reject the non-EMA when we want EMA (both share "step=...ckpt" prefix).
            if prefer_ema and "-EMA" not in entry:
                continue
            if not prefer_ema and "-EMA" in entry:
                continue
            return ckpt_dir / entry
    return None


def load_checkpoint_by_path(ckpt_path: str, is_repa: bool, device: str = None):
    """Load a Proteina / ProteinaREPA LightningModule from an explicit ckpt path.

    Returns a fully-loaded eval-mode model on ``device`` (defaults to cuda if available).
    For REPA checkpoints this includes the frozen encoder + REPA loss module.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
    """Load a last.ckpt for a registry entry (used by run_all.py).

    Returns:
        (model, meta) where meta = {"is_repa", "repa_layers", "run_dir"}.
    """
    if name not in CHECKPOINT_REGISTRY:
        raise KeyError(
            f"Unknown checkpoint: {name}. Registry: {list(CHECKPOINT_REGISTRY)}"
        )
    run_dir, is_repa, repa_layers = CHECKPOINT_REGISTRY[name]
    ckpt_path = STORE_ROOT / run_dir / "checkpoints" / "last.ckpt"
    model = load_checkpoint_by_path(ckpt_path, is_repa, device=device)
    return model, {"is_repa": is_repa, "repa_layers": repa_layers, "run_dir": run_dir}
