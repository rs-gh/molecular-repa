"""Local checkpoint registry for the generation-speedup workstream.

Deliberately duplicated (not imported) from evaluation/proteina/representation/lib/checkpoints.py
so this tool can evolve independently. Mirrors the 12 sample-matched entries across
n=128 / n=256 / n=512 model sizes. All entries are commented out except the active
benchmarking target.

To benchmark a new checkpoint: uncomment its line below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

STORE_ROOT = Path("/rds/user/sr2173/hpc-work/proteina/store")


# (run_dir, is_repa, repa_layer, step)
#
# Un-comment an entry to include it in `resolve_active()`.
CHECKPOINTS: dict[str, tuple[str, bool, int, int]] = {
    # --------------------------------------------------------------------- #
    # n=128 — sample-matched @ ~19.5M samples
    # (baseline bs=24 throughout; REPA bs=24→80 switch at step 220K,
    #  so step 400K = 220K × 24 + 180K × 80 = 19.68M)
    # --------------------------------------------------------------------- #
    # "baseline_128": ("proteina_60m_baseline_128", False, 4, 800000),
    # "repa_l0_128":  ("proteina_60m_repa_l0_128_per_residue", True, 0, 400000),
    # "repa_l4_128":  ("proteina_60m_repa_l4_128_per_residue", True, 4, 400000),
    # "repa_l9_128":  ("proteina_60m_repa_l9_128_per_residue", True, 9, 400000),
    # --------------------------------------------------------------------- #
    # n=256 — sample-matched @ ~7M samples
    # (all runs bs=12→24 switch at step 220K,
    #  so step 400K = 220K × 12 + 180K × 24 = 6.96M)
    # --------------------------------------------------------------------- #
    # "baseline_256": ("proteina_60m_baseline_256", False, 4, 400000),
    # "repa_l0_256":  ("proteina_60m_repa_l0_256_per_residue", True, 0, 400000),
    # "repa_l4_256":  ("proteina_60m_repa_l4_256_per_residue", True, 4, 400000),
    # "repa_l9_256":  ("proteina_60m_repa_l9_256_per_residue", True, 9, 400000),
    # --------------------------------------------------------------------- #
    # n=512 — sample-matched @ exactly 3.0M samples
    # (baseline bs=6 × 500K = REPA bs=4 × 750K)
    # --------------------------------------------------------------------- #
    "baseline_512_sm": ("proteina_60m_baseline_v2", False, 4, 500000),
    # "repa_l0_512_sm": ("proteina_60m_repa_layer0_v2", True,  0, 750000),
    # "repa_l4_512_sm": ("proteina_60m_repa_v2",        True,  4, 750000),
    # "repa_l9_512_sm": ("proteina_60m_repa_layer9_v2", True,  9, 750000),
}


@dataclass(frozen=True)
class ResolvedCkpt:
    name: str
    ckpt_path: Path
    is_repa: bool
    repa_layer: int
    step: int


def find_ema_ckpt(run_dir: str, step: int) -> Path:
    """Locate the EMA checkpoint matching `step` inside STORE_ROOT/run_dir/checkpoints."""
    ckpt_dir = STORE_ROOT / run_dir / "checkpoints"
    padded = f"{step:012d}"
    suffix = f"step={padded}-EMA.ckpt"
    for entry in os.listdir(ckpt_dir):
        if entry.endswith(suffix):
            return ckpt_dir / entry
    raise FileNotFoundError(f"No EMA ckpt for step={step} under {ckpt_dir}")


def resolve_active() -> list[ResolvedCkpt]:
    """Expand the uncommented CHECKPOINTS entries into ResolvedCkpt records."""
    resolved = []
    for name, (run_dir, is_repa, layer, step) in CHECKPOINTS.items():
        resolved.append(
            ResolvedCkpt(
                name=name,
                ckpt_path=find_ema_ckpt(run_dir, step),
                is_repa=is_repa,
                repa_layer=layer,
                step=step,
            )
        )
    return resolved


if __name__ == "__main__":
    for rc in resolve_active():
        print(rc)
