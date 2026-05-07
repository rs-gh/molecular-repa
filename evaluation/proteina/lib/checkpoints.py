"""Shared checkpoint registry for Proteina evaluation suites.

This is the single source of truth for which checkpoints are evaluated across
both the generation (FID/designability) and representation (contact/CATH probe)
suites.  Both ``generation/scripts/run_sweep.py`` and
``representation/scripts/run_sweep.py`` import from here.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECKPOINT TAXONOMY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Three evaluation regimes, each with a corresponding sweep profile in
``generation/sweep_config.yaml`` and ``representation/sweep_config.yaml``:

1.  n=512 convergence  (profile: ``n512_convergence``)
    Run names: baseline, repa_l0, repa_l4, repa_l9
    Steps: 11-12 log-spaced points from 10K to 740K/840K (BASELINE_STEPS /
    REPA_STEPS / REPA_L0_STEPS).  Purpose: training-progression / convergence
    curves.  Steps are log-spaced to give dense early coverage (quality
    changes fast) and sparser late coverage.  These are the same steps that
    the now-deleted eval_fid_lite_sweep.sh scripts used.

2.  Sample-matched single points  (profiles: ``n128``, ``n256``, ``n512_sm``)
    Run names: baseline_128/256/512_sm, repa_l{0,4,9}_128/256/512_sm
    One checkpoint per run, chosen so every run has seen the same number of
    training samples.  Purpose: fair cross-size and cross-method comparison at
    equal training budget.

3.  ESM-REPA  (no dedicated sweep profile; included in representation n128)
    Run names: esm_repa_l{0,4,9}_128
    Uses step=None (last-EMA fallback) because these runs stopped at non-round
    steps (l0=87.5K, l4=248.5K, l9=266K).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW SAMPLE-MATCHED STEPS WERE DERIVED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Batch sizes are not constant throughout training — they were increased at
step 220K for n=128 and n=256 runs as memory optimisations (length-bucketed
sampling) were applied.  This means samples_seen is piecewise-linear in step,
not simply step x batch_size.

n=512  (fixed batch sizes throughout; no batch-size switch):
    baseline  bs=6  ->  500K x 6 = 3.0M samples
    repa      bs=4  ->  750K x 4 = 3.0M samples   <- exact match at 3.0M
    -> use step 500K (baseline) and 750K (repa) for n=512 sample-matched

n=256  (bs switched 12 -> 24 at step 220K for all runs):
    All runs:  220K x 12 + 180K x 24 = 2.64M + 4.32M = 6.96M samples @ step 400K
    -> use step 400K for all n=256 runs (~7M samples)

n=128  (baseline fixed bs=24 throughout; REPA bs switched 24 -> 80 at step 220K):
    baseline:  800K x 24 = 19.2M samples
    repa:      220K x 24 + 180K x 80 = 5.28M + 14.4M = 19.68M samples @ step 400K
    -> baseline step 800K ~= repa step 400K at ~19.5M samples

Batch size history is documented in docs/research/proteina_training_runs.md
(batch size sweep / bucketed training section) and the training configs at
src/proteina/configs/experiment_config/training/.
"""

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


# ── n=512 convergence step schedules ─────────────────────────────────────────
# Log-spaced from 10K to the final checkpoint, matching the old
# eval_fid_lite_sweep.sh step arrays.  All use EMA checkpoints.

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
REPA_L9_STEPS = REPA_STEPS  # same log-spaced schedule as layer-4 default


# ── RUN_SCHEDULES ─────────────────────────────────────────────────────────────
# Master registry used by both generation and representation sweep scripts.
# Shape: name -> (run_dir, is_repa, repa_layer, step_list)
#   run_dir    : subdirectory under STORE_ROOT
#   is_repa    : load as ProteinaREPA (True) or Proteina (False)
#   repa_layer : transformer layer carrying the REPA loss (representation
#                probing uses this as the default layer to probe)
#   step_list  : list of global steps to evaluate; None -> last-EMA fallback

RUN_SCHEDULES = {
    # ── n=512 convergence (11-12 log-spaced steps each) ──────────────────── #
    "baseline": ("proteina_60m_baseline_v2", False, 4, BASELINE_STEPS),
    "repa_l4": ("proteina_60m_repa_v2", True, 4, REPA_STEPS),
    "repa_l0": ("proteina_60m_repa_layer0_v2", True, 0, REPA_L0_STEPS),
    "repa_l9": ("proteina_60m_repa_layer9_v2", True, 9, REPA_L9_STEPS),
    # ── n=128 sample-matched @ ~19.5M samples ────────────────────────────── #
    # baseline bs=24 fixed -> step 800K = 19.2M samples
    # repa bs=24->80 at step 220K -> step 400K = 19.68M samples
    "baseline_128": ("proteina_60m_baseline_128", False, 4, [800000]),
    "repa_l0_128": ("proteina_60m_repa_l0_128_per_residue", True, 0, [400000]),
    "repa_l4_128": ("proteina_60m_repa_l4_128_per_residue", True, 4, [400000]),
    "repa_l9_128": ("proteina_60m_repa_l9_128_per_residue", True, 9, [400000]),
    # ── n=128 ESM-REPA (last-EMA; non-round final steps) ─────────────────── #
    # l0=87.5K, l4=248.5K, l9=266K -- use None to fall back to last-EMA
    "esm_repa_l0_128": ("proteina_60m_repa_esm_l0_128_per_residue", True, 0, [None]),
    "esm_repa_l4_128": ("proteina_60m_repa_esm_l4_128_per_residue", True, 4, [None]),
    "esm_repa_l9_128": ("proteina_60m_repa_esm_l9_128_per_residue", True, 9, [None]),
    # ── n=128 bs=80 sweep (mid + end at sample-matched ~8M / ~16M) ───────── #
    # All bs=80 throughout; step 100K = 8M samples, step 200K = 16M samples.
    # bs80_lr3x has no step=200K yet (training stopped at ~165K) -> last-EMA
    # falls back to ~13M samples. All REPA students use repa_layer=4.
    "baseline_128_bs80": ("proteina_60m_baseline_128_bs80", False, 4, [100000, 200000]),
    "repa_l4_128_bs80": (
        "proteina_60m_repa_l4_128_per_residue_bs80",
        True,
        4,
        [100000, 200000],
    ),
    "repa_l4_128_bs80_lr3x": (
        "proteina_60m_repa_l4_128_per_residue_bs80_lr3x",
        True,
        4,
        [100000, None],
    ),
    "repa_l4_128_random": (
        "proteina_60m_repa_l4_128_per_residue_random",
        True,
        4,
        [100000, 200000],
    ),
    # ── n=256 sample-matched @ ~7M samples ───────────────────────────────── #
    # all runs bs=12->24 at step 220K -> step 400K = 6.96M samples
    "baseline_256": ("proteina_60m_baseline_256", False, 4, [400000]),
    "repa_l0_256": ("proteina_60m_repa_l0_256_per_residue", True, 0, [400000]),
    "repa_l4_256": ("proteina_60m_repa_l4_256_per_residue", True, 4, [400000]),
    "repa_l9_256": ("proteina_60m_repa_l9_256_per_residue", True, 9, [400000]),
    # ── n=512 sample-matched @ 3.0M samples ──────────────────────────────── #
    # baseline bs=6 fixed -> step 500K = 3.0M; repa bs=4 fixed -> step 750K = 3.0M
    # _sm suffix distinguishes these single-step entries from the full convergence
    # schedule entries above (both point at the same run dirs).
    "baseline_512_sm": ("proteina_60m_baseline_v2", False, 4, [500000]),
    "repa_l0_512_sm": ("proteina_60m_repa_layer0_v2", True, 0, [750000]),
    "repa_l4_512_sm": ("proteina_60m_repa_v2", True, 4, [750000]),
    "repa_l9_512_sm": ("proteina_60m_repa_layer9_v2", True, 9, [750000]),
    # ── n=256 paper-protocol epoch-matched checkpoints ───────────────────── #
    # Three comparison groups, each at an epoch-matched checkpoint (epoch =
    # one full pass over training data; step counts vary across runs because
    # of bs=12→24 transitions and AFDB vs PDB dataset sizes).
    #   PDB main (drop _random): target ~ep24, span ep21-26 (all step 400K)
    #   PDB random sweep: target ep17 (capped by _random)
    #   AFDB sweep: target ep20 (exact match)
    # Paper-protocol runs use the suffix-stripped key for schedule lookup;
    # GEN_RUN_CONFIGS holds the suffixed `_fid` / `_des` variants pointing at
    # one of the two shared protocol configs.
    "baseline_256_ep21": ("proteina_60m_baseline_256", False, 4, [400000]),
    "repa_l0_256_ep26": ("proteina_60m_repa_l0_256_per_residue", True, 0, [400000]),
    "repa_l4_256_ep22": ("proteina_60m_repa_l4_256_per_residue", True, 4, [400000]),
    "repa_l9_256_ep25": ("proteina_60m_repa_l9_256_per_residue", True, 9, [400000]),
    "repa_l9_256_ep17": ("proteina_60m_repa_l9_256_per_residue", True, 9, [300000]),
    "repa_l4_256_random_ep17": (
        "proteina_60m_repa_l4_256_per_residue_random",
        True,
        4,
        [200000],
    ),
    "baseline_afdb_256_ep20": (
        "proteina_60m_baseline_afdb_swissprot_256",
        False,
        4,
        [200000],
    ),
    "repa_l4_afdb_256_ep20": (
        "proteina_60m_repa_l4_256_afdb_per_residue",
        True,
        4,
        [200000],
    ),
    # ── n=128 paper-protocol entries ─────────────────────────────────────── #
    # One per (run, step) snapshot. step suffixes: _step100k / _step200k /
    # _step400k / _steplast (= last-EMA, used for runs still training so we
    # explicitly note we're picking up wherever they are).
    #
    # ABLATION 1 (layer @ bs=80): baseline + l0 + l4 + l9. l0/l9 only have
    # last-EMA at ~63-66K (= ~5M samples) — early; redo at 200K when training
    # reaches it. Plot caption should note the asymmetry.
    "baseline_128_bs80_step200k": (
        "proteina_60m_baseline_128_bs80",
        False,
        4,
        [200000],
    ),
    "repa_l0_128_bs80_steplast": (
        "proteina_60m_repa_l0_128_per_residue_bs80",
        True,
        0,
        [None],  # last-EMA, currently ~66.5K (5.32M samples)
    ),
    "repa_l4_128_bs80_step200k": (
        "proteina_60m_repa_l4_128_per_residue_bs80",
        True,
        4,
        [200000],
    ),
    "repa_l9_128_bs80_steplast": (
        "proteina_60m_repa_l9_128_per_residue_bs80",
        True,
        9,
        [None],  # last-EMA, currently ~63K (5.04M samples)
    ),
    # ABLATION 2 (encoder @ L4 bs=80): baseline + 6 REPA targets. pw_*  only
    # have 100K — earlier (8M vs 16M samples) — redo at 200K once available.
    # ESM_t30 (penultimate-layer target) has no checkpoints yet, dropped here.
    "repa_l4_128_random_step200k": (
        "proteina_60m_repa_l4_128_per_residue_random",
        True,
        4,
        [200000],
    ),
    "repa_l4_128_pw_structure_step100k": (
        "proteina_60m_repa_l4_128_per_residue_pw_structure",
        True,
        4,
        [100000],  # only 100K available — note 8M sample budget
    ),
    "repa_l4_128_pw_torsional_step100k": (
        "proteina_60m_repa_l4_128_per_residue_pw_torsional",
        True,
        4,
        [100000],  # only 100K available — note 8M sample budget
    ),
    "repa_mpnn_l4_128_bs80_step200k": (
        "proteina_60m_repa_mpnn_l4_128_per_residue_bs80",
        True,
        4,
        [200000],
    ),
    "repa_esm_l4_128_step200k": (
        "proteina_60m_repa_esm_l4_128_per_residue",
        True,
        4,
        [200000],
    ),
    # ABLATION 3 (bs+lr ablation): clean factorial. baseline_128_bs24 uses
    # the original `proteina_60m_baseline_128` directory (bs=24 was the
    # original n=128 batch size). repa_l4_128_bs24 uses the newer explicit
    # bs=24 re-run (`_bs24` suffix), not the older `_per_residue` directory.
    "baseline_128_bs24_step200k": (
        "proteina_60m_baseline_128",
        False,
        4,
        [200000],
    ),
    "baseline_128_bs24_step400k": (
        "proteina_60m_baseline_128",
        False,
        4,
        [400000],
    ),
    "baseline_128_bs80_lr3x_step200k": (
        "proteina_60m_baseline_128_bs80_lr3x",
        False,
        4,
        [200000],
    ),
    # repa_l4_128_bs80_lr3x stopped at step ~161K (epoch 123) without a 200K
    # checkpoint — use last-EMA (~12.9M samples vs 16M for other bs=80 runs).
    "repa_l4_128_bs24_step200k": (
        "proteina_60m_repa_l4_128_per_residue_bs24",
        True,
        4,
        [200000],
    ),
    "repa_l4_128_bs24_step400k": (
        "proteina_60m_repa_l4_128_per_residue_bs24",
        True,
        4,
        [400000],
    ),
    "repa_l4_128_bs80_lr3x_steplast": (
        "proteina_60m_repa_l4_128_per_residue_bs80_lr3x",
        True,
        4,
        [None],  # last-EMA at ~161K (12.88M samples) — see note above
    ),
    # TODO (lambda ablation): once `proteina_60m_repa_l4_128_per_residue_bs80_lambda1`
    # (λ=1.0) reaches 100K-step EMA, add:
    #   "repa_l4_128_bs80_lambda1_step100k": (..., True, 4, [100000]),
    # And similarly extend `lambda2` (λ=2.0) once it has a 100K ckpt (currently
    # at 17.5K). The bs80 entry above (λ=0.5) is the third leg of the ablation.
}


# ── GEN_RUN_CONFIGS ───────────────────────────────────────────────────────────
# Maps RUN_SCHEDULES key -> Hydra inference config name used by the generation
# sweep (generation/scripts/run_sweep.py).  All configs are "lite" variants
# (~300 PDBs, ~35 min/checkpoint) suitable for convergence-curve sweeps.
# Full-eval configs (6,125 PDBs) are used by eval_fid.sh directly.

GEN_RUN_CONFIGS = {
    "baseline": "inference/lite/inference_fid_60m_baseline_lite",
    "repa_l4": "inference/lite/inference_fid_60m_repa_lite",
    "repa_l0": "inference/lite/inference_fid_60m_repa_layer0_lite",
    "repa_l9": "inference/lite/inference_fid_60m_repa_layer9_lite",
    "baseline_128": "inference/lite/inference_fid_60m_baseline_128_lite",
    "repa_l0_128": "inference/lite/inference_fid_60m_repa_l0_128_lite",
    "repa_l4_128": "inference/lite/inference_fid_60m_repa_l4_128_lite",
    "repa_l9_128": "inference/lite/inference_fid_60m_repa_l9_128_lite",
    "baseline_256": "inference/lite/inference_fid_60m_baseline_256_lite",
    "repa_l0_256": "inference/lite/inference_fid_60m_repa_l0_256_lite",
    "repa_l4_256": "inference/lite/inference_fid_60m_repa_l4_256_lite",
    "repa_l9_256": "inference/lite/inference_fid_60m_repa_l9_256_lite",
    "baseline_512_sm": "inference/lite/inference_fid_60m_baseline_512_sm_lite",
    "repa_l0_512_sm": "inference/lite/inference_fid_60m_repa_l0_512_sm_lite",
    "repa_l4_512_sm": "inference/lite/inference_fid_60m_repa_l4_512_sm_lite",
    "repa_l9_512_sm": "inference/lite/inference_fid_60m_repa_l9_512_sm_lite",
    "baseline_128_bs80": "inference/lite/inference_fid_60m_baseline_128_bs80_lite",
    "repa_l4_128_bs80": "inference/lite/inference_fid_60m_repa_l4_128_bs80_lite",
    "repa_l4_128_bs80_lr3x": "inference/lite/inference_fid_60m_repa_l4_128_bs80_lr3x_lite",
    "repa_l4_128_random": "inference/lite/inference_fid_60m_repa_l4_128_random_lite",
    # ── n=256 paper protocol (unified FID + designability + diversity) ─── #
    # One generation pool per ckpt feeds all metric families. FID/fJSD/fS run
    # on the full 1125-PDB pool; designability/diversity run on the 5
    # paper-protocol lengths {50, 100, 150, 200, 250} via the orchestrator's
    # --designability_lengths flag (set in sweep_config.yaml).
    "baseline_256_ep21": "inference/paper/inference_fid_60m_paper",
    "repa_l0_256_ep26": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_ep22": "inference/paper/inference_fid_60m_paper",
    "repa_l9_256_ep25": "inference/paper/inference_fid_60m_paper",
    "repa_l9_256_ep17": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_random_ep17": "inference/paper/inference_fid_60m_paper",
    "baseline_afdb_256_ep20": "inference/paper/inference_fid_60m_paper",
    "repa_l4_afdb_256_ep20": "inference/paper/inference_fid_60m_paper",
    # ── n=128 paper protocol (truncated to in-distribution lengths) ──────── #
    # All point at inference/inference_fid_60m_n128_paper which generates 500
    # PDBs at lengths {50, 75, 100, 125} × 125 each (vs n=256's 1125 PDBs).
    # Designability subsamples 50 of 125 per length × 4 lengths = 200 PDBs.
    "baseline_128_bs80_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l0_128_bs80_steplast": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_bs80_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l9_128_bs80_steplast": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_random_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_pw_structure_step100k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_pw_torsional_step100k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l4_128_bs80_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_esm_l4_128_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_128_bs24_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_128_bs24_step400k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_128_bs80_lr3x_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_bs24_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_bs24_step400k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_bs80_lr3x_steplast": "inference/paper/inference_fid_60m_n128_paper",
}


# ── External / pretrained checkpoints ────────────────────────────────────────
# Static files at known absolute paths (not under STORE_ROOT).
# Used by representation probing to evaluate NVIDIA's released weights at all
# transformer layers (mirrors REPA paper Fig. 3a).
# Shape: name -> (absolute_path, is_repa, expected_nlayers)

PRETRAINED_CHECKPOINTS = {
    # 58.93M params, ProteinTransformerAF3 with nlayers=12
    # (distinct from our in-house 10-layer 60M runs).
    "pretrained_dfs_60m": (
        "/home/sr2173/git/molecular-repa/.local_ckpts/proteina_v1.3_DFS_60M_notri.ckpt",
        False,
        12,
    ),
}


# ── Flat last.ckpt registry (used by representation/scripts/run_all.py) ──────

CHECKPOINT_REGISTRY = {
    "baseline": (RUN_SCHEDULES["baseline"][0], False, None),
    "repa_l0": (RUN_SCHEDULES["repa_l0"][0], True, [0]),
    "repa_l4": (RUN_SCHEDULES["repa_l4"][0], True, [4]),
    "repa_l9": (RUN_SCHEDULES["repa_l9"][0], True, [9]),
}


# ── Path resolution helpers ───────────────────────────────────────────────────


def find_checkpoint_path(
    run_dir: str, step: Optional[int], prefer_ema: bool = True
) -> Optional[Path]:
    """Locate the checkpoint file for a given (run_dir, step).

    Matches the naming convention ``chk_epoch=*_step=<12-digit>.ckpt``
    (or the ``-EMA`` variant).  When ``step`` is ``None``, falls back to
    ``last-EMA.ckpt`` (or ``last.ckpt`` when ``prefer_ema`` is False).

    Args:
        run_dir: Run directory name under STORE_ROOT.
        step: Global step number, or None to use the last checkpoint.
        prefer_ema: If True, return the EMA variant (matches FID sweep convention).

    Returns:
        Path to the checkpoint or None if not found.
    """
    ckpt_dir = STORE_ROOT / run_dir / "checkpoints"
    if step is None:
        last = ckpt_dir / ("last-EMA.ckpt" if prefer_ema else "last.ckpt")
        return last if last.exists() else None
    padded = f"{step:012d}"
    suffix = "-EMA.ckpt" if prefer_ema else ".ckpt"
    for entry in os.listdir(ckpt_dir):
        if f"step={padded}" in entry and entry.endswith(suffix):
            if prefer_ema and "-EMA" not in entry:
                continue
            if not prefer_ema and "-EMA" in entry:
                continue
            return ckpt_dir / entry
    return None


def resolve_step(ckpt_path: Path, step: Optional[int]) -> int:
    """Return the integer step for a checkpoint.

    When ``step`` is not None, returns it directly.  When ``step`` is None
    (last-EMA fallback), reads ``global_step`` from the checkpoint file.
    """
    if step is not None:
        return step
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return int(raw["global_step"])


def load_checkpoint_by_path(ckpt_path: str, is_repa: bool, device: str = None):
    """Load a Proteina / ProteinaREPA LightningModule from an explicit ckpt path.

    Returns a fully-loaded eval-mode model on ``device`` (defaults to cuda if available).
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
