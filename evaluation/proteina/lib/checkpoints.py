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
    # Step-matched (NOT sample-matched) companion to repa_l9_256_ep17.
    # Both at step=300K but L4 bumped bs=12→24 at ~269K (only 31K steps
    # post-bump = 3.97M smp), while L9 bumped at ~196K (104K steps post-bump
    # = 4.85M smp). L9 has ~22% more samples at the same step. Used to test
    # whether L4 crosses the designability threshold earlier than L9 — but
    # interpret the comparison knowing L9 had the bigger sample budget.
    "repa_l4_256_ep13_step300k": (
        "proteina_60m_repa_l4_256_per_residue",
        True,
        4,
        [300000],
    ),
    # Final dated snapshot of the canonical n256 REPA L4 per_residue run
    # (ep31, step 500K, ~10.49M samples — bs=12→24 bumped at step 269K so
    # samples ≈ 12*269K + 24*231K). Companion to repa_l4_256_ep22 for a
    # late-training point on the same run; lambda025/lambda1/lambda2
    # comparison can also use this as the upper-bound λ=0.5 anchor once
    # they reach equivalent budgets.
    "repa_l4_256_ep31_step500k": (
        "proteina_60m_repa_l4_256_per_residue",
        True,
        4,
        [500000],
    ),
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
    # ── n=256 averaging-ablation per_sample variants ─────────────────────── #
    # Sample-matched companions to the per_residue rows (L0 ep26 / L4 ep22 /
    # L9 ep25). Pinning to last-EMA (None) per user choice 2026-05-07; the
    # actual step is captured in the sweep jsonl row from the resolved ckpt
    # filename, so the table can use the real number.
    "repa_l0_256_per_sample_steplast": (
        "proteina_60m_repa_l0_256_per_sample",
        True,
        0,
        [None],
    ),
    "repa_l4_256_per_sample_steplast": (
        "proteina_60m_repa_l4_256_per_sample",
        True,
        4,
        [None],
    ),
    # Explicit step pin for L4 per_sample because last-EMA.ckpt is a stale
    # pre-restart file from 2026-04-17 pointing at step ~56K (a fresh-init
    # run). The real latest is at last-v1-EMA.ckpt (= ep25/400K snapshot,
    # dated 04-20). The _steplast row above resolves to garbage (proven:
    # eval 28993862 task 1 returned step=56000, des=0.0); use this _step400k
    # row instead. Sample-matched to L4 per_residue ep22 (~6.69M smp).
    "repa_l4_256_per_sample_step400k": (
        "proteina_60m_repa_l4_256_per_sample",
        True,
        4,
        [400000],
    ),
    "repa_l9_256_per_sample_steplast": (
        "proteina_60m_repa_l9_256_per_sample",
        True,
        9,
        [None],
    ),
    # ── n=256 ESM2 encoder ablation ──────────────────────────────────────── #
    # Only on-disk 256 ESM run is L9 + t=30 conditioning, trained at bs=12
    # throughout (intentionally, via pdb_lmdb_256_bs12 — ESM-650M OOMs at
    # bs=24). Layer mismatch (L9-t30 ≠ the L4 default for the encoder block)
    # is footnoted in the table, not fixed here. Last-EMA per user choice.
    "repa_esm_l9_t30_256_steplast": (
        "proteina_60m_repa_esm_l9_t30_256_per_residue",
        True,
        9,
        [None],
    ),
    # ── n=256 ProteinMPNN encoder ablation ───────────────────────────────── #
    # ProteinMPNN target encoder, bs=24 nominal, started 2026-05-06. Step=300K
    # is the latest numbered EMA snapshot (May 9 12:46). 400K snapshot will
    # land later; add a `_step400k` entry when it does for parity with the
    # CA-GearNet L4 ep22 anchor at step=400K.
    "repa_mpnn_l4_256_per_residue_step300k": (
        "proteina_60m_repa_mpnn_l4_256_per_residue",
        True,
        4,
        [300000],
    ),
    # ── n=256 λ ablation (REPA L4, PDB, GearNet-CA, per_residue, varying λ) ─ #
    # λ=0.5 leg uses existing `repa_l4_256_ep22` (step=400K). λ=1.0 and λ=2.0
    # are still training; pinned at the latest numbered EMA snapshot available
    # (lambda1@300K, lambda2@200K). Not step-matched to the λ=0.5 anchor —
    # footnote in the table when consolidating, and replace with _step400k
    # entries once those snapshots land.
    "repa_l4_256_per_residue_lambda1_step300k": (
        "proteina_60m_repa_l4_256_per_residue_lambda1",
        True,
        4,
        [300000],
    ),
    "repa_l4_256_per_residue_lambda2_step200k": (
        "proteina_60m_repa_l4_256_per_residue_lambda2",
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
    "repa_l0_128_bs80_step200k": (
        "proteina_60m_repa_l0_128_per_residue_bs80",
        True,
        0,
        [200000],
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
    "repa_l9_128_bs80_step200k": (
        "proteina_60m_repa_l9_128_per_residue_bs80",
        True,
        9,
        [200000],
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
    # ── n=128 lambda ablation (REPA L4, bs=80, varying λ) ────────────────── #
    # λ=0.5 leg uses existing `repa_l4_128_bs80_step200k`. λ=1.0 leg
    # (`..._bs80_lambda1`) is currently parked at step ~3.5K — relaunched
    # 2026-05-08, will be added to the ablation once it has a usable ckpt.
    "repa_l4_128_bs80_lambda2_steplast": (
        "proteina_60m_repa_l4_128_per_residue_bs80_lambda2",
        True,
        4,
        [None],  # last-EMA (~step 182K, ~14.5M samples assuming clean bs=80)
    ),
    # ── n=128 weight-decay ablation (REPA L4, bs=80, wd=1e-2 vs default) ──── #
    # Pairs with `repa_l4_128_bs80_step200k` (default wd) at matching step
    # for a clean wd ablation. Run dir uses `wd1e-2` (with dash); registry
    # key drops the dash for downstream-tooling friendliness.
    "repa_l4_128_bs80_wd1e2_step200k": (
        "proteina_60m_repa_l4_128_per_residue_bs80_wd1e-2",
        True,
        4,
        [200000],
    ),
    # ── n=128 layer ablation (per_residue, mixed bs=24→80, step=400k) ────── #
    # The original L0/L4/L9 per_residue runs predate the explicit `_bs24` /
    # `_bs80` split. All three share the SAME mixed-bs schedule (bs=24 for
    # the first 220k steps, then bs=80) so cross-layer comparison is fair
    # despite the bump. Sample budget at step=400k: 220k×24 + 180k×80 =
    # 19.68M (same convention as the original n=128 sample-matched note).
    "repa_l0_128_per_residue_step400k": (
        "proteina_60m_repa_l0_128_per_residue",
        True,
        0,
        [400000],
    ),
    "repa_l4_128_per_residue_step400k": (
        "proteina_60m_repa_l4_128_per_residue",
        True,
        4,
        [400000],
    ),
    "repa_l9_128_per_residue_step400k": (
        "proteina_60m_repa_l9_128_per_residue",
        True,
        9,
        [400000],
    ),
    # ── External pretrained reference (NVIDIA NGC DFS-60M v1.3) ──────────── #
    # 12-layer ProteinTransformerAF3 (vs our 10-layer in-house 60M); released
    # global_step=1.3M, epoch=177. Same path as PRETRAINED_CHECKPOINTS, but
    # exposed as a normal RUN_SCHEDULES entry so the n=128 paper orchestrator
    # treats it as one more cell. Symlinked at
    # ``$STORE_ROOT/proteina_pretrained_dfs_60m/checkpoints/last-EMA.ckpt``
    # → .local_ckpts/proteina_v1.3_DFS_60M_notri.ckpt so find_checkpoint_path
    # resolves cleanly with step=None. is_repa=False (load as plain Proteina).
    "pretrained_dfs_60m_n128_paper": (
        "proteina_pretrained_dfs_60m",
        False,
        4,  # repa_layer unused for non-REPA loads
        [None],
    ),
    "pretrained_dfs_60m_n256_paper": (
        "proteina_pretrained_dfs_60m",
        False,
        4,  # repa_layer unused for non-REPA loads
        [None],
    ),
    # ── n=128 lambda ablation: λ=0.25 and λ=1.0 legs at step=200K ────────── #
    # Companions to existing λ=0.5 (repa_l4_128_bs80_step200k) and λ=2.0
    # (repa_l4_128_bs80_lambda2_steplast). Both stores reached 200K on
    # 2026-05-14; eval'd inline with novelty_foldseek via n128_paper_lambda_ext.
    "repa_l4_128_bs80_lambda025_step200k": (
        "proteina_60m_repa_l4_128_per_residue_bs80_lambda025",
        True,
        4,
        [200000],
    ),
    "repa_l4_128_bs80_lambda1_step200k": (
        "proteina_60m_repa_l4_128_per_residue_bs80_lambda1",
        True,
        4,
        [200000],
    ),
    # ── n=128 AFDB-Swissprot training (2-GPU bs80 via per-GPU 40 × 2 GPU) ── #
    # Mirrors n256 AFDB block. All three at step=200K; baseline + mpnn-L4
    # additionally at step=400K (their stores reached 500K by 2026-05-14).
    "baseline_afdb_128_bs80_step200k": (
        "proteina_60m_baseline_afdb_128_bs80_2gpu",
        False,
        4,
        [200000],
    ),
    "baseline_afdb_128_bs80_step400k": (
        "proteina_60m_baseline_afdb_128_bs80_2gpu",
        False,
        4,
        [400000],
    ),
    "repa_l4_afdb_128_bs80_step200k": (
        "proteina_60m_repa_l4_128_afdb_per_residue_bs80_2gpu",
        True,
        4,
        [200000],
    ),
    "repa_mpnn_l4_afdb_128_bs80_step200k": (
        "proteina_60m_repa_mpnn_l4_128_afdb_per_residue_bs80_2gpu",
        True,
        4,
        [200000],
    ),
    "repa_mpnn_l4_afdb_128_bs80_step400k": (
        "proteina_60m_repa_mpnn_l4_128_afdb_per_residue_bs80_2gpu",
        True,
        4,
        [400000],
    ),
    # n=128 AFDB extension (2026-05-16): close the 4-way comparison at 600K
    # (max for repa_l4_afdb), then long-tail at 800K/1M for baseline + mpnn-L4
    # and 1200K for baseline only (mpnn-L4 store maxes at 1100K).
    "baseline_afdb_128_bs80_step600k": (
        "proteina_60m_baseline_afdb_128_bs80_2gpu",
        False,
        4,
        [600000],
    ),
    "baseline_afdb_128_bs80_step800k": (
        "proteina_60m_baseline_afdb_128_bs80_2gpu",
        False,
        4,
        [800000],
    ),
    "baseline_afdb_128_bs80_step1000k": (
        "proteina_60m_baseline_afdb_128_bs80_2gpu",
        False,
        4,
        [1000000],
    ),
    "baseline_afdb_128_bs80_step1200k": (
        "proteina_60m_baseline_afdb_128_bs80_2gpu",
        False,
        4,
        [1200000],
    ),
    "repa_l4_afdb_128_bs80_step600k": (
        "proteina_60m_repa_l4_128_afdb_per_residue_bs80_2gpu",
        True,
        4,
        [600000],
    ),
    "repa_mpnn_l4_afdb_128_bs80_step600k": (
        "proteina_60m_repa_mpnn_l4_128_afdb_per_residue_bs80_2gpu",
        True,
        4,
        [600000],
    ),
    "repa_mpnn_l4_afdb_128_bs80_step800k": (
        "proteina_60m_repa_mpnn_l4_128_afdb_per_residue_bs80_2gpu",
        True,
        4,
        [800000],
    ),
    "repa_mpnn_l4_afdb_128_bs80_step1000k": (
        "proteina_60m_repa_mpnn_l4_128_afdb_per_residue_bs80_2gpu",
        True,
        4,
        [1000000],
    ),
    # ── n=128 convergence sweep (added 2026-05-17) ─────────────────────────── #
    # Companion to n=256 convergence. PDB uses dense @100k cadence (max horizon
    # 700k); AFDB uses log spacing {100,200,400,700,1000} matching n256_convergence.
    # 6-config PDB matrix (baseline + L4/L9 × {gn, mpnn} + L4 random-gn) and
    # 4-config AFDB matrix (baseline + L4-gn + L4/L9-mpnn). Steps strictly
    # above each run's max are skipped (no snap-to-last).
    # ---- PDB: baseline_128_bs80 (max 700k, last-EMA 714,000) ---- #
    "baseline_128_bs80_step100k": (
        "proteina_60m_baseline_128_bs80",
        False,
        4,
        [100000],
    ),
    "baseline_128_bs80_step300k": (
        "proteina_60m_baseline_128_bs80",
        False,
        4,
        [300000],
    ),
    "baseline_128_bs80_step400k": (
        "proteina_60m_baseline_128_bs80",
        False,
        4,
        [400000],
    ),
    "baseline_128_bs80_step500k": (
        "proteina_60m_baseline_128_bs80",
        False,
        4,
        [500000],
    ),
    "baseline_128_bs80_step600k": (
        "proteina_60m_baseline_128_bs80",
        False,
        4,
        [600000],
    ),
    "baseline_128_bs80_step700k": (
        "proteina_60m_baseline_128_bs80",
        False,
        4,
        [700000],
    ),
    # ---- PDB: repa_l4_gn_bs80 (max 500k, last-EMA 514,500) ---- #
    "repa_l4_128_bs80_step100k": (
        "proteina_60m_repa_l4_128_per_residue_bs80",
        True,
        4,
        [100000],
    ),
    "repa_l4_128_bs80_step300k": (
        "proteina_60m_repa_l4_128_per_residue_bs80",
        True,
        4,
        [300000],
    ),
    "repa_l4_128_bs80_step400k": (
        "proteina_60m_repa_l4_128_per_residue_bs80",
        True,
        4,
        [400000],
    ),
    "repa_l4_128_bs80_step500k": (
        "proteina_60m_repa_l4_128_per_residue_bs80",
        True,
        4,
        [500000],
    ),
    # ---- PDB: repa_l9_gn_bs80 (max 400k, last-EMA 483,000) ---- #
    "repa_l9_128_bs80_step100k": (
        "proteina_60m_repa_l9_128_per_residue_bs80",
        True,
        9,
        [100000],
    ),
    "repa_l9_128_bs80_step300k": (
        "proteina_60m_repa_l9_128_per_residue_bs80",
        True,
        9,
        [300000],
    ),
    "repa_l9_128_bs80_step400k": (
        "proteina_60m_repa_l9_128_per_residue_bs80",
        True,
        9,
        [400000],
    ),
    # ---- PDB: repa_mpnn_l4_bs80 (max 200k, last-EMA 231,000; short anchor) ---- #
    "repa_mpnn_l4_128_bs80_step100k": (
        "proteina_60m_repa_mpnn_l4_128_per_residue_bs80",
        True,
        4,
        [100000],
    ),
    # ---- PDB: repa_mpnn_l9_bs80_2gpu (new run, max 500k, last-EMA 518,000) ---- #
    "repa_mpnn_l9_128_bs80_2gpu_step100k": (
        "proteina_60m_repa_mpnn_l9_128_per_residue_bs80_2gpu",
        True,
        9,
        [100000],
    ),
    "repa_mpnn_l9_128_bs80_2gpu_step200k": (
        "proteina_60m_repa_mpnn_l9_128_per_residue_bs80_2gpu",
        True,
        9,
        [200000],
    ),
    "repa_mpnn_l9_128_bs80_2gpu_step300k": (
        "proteina_60m_repa_mpnn_l9_128_per_residue_bs80_2gpu",
        True,
        9,
        [300000],
    ),
    "repa_mpnn_l9_128_bs80_2gpu_step400k": (
        "proteina_60m_repa_mpnn_l9_128_per_residue_bs80_2gpu",
        True,
        9,
        [400000],
    ),
    "repa_mpnn_l9_128_bs80_2gpu_step500k": (
        "proteina_60m_repa_mpnn_l9_128_per_residue_bs80_2gpu",
        True,
        9,
        [500000],
    ),
    # ---- PDB: repa_l4_random (rand-gn, max 400k, last-EMA 490,000) ---- #
    "repa_l4_128_random_step100k": (
        "proteina_60m_repa_l4_128_per_residue_random",
        True,
        4,
        [100000],
    ),
    "repa_l4_128_random_step300k": (
        "proteina_60m_repa_l4_128_per_residue_random",
        True,
        4,
        [300000],
    ),
    "repa_l4_128_random_step400k": (
        "proteina_60m_repa_l4_128_per_residue_random",
        True,
        4,
        [400000],
    ),
    # ---- AFDB: baseline (max 1200k); need 100, 700 (200/400/1000 already exist) ---- #
    "baseline_afdb_128_bs80_step100k": (
        "proteina_60m_baseline_afdb_128_bs80_2gpu",
        False,
        4,
        [100000],
    ),
    "baseline_afdb_128_bs80_step700k": (
        "proteina_60m_baseline_afdb_128_bs80_2gpu",
        False,
        4,
        [700000],
    ),
    # ---- AFDB: repa_l4_gn (max 600k); need 100, 400 (200/600 already exist) ---- #
    "repa_l4_afdb_128_bs80_step100k": (
        "proteina_60m_repa_l4_128_afdb_per_residue_bs80_2gpu",
        True,
        4,
        [100000],
    ),
    "repa_l4_afdb_128_bs80_step400k": (
        "proteina_60m_repa_l4_128_afdb_per_residue_bs80_2gpu",
        True,
        4,
        [400000],
    ),
    # ---- AFDB: repa_mpnn_l4 (max 1100k); need 100, 700 (200/400/600/800/1000 exist) ---- #
    "repa_mpnn_l4_afdb_128_bs80_step100k": (
        "proteina_60m_repa_mpnn_l4_128_afdb_per_residue_bs80_2gpu",
        True,
        4,
        [100000],
    ),
    "repa_mpnn_l4_afdb_128_bs80_step700k": (
        "proteina_60m_repa_mpnn_l4_128_afdb_per_residue_bs80_2gpu",
        True,
        4,
        [700000],
    ),
    # ---- AFDB: repa_mpnn_l9 (new run, max 400k, last-EMA 490,000) ---- #
    "repa_mpnn_l9_afdb_128_bs80_2gpu_step100k": (
        "proteina_60m_repa_mpnn_l9_128_afdb_per_residue_bs80_2gpu",
        True,
        9,
        [100000],
    ),
    "repa_mpnn_l9_afdb_128_bs80_2gpu_step200k": (
        "proteina_60m_repa_mpnn_l9_128_afdb_per_residue_bs80_2gpu",
        True,
        9,
        [200000],
    ),
    "repa_mpnn_l9_afdb_128_bs80_2gpu_step400k": (
        "proteina_60m_repa_mpnn_l9_128_afdb_per_residue_bs80_2gpu",
        True,
        9,
        [400000],
    ),
    # ── n=256 AFDB extension: step=400K/700K/900K snapshots ──────────────── #
    # Existing baseline_afdb_256_ep20 / repa_l4_afdb_256_ep20 are pinned at
    # step=200K (epoch 20). These step-keyed companions sample further along
    # the same stores for a step-vs-FID curve, and add the mpnn-aligned
    # encoder runs (mpnn-L4 step400k, mpnn-L9 step400k/700k) that didn't
    # exist when n256_paper_afdb was first run.
    "baseline_afdb_256_step400k": (
        "proteina_60m_baseline_afdb_swissprot_256",
        False,
        4,
        [400000],
    ),
    "baseline_afdb_256_step700k": (
        "proteina_60m_baseline_afdb_swissprot_256",
        False,
        4,
        [700000],
    ),
    "baseline_afdb_256_step900k": (
        "proteina_60m_baseline_afdb_swissprot_256",
        False,
        4,
        [900000],
    ),
    "repa_l4_afdb_256_step400k": (
        "proteina_60m_repa_l4_256_afdb_per_residue",
        True,
        4,
        [400000],
    ),
    "repa_l4_afdb_256_step700k": (
        "proteina_60m_repa_l4_256_afdb_per_residue",
        True,
        4,
        [700000],
    ),
    "repa_mpnn_l4_afdb_256_step400k": (
        "proteina_60m_repa_mpnn_l4_256_afdb_per_residue",
        True,
        4,
        [400000],
    ),
    "repa_mpnn_l9_afdb_256_step400k": (
        "proteina_60m_repa_mpnn_l9_256_afdb_per_residue",
        True,
        9,
        [400000],
    ),
    "repa_mpnn_l9_afdb_256_step700k": (
        "proteina_60m_repa_mpnn_l9_256_afdb_per_residue",
        True,
        9,
        [700000],
    ),
    # CA-GearNet L9 AFDB-256 (added 2026-05-16): periodic ckpts 100k–400k @100k,
    # last-EMA at 423.5k. Shortest n=256 AFDB run; not previously swept.
    "repa_l9_afdb_256_step200k": (
        "proteina_60m_repa_l9_256_afdb_per_residue",
        True,
        9,
        [200000],
    ),
    "repa_l9_afdb_256_step400k": (
        "proteina_60m_repa_l9_256_afdb_per_residue",
        True,
        9,
        [400000],
    ),
    "repa_l9_afdb_256_steplast": (
        "proteina_60m_repa_l9_256_afdb_per_residue",
        True,
        9,
        [None],
    ),
    # ── n=256 lambda extension: λ=1.0@step200K, λ=2.0@step300K ───────────── #
    # Step-matched companions: λ=1 already had _step300k entry above; pair it
    # with a _step200k snapshot. λ=2 already had _step200k; pair with
    # _step300k (latest numbered ckpt as of 2026-05-14).
    "repa_l4_256_per_residue_lambda1_step200k": (
        "proteina_60m_repa_l4_256_per_residue_lambda1",
        True,
        4,
        [200000],
    ),
    "repa_l4_256_per_residue_lambda2_step300k": (
        "proteina_60m_repa_l4_256_per_residue_lambda2",
        True,
        4,
        [300000],
    ),
    # ── n=256 convergence sweep (added 2026-05-16) ──────────────────────── #
    # Log-spaced step curve at {100k, 200k, 400k, 700k, 1000k, 1300k, 1600k}
    # for the 9-config matrix: {PDB, AFDB} × baseline + {PDB, AFDB} × {L4, L9}
    # × {GearNet, MPNN}, minus PDB L9 MPNN (no periodic ckpt). Steps above each
    # run's max are skipped (no snap-to-last). 47 unique ckpts total; this block
    # adds the 38 missing — the other 9 reuse earlier _step{400,700,900,1000}k
    # entries (afdb baseline/l4-gn) and _step200k AFDB L9 entries.
    "baseline_256_bs24_2gpu_step100k": (
        "proteina_60m_baseline_256_bs24_2gpu",
        False,
        4,
        [100000],
    ),
    "baseline_256_bs24_2gpu_step200k": (
        "proteina_60m_baseline_256_bs24_2gpu",
        False,
        4,
        [200000],
    ),
    "baseline_256_bs24_2gpu_step400k": (
        "proteina_60m_baseline_256_bs24_2gpu",
        False,
        4,
        [400000],
    ),
    "baseline_256_bs24_2gpu_step700k": (
        "proteina_60m_baseline_256_bs24_2gpu",
        False,
        4,
        [700000],
    ),
    # Added 2026-05-27 to fill the 700K-1M crossover gap (baseline only had
    # 700K then 1000K evaluated; T-D crossover lives in this window).
    "baseline_256_bs24_2gpu_step800k": (
        "proteina_60m_baseline_256_bs24_2gpu",
        False,
        4,
        [800000],
    ),
    "baseline_256_bs24_2gpu_step900k": (
        "proteina_60m_baseline_256_bs24_2gpu",
        False,
        4,
        [900000],
    ),
    "baseline_256_bs24_2gpu_step1000k": (
        "proteina_60m_baseline_256_bs24_2gpu",
        False,
        4,
        [1000000],
    ),
    "baseline_256_bs24_2gpu_step1300k": (
        "proteina_60m_baseline_256_bs24_2gpu",
        False,
        4,
        [1300000],
    ),
    "repa_l4_256_per_residue_bs24_2gpu_step100k": (
        "proteina_60m_repa_l4_256_per_residue_bs24_2gpu",
        True,
        4,
        [100000],
    ),
    "repa_l4_256_per_residue_bs24_2gpu_step200k": (
        "proteina_60m_repa_l4_256_per_residue_bs24_2gpu",
        True,
        4,
        [200000],
    ),
    "repa_l4_256_per_residue_bs24_2gpu_step400k": (
        "proteina_60m_repa_l4_256_per_residue_bs24_2gpu",
        True,
        4,
        [400000],
    ),
    "repa_l4_256_per_residue_bs24_2gpu_step700k": (
        "proteina_60m_repa_l4_256_per_residue_bs24_2gpu",
        True,
        4,
        [700000],
    ),
    "repa_mpnn_l4_256_per_residue_step100k": (
        "proteina_60m_repa_mpnn_l4_256_per_residue",
        True,
        4,
        [100000],
    ),
    "repa_mpnn_l4_256_per_residue_step200k": (
        "proteina_60m_repa_mpnn_l4_256_per_residue",
        True,
        4,
        [200000],
    ),
    "repa_mpnn_l4_256_per_residue_step400k": (
        "proteina_60m_repa_mpnn_l4_256_per_residue",
        True,
        4,
        [400000],
    ),
    "repa_mpnn_l4_256_per_residue_step700k": (
        "proteina_60m_repa_mpnn_l4_256_per_residue",
        True,
        4,
        [700000],
    ),
    # Added 2026-05-27 to fill 700K-1M crossover gap.
    "repa_mpnn_l4_256_per_residue_step800k": (
        "proteina_60m_repa_mpnn_l4_256_per_residue",
        True,
        4,
        [800000],
    ),
    "repa_mpnn_l4_256_per_residue_step900k": (
        "proteina_60m_repa_mpnn_l4_256_per_residue",
        True,
        4,
        [900000],
    ),
    "repa_mpnn_l4_256_per_residue_step1000k": (
        "proteina_60m_repa_mpnn_l4_256_per_residue",
        True,
        4,
        [1000000],
    ),
    "repa_mpnn_l4_256_per_residue_step1300k": (
        "proteina_60m_repa_mpnn_l4_256_per_residue",
        True,
        4,
        [1300000],
    ),
    "repa_mpnn_l4_256_per_residue_step1600k": (
        "proteina_60m_repa_mpnn_l4_256_per_residue",
        True,
        4,
        [1600000],
    ),
    "repa_l9_256_per_residue_bs24_2gpu_step100k": (
        "proteina_60m_repa_l9_256_per_residue_bs24_2gpu",
        True,
        9,
        [100000],
    ),
    "repa_l9_256_per_residue_bs24_2gpu_step200k": (
        "proteina_60m_repa_l9_256_per_residue_bs24_2gpu",
        True,
        9,
        [200000],
    ),
    "repa_l9_256_per_residue_bs24_2gpu_step400k": (
        "proteina_60m_repa_l9_256_per_residue_bs24_2gpu",
        True,
        9,
        [400000],
    ),
    "repa_l9_256_per_residue_bs24_2gpu_step700k": (
        "proteina_60m_repa_l9_256_per_residue_bs24_2gpu",
        True,
        9,
        [700000],
    ),
    "baseline_afdb_256_step100k": (
        "proteina_60m_baseline_afdb_swissprot_256",
        False,
        4,
        [100000],
    ),
    "baseline_afdb_256_step200k": (
        "proteina_60m_baseline_afdb_swissprot_256",
        False,
        4,
        [200000],
    ),
    "baseline_afdb_256_step1000k": (
        "proteina_60m_baseline_afdb_swissprot_256",
        False,
        4,
        [1000000],
    ),
    "baseline_afdb_256_step1300k": (
        "proteina_60m_baseline_afdb_swissprot_256",
        False,
        4,
        [1300000],
    ),
    "baseline_afdb_256_step1600k": (
        "proteina_60m_baseline_afdb_swissprot_256",
        False,
        4,
        [1600000],
    ),
    "repa_l4_afdb_256_step100k": (
        "proteina_60m_repa_l4_256_afdb_per_residue",
        True,
        4,
        [100000],
    ),
    "repa_l4_afdb_256_step200k": (
        "proteina_60m_repa_l4_256_afdb_per_residue",
        True,
        4,
        [200000],
    ),
    "repa_l4_afdb_256_step1000k": (
        "proteina_60m_repa_l4_256_afdb_per_residue",
        True,
        4,
        [1000000],
    ),
    "repa_mpnn_l4_afdb_256_step100k": (
        "proteina_60m_repa_mpnn_l4_256_afdb_per_residue",
        True,
        4,
        [100000],
    ),
    "repa_mpnn_l4_afdb_256_step200k": (
        "proteina_60m_repa_mpnn_l4_256_afdb_per_residue",
        True,
        4,
        [200000],
    ),
    "repa_mpnn_l4_afdb_256_step700k": (
        "proteina_60m_repa_mpnn_l4_256_afdb_per_residue",
        True,
        4,
        [700000],
    ),
    "repa_mpnn_l4_afdb_256_step1000k": (
        "proteina_60m_repa_mpnn_l4_256_afdb_per_residue",
        True,
        4,
        [1000000],
    ),
    "repa_l9_afdb_256_step100k": (
        "proteina_60m_repa_l9_256_afdb_per_residue",
        True,
        9,
        [100000],
    ),
    "repa_mpnn_l9_afdb_256_step100k": (
        "proteina_60m_repa_mpnn_l9_256_afdb_per_residue",
        True,
        9,
        [100000],
    ),
    "repa_mpnn_l9_afdb_256_step200k": (
        "proteina_60m_repa_mpnn_l9_256_afdb_per_residue",
        True,
        9,
        [200000],
    ),
    "repa_mpnn_l9_afdb_256_step1000k": (
        "proteina_60m_repa_mpnn_l9_256_afdb_per_residue",
        True,
        9,
        [1000000],
    ),
    "repa_mpnn_l9_afdb_256_step1300k": (
        "proteina_60m_repa_mpnn_l9_256_afdb_per_residue",
        True,
        9,
        [1300000],
    ),
    # ── n=256 convergence sweep extension (added 2026-05-17) ────────────── #
    # Fills gaps that emerged after the initial sweep:
    #   - Runs that have trained further since 2026-05-16 (new ckpts landed).
    #   - The PDB repa_mpnn_l9 config that had no periodic ckpt at sweep time
    #     (only 31.5k last-EMA); now has 100k–500k periodic.
    #   - Adds the L4-random-GearNet PDB control run (was not in original sweep).
    # PDB additions:
    "baseline_256_bs24_2gpu_step1500k": (
        "proteina_60m_baseline_256_bs24_2gpu",
        False,
        4,
        [1500000],
    ),
    "repa_l4_256_per_residue_bs24_2gpu_step800k": (
        "proteina_60m_repa_l4_256_per_residue_bs24_2gpu",
        True,
        4,
        [800000],
    ),
    "repa_l4_256_per_residue_bs24_2gpu_step900k": (
        "proteina_60m_repa_l4_256_per_residue_bs24_2gpu",
        True,
        4,
        [900000],
    ),
    "repa_l9_256_per_residue_bs24_2gpu_step800k": (
        "proteina_60m_repa_l9_256_per_residue_bs24_2gpu",
        True,
        9,
        [800000],
    ),
    "repa_l9_256_per_residue_bs24_2gpu_step900k": (
        "proteina_60m_repa_l9_256_per_residue_bs24_2gpu",
        True,
        9,
        [900000],
    ),
    "repa_mpnn_l9_256_per_residue_step100k": (
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        True,
        9,
        [100000],
    ),
    "repa_mpnn_l9_256_per_residue_step200k": (
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        True,
        9,
        [200000],
    ),
    "repa_mpnn_l9_256_per_residue_step400k": (
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        True,
        9,
        [400000],
    ),
    "repa_l4_256_per_residue_random_bs24_2gpu_step100k": (
        "proteina_60m_repa_l4_256_per_residue_random_bs24_2gpu",
        True,
        4,
        [100000],
    ),
    "repa_l4_256_per_residue_random_bs24_2gpu_step200k": (
        "proteina_60m_repa_l4_256_per_residue_random_bs24_2gpu",
        True,
        4,
        [200000],
    ),
    "repa_l4_256_per_residue_random_bs24_2gpu_step400k": (
        "proteina_60m_repa_l4_256_per_residue_random_bs24_2gpu",
        True,
        4,
        [400000],
    ),
    "repa_l4_256_per_residue_random_bs24_2gpu_step700k": (
        "proteina_60m_repa_l4_256_per_residue_random_bs24_2gpu",
        True,
        4,
        [700000],
    ),
    # AFDB additions:
    "repa_l9_afdb_256_step500k": (
        "proteina_60m_repa_l9_256_afdb_per_residue",
        True,
        9,
        [500000],
    ),
    "repa_l9_afdb_256_step600k": (
        "proteina_60m_repa_l9_256_afdb_per_residue",
        True,
        9,
        [600000],
    ),
    "repa_l4_afdb_256_step1200k": (
        "proteina_60m_repa_l4_256_afdb_per_residue",
        True,
        4,
        [1200000],
    ),
    "repa_mpnn_l4_afdb_256_step1100k": (
        "proteina_60m_repa_mpnn_l4_256_afdb_per_residue",
        True,
        4,
        [1100000],
    ),
    "repa_mpnn_l9_afdb_256_step1400k": (
        "proteina_60m_repa_mpnn_l9_256_afdb_per_residue",
        True,
        9,
        [1400000],
    ),
    "repa_mpnn_l9_afdb_256_step1500k": (
        "proteina_60m_repa_mpnn_l9_256_afdb_per_residue",
        True,
        9,
        [1500000],
    ),
    # ── n=256 convergence extension 2 (added 2026-05-25) — past-horizon ckpts ──
    "repa_l4_256_per_residue_bs24_2gpu_step1000k": (
        "proteina_60m_repa_l4_256_per_residue_bs24_2gpu",
        True,
        4,
        [1000000],
    ),
    "repa_l9_256_per_residue_bs24_2gpu_step1000k": (
        "proteina_60m_repa_l9_256_per_residue_bs24_2gpu",
        True,
        9,
        [1000000],
    ),
    "repa_l9_256_per_residue_bs24_2gpu_step1100k": (
        "proteina_60m_repa_l9_256_per_residue_bs24_2gpu",
        True,
        9,
        [1100000],
    ),
    "repa_mpnn_l9_256_per_residue_step800k": (
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        True,
        9,
        [800000],
    ),
    "repa_mpnn_l9_256_per_residue_step900k": (
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        True,
        9,
        [900000],
    ),
    "repa_mpnn_l9_256_per_residue_step1000k": (
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        True,
        9,
        [1000000],
    ),
    "repa_mpnn_l9_256_per_residue_step1100k": (
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        True,
        9,
        [1100000],
    ),
    "repa_mpnn_l9_256_per_residue_step1200k": (
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        True,
        9,
        [1200000],
    ),
    "repa_l9_afdb_256_step700k": (
        "proteina_60m_repa_l9_256_afdb_per_residue",
        True,
        9,
        [700000],
    ),
    "repa_l9_afdb_256_step800k": (
        "proteina_60m_repa_l9_256_afdb_per_residue",
        True,
        9,
        [800000],
    ),
    # ── n=128 PDB convergence extension (added 2026-05-25) — past-horizon ─── #
    "repa_l4_128_bs80_step600k": (
        "proteina_60m_repa_l4_128_per_residue_bs80",
        True,
        4,
        [600000],
    ),
    "repa_l4_128_random_step500k": (
        "proteina_60m_repa_l4_128_per_residue_random",
        True,
        4,
        [500000],
    ),
    "repa_l4_128_random_step600k": (
        "proteina_60m_repa_l4_128_per_residue_random",
        True,
        4,
        [600000],
    ),
    "repa_l9_128_bs80_step500k": (
        "proteina_60m_repa_l9_128_per_residue_bs80",
        True,
        9,
        [500000],
    ),
    "repa_mpnn_l9_128_bs80_2gpu_step600k": (
        "proteina_60m_repa_mpnn_l9_128_per_residue_bs80_2gpu",
        True,
        9,
        [600000],
    ),
    "repa_mpnn_l9_128_bs80_2gpu_step700k": (
        "proteina_60m_repa_mpnn_l9_128_per_residue_bs80_2gpu",
        True,
        9,
        [700000],
    ),
    # ── n=256 convergence extension 3 (added 2026-05-25) ────────────────────
    # Past-horizon ckpts only: runs that have trained beyond their previously-
    # evaluated max step. Intermediate gaps deliberately left out — we extend
    # the curve, not densify it.
    "baseline_256_bs24_2gpu_step1600k": (
        "proteina_60m_baseline_256_bs24_2gpu",
        False,
        4,
        [1600000],
    ),
    "baseline_afdb_256_step1700k": (
        "proteina_60m_baseline_afdb_swissprot_256",
        False,
        4,
        [1700000],
    ),
    "repa_l9_afdb_256_step900k": (
        "proteina_60m_repa_l9_256_afdb_per_residue",
        True,
        9,
        [900000],
    ),
    # ── n=128 AFDB convergence extension 3 (added 2026-05-25) ───────────────
    # Past-horizon ckpts for AFDB-128 runs that trained beyond ext2 horizon.
    "baseline_afdb_128_bs80_step1100k": (
        "proteina_60m_baseline_afdb_128_bs80_2gpu",
        False,
        4,
        [1100000],
    ),
    "repa_l4_afdb_128_bs80_step500k": (
        "proteina_60m_repa_l4_128_afdb_per_residue_bs80_2gpu",
        True,
        4,
        [500000],
    ),
    "repa_mpnn_l4_afdb_128_bs80_step1100k": (
        "proteina_60m_repa_mpnn_l4_128_afdb_per_residue_bs80_2gpu",
        True,
        4,
        [1100000],
    ),
    "repa_mpnn_l9_afdb_128_bs80_2gpu_step500k": (
        "proteina_60m_repa_mpnn_l9_128_afdb_per_residue_bs80_2gpu",
        True,
        9,
        [500000],
    ),
    "repa_mpnn_l9_afdb_128_bs80_2gpu_step600k": (
        "proteina_60m_repa_mpnn_l9_128_afdb_per_residue_bs80_2gpu",
        True,
        9,
        [600000],
    ),
    # ── n=256 convergence extension 4 (added 2026-05-26) ────────────────────
    # Past-horizon ckpts that landed while ext3 was evaluating — the
    # currently-training jobs ticked over their next 100k boundary.
    "baseline_256_bs24_2gpu_step1700k": (
        "proteina_60m_baseline_256_bs24_2gpu",
        False,
        4,
        [1700000],
    ),
    "repa_l9_256_per_residue_bs24_2gpu_step1200k": (
        "proteina_60m_repa_l9_256_per_residue_bs24_2gpu",
        True,
        9,
        [1200000],
    ),
    "repa_mpnn_l9_256_per_residue_step1300k": (
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        True,
        9,
        [1300000],
    ),
    "repa_l4_256_per_residue_random_bs24_2gpu_step800k": (
        "proteina_60m_repa_l4_256_per_residue_random_bs24_2gpu",
        True,
        4,
        [800000],
    ),
    "baseline_afdb_256_step1800k": (
        "proteina_60m_baseline_afdb_swissprot_256",
        False,
        4,
        [1800000],
    ),
    "repa_l4_afdb_256_step1300k": (
        "proteina_60m_repa_l4_256_afdb_per_residue",
        True,
        4,
        [1300000],
    ),
    # ── n=256 convergence extension 5 (added 2026-05-26) ────────────────────
    # Newest 100k-boundary ckpts on disk; not yet in any prior ext profile.
    "baseline_256_bs24_2gpu_step1800k": (
        "proteina_60m_baseline_256_bs24_2gpu",
        False,
        4,
        [1800000],
    ),
    "repa_mpnn_l9_256_per_residue_step1400k": (
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        True,
        9,
        [1400000],
    ),
    # ── n=256 convergence extension 6 (added 2026-05-28) ────────────────────
    # L9-GN stopped at 1.4M; L9-MPNN continued to 1.7M.
    "repa_l9_256_per_residue_bs24_2gpu_step1300k": (
        "proteina_60m_repa_l9_256_per_residue_bs24_2gpu",
        True,
        9,
        [1300000],
    ),
    "repa_l9_256_per_residue_bs24_2gpu_step1400k": (
        "proteina_60m_repa_l9_256_per_residue_bs24_2gpu",
        True,
        9,
        [1400000],
    ),
    "repa_mpnn_l9_256_per_residue_step1500k": (
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        True,
        9,
        [1500000],
    ),
    "repa_mpnn_l9_256_per_residue_step1600k": (
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        True,
        9,
        [1600000],
    ),
    "repa_mpnn_l9_256_per_residue_step1700k": (
        "proteina_60m_repa_mpnn_l9_256_per_residue",
        True,
        9,
        [1700000],
    ),
    "repa_l9_afdb_256_step1000k": (
        "proteina_60m_repa_l9_256_afdb_per_residue",
        True,
        9,
        [1000000],
    ),
    # First eval anchor for the AFDB random-init-GearNet run (training started
    # 2026-05; only 100k ckpt exists at submission time).
    "repa_l4_afdb_256_random_step100k": (
        "proteina_60m_repa_l4_256_afdb_per_residue_random_bs24_2gpu",
        True,
        4,
        [100000],
    ),
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
    "repa_l4_256_ep13_step300k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_ep31_step500k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_random_ep17": "inference/paper/inference_fid_60m_paper",
    "baseline_afdb_256_ep20": "inference/paper/inference_fid_60m_paper",
    "repa_l4_afdb_256_ep20": "inference/paper/inference_fid_60m_paper",
    "repa_l0_256_per_sample_steplast": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_sample_steplast": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_sample_step400k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_256_per_sample_steplast": "inference/paper/inference_fid_60m_paper",
    "repa_esm_l9_t30_256_steplast": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_256_per_residue_step300k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_residue_lambda1_step300k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_residue_lambda2_step200k": "inference/paper/inference_fid_60m_paper",
    # ── n=128 paper protocol (truncated to in-distribution lengths) ──────── #
    # All point at inference/inference_fid_60m_n128_paper which generates 500
    # PDBs at lengths {50, 75, 100, 125} × 125 each (vs n=256's 1125 PDBs).
    # Designability subsamples 50 of 125 per length × 4 lengths = 200 PDBs.
    "baseline_128_bs80_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l0_128_bs80_steplast": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l0_128_bs80_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_bs80_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l9_128_bs80_steplast": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l9_128_bs80_step200k": "inference/paper/inference_fid_60m_n128_paper",
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
    # External pretrained reference — same n=128 paper protocol
    # ({50,75,100,125} × 125) so it sits in the same plot rows as our runs.
    "pretrained_dfs_60m_n128_paper": "inference/paper/inference_fid_60m_n128_paper",
    # n=128 per_residue layer ablation (mixed-bs runs through paper protocol)
    "repa_l0_128_per_residue_step400k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_per_residue_step400k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l9_128_per_residue_step400k": "inference/paper/inference_fid_60m_n128_paper",
    # n=128 lambda ablation (λ=2.0; λ=0.5 leg reuses repa_l4_128_bs80_step200k)
    "repa_l4_128_bs80_lambda2_steplast": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_bs80_wd1e2_step200k": "inference/paper/inference_fid_60m_n128_paper",
    # Same checkpoint, n=256 paper protocol ({50..250 step 25} × 125 = 1125 PDBs).
    "pretrained_dfs_60m_n256_paper": "inference/paper/inference_fid_60m_paper",
    # ── n=128 lambda ablation extension (λ=0.25, λ=1.0) ──────────────────── #
    "repa_l4_128_bs80_lambda025_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_bs80_lambda1_step200k": "inference/paper/inference_fid_60m_n128_paper",
    # ── n=128 AFDB-Swissprot training ────────────────────────────────────── #
    "baseline_afdb_128_bs80_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_afdb_128_bs80_step400k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_afdb_128_bs80_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l4_afdb_128_bs80_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l4_afdb_128_bs80_step400k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_afdb_128_bs80_step600k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_afdb_128_bs80_step800k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_afdb_128_bs80_step1000k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_afdb_128_bs80_step1200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_afdb_128_bs80_step600k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l4_afdb_128_bs80_step600k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l4_afdb_128_bs80_step800k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l4_afdb_128_bs80_step1000k": "inference/paper/inference_fid_60m_n128_paper",
    # ── n=128 convergence sweep (added 2026-05-17) ──────────────────────── #
    "baseline_128_bs80_step100k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_128_bs80_step300k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_128_bs80_step400k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_128_bs80_step500k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_128_bs80_step600k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_128_bs80_step700k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_bs80_step100k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_bs80_step300k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_bs80_step400k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_bs80_step500k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l9_128_bs80_step100k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l9_128_bs80_step300k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l9_128_bs80_step400k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l4_128_bs80_step100k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l9_128_bs80_2gpu_step100k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l9_128_bs80_2gpu_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l9_128_bs80_2gpu_step300k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l9_128_bs80_2gpu_step400k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l9_128_bs80_2gpu_step500k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_random_step100k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_random_step300k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_random_step400k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_afdb_128_bs80_step100k": "inference/paper/inference_fid_60m_n128_paper",
    "baseline_afdb_128_bs80_step700k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_afdb_128_bs80_step100k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_afdb_128_bs80_step400k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l4_afdb_128_bs80_step100k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l4_afdb_128_bs80_step700k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l9_afdb_128_bs80_2gpu_step100k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l9_afdb_128_bs80_2gpu_step200k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l9_afdb_128_bs80_2gpu_step400k": "inference/paper/inference_fid_60m_n128_paper",
    # ── n=256 AFDB extension (step-curve + mpnn-aligned encoders) ────────── #
    "baseline_afdb_256_step400k": "inference/paper/inference_fid_60m_paper",
    "baseline_afdb_256_step700k": "inference/paper/inference_fid_60m_paper",
    "baseline_afdb_256_step900k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_afdb_256_step400k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_afdb_256_step700k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_afdb_256_step400k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_afdb_256_step400k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_afdb_256_step700k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_afdb_256_step200k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_afdb_256_step400k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_afdb_256_steplast": "inference/paper/inference_fid_60m_paper",
    # ── n=256 lambda ablation extension ──────────────────────────────────── #
    "repa_l4_256_per_residue_lambda1_step200k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_residue_lambda2_step300k": "inference/paper/inference_fid_60m_paper",
    # ── n=256 convergence sweep (added 2026-05-16) ──────────────────────── #
    "baseline_256_bs24_2gpu_step100k": "inference/paper/inference_fid_60m_paper",
    "baseline_256_bs24_2gpu_step200k": "inference/paper/inference_fid_60m_paper",
    "baseline_256_bs24_2gpu_step400k": "inference/paper/inference_fid_60m_paper",
    "baseline_256_bs24_2gpu_step700k": "inference/paper/inference_fid_60m_paper",
    "baseline_256_bs24_2gpu_step800k": "inference/paper/inference_fid_60m_paper",
    "baseline_256_bs24_2gpu_step900k": "inference/paper/inference_fid_60m_paper",
    "baseline_256_bs24_2gpu_step1000k": "inference/paper/inference_fid_60m_paper",
    "baseline_256_bs24_2gpu_step1300k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_residue_bs24_2gpu_step100k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_residue_bs24_2gpu_step200k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_residue_bs24_2gpu_step400k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_residue_bs24_2gpu_step700k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_256_per_residue_step100k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_256_per_residue_step200k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_256_per_residue_step400k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_256_per_residue_step700k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_256_per_residue_step800k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_256_per_residue_step900k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_256_per_residue_step1000k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_256_per_residue_step1300k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_256_per_residue_step1600k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_256_per_residue_bs24_2gpu_step100k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_256_per_residue_bs24_2gpu_step200k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_256_per_residue_bs24_2gpu_step400k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_256_per_residue_bs24_2gpu_step700k": "inference/paper/inference_fid_60m_paper",
    "baseline_afdb_256_step100k": "inference/paper/inference_fid_60m_paper",
    "baseline_afdb_256_step200k": "inference/paper/inference_fid_60m_paper",
    "baseline_afdb_256_step1000k": "inference/paper/inference_fid_60m_paper",
    "baseline_afdb_256_step1300k": "inference/paper/inference_fid_60m_paper",
    "baseline_afdb_256_step1600k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_afdb_256_step100k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_afdb_256_step200k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_afdb_256_step1000k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_afdb_256_step100k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_afdb_256_step200k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_afdb_256_step700k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_afdb_256_step1000k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_afdb_256_step100k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_afdb_256_step100k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_afdb_256_step200k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_afdb_256_step1000k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_afdb_256_step1300k": "inference/paper/inference_fid_60m_paper",
    # ── n=256 convergence sweep extension (added 2026-05-17) ────────────── #
    # PDB additions:
    "baseline_256_bs24_2gpu_step1500k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_residue_bs24_2gpu_step800k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_residue_bs24_2gpu_step900k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_256_per_residue_bs24_2gpu_step800k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_256_per_residue_bs24_2gpu_step900k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_256_per_residue_step100k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_256_per_residue_step200k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_256_per_residue_step400k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_residue_random_bs24_2gpu_step100k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_residue_random_bs24_2gpu_step200k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_residue_random_bs24_2gpu_step400k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_residue_random_bs24_2gpu_step700k": "inference/paper/inference_fid_60m_paper",
    # AFDB additions:
    "repa_l9_afdb_256_step500k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_afdb_256_step600k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_afdb_256_step1200k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l4_afdb_256_step1100k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_afdb_256_step1400k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_afdb_256_step1500k": "inference/paper/inference_fid_60m_paper",
    # ── n=256 convergence extension 2 (added 2026-05-25) ─────────────────── #
    "repa_l4_256_per_residue_bs24_2gpu_step1000k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_256_per_residue_bs24_2gpu_step1000k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_256_per_residue_bs24_2gpu_step1100k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_256_per_residue_step800k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_256_per_residue_step900k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_256_per_residue_step1000k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_256_per_residue_step1100k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_256_per_residue_step1200k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_afdb_256_step700k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_afdb_256_step800k": "inference/paper/inference_fid_60m_paper",
    # ── n=128 PDB convergence extension (added 2026-05-25) ───────────────── #
    "repa_l4_128_bs80_step600k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_random_step500k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_128_random_step600k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l9_128_bs80_step500k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l9_128_bs80_2gpu_step600k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l9_128_bs80_2gpu_step700k": "inference/paper/inference_fid_60m_n128_paper",
    # ── n=256 convergence extension 3 (added 2026-05-25) ─────────────────── #
    "baseline_256_bs24_2gpu_step1600k": "inference/paper/inference_fid_60m_paper",
    "baseline_afdb_256_step1700k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_afdb_256_step900k": "inference/paper/inference_fid_60m_paper",
    # ── n=128 AFDB convergence extension 3 (added 2026-05-25) ────────────── #
    "baseline_afdb_128_bs80_step1100k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_l4_afdb_128_bs80_step500k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l4_afdb_128_bs80_step1100k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l9_afdb_128_bs80_2gpu_step500k": "inference/paper/inference_fid_60m_n128_paper",
    "repa_mpnn_l9_afdb_128_bs80_2gpu_step600k": "inference/paper/inference_fid_60m_n128_paper",
    # ── n=256 convergence extension 4 (added 2026-05-26) ─────────────────── #
    "baseline_256_bs24_2gpu_step1700k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_256_per_residue_bs24_2gpu_step1200k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_256_per_residue_step1300k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_256_per_residue_random_bs24_2gpu_step800k": "inference/paper/inference_fid_60m_paper",
    "baseline_afdb_256_step1800k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_afdb_256_step1300k": "inference/paper/inference_fid_60m_paper",
    # ── n=256 convergence extension 5 (added 2026-05-26) ─────────────────── #
    "baseline_256_bs24_2gpu_step1800k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_256_per_residue_step1400k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_afdb_256_step1000k": "inference/paper/inference_fid_60m_paper",
    "repa_l4_afdb_256_random_step100k": "inference/paper/inference_fid_60m_paper",
    # ── n=256 convergence extension 6 (added 2026-05-28) ─────────────────── #
    "repa_l9_256_per_residue_bs24_2gpu_step1300k": "inference/paper/inference_fid_60m_paper",
    "repa_l9_256_per_residue_bs24_2gpu_step1400k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_256_per_residue_step1500k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_256_per_residue_step1600k": "inference/paper/inference_fid_60m_paper",
    "repa_mpnn_l9_256_per_residue_step1700k": "inference/paper/inference_fid_60m_paper",
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
