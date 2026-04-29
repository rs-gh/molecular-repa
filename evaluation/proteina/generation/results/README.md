# Proteina generation-quality results

## Overview

This directory holds consolidated sweep results for generation quality
(FID, fold score, designability, scRMSD) across all evaluated checkpoints.
The primary outputs are the `sweep_results.jsonl` files in each subdirectory
and the figures under `figures/`.

---

## Sweep profiles

Three named profiles are defined in
[`../sweep_config.yaml`](../sweep_config.yaml), each covering a different
evaluation regime. Profiles are passed as `--config <name>` to the sweep
scripts.

| Profile | Purpose | Runs | Steps per run | Samples matched at |
|---|---|---|---|---|
| `n512_convergence` | Training-progression / convergence curves for n=512 models | `baseline`, `repa_l{0,4,9}` | 11–12 log-spaced (10K→840K) | N/A |
| `n128` | Cross-method comparison at equal training budget, n=128 | `baseline_128`, `repa_l{0,4,9}_128` | 1 (baseline 800K, REPA 400K) | ~19.5M samples |
| `n256` | Cross-method comparison at equal training budget, n=256 | `baseline_256`, `repa_l{0,4,9}_256` | 1 (all 400K) | ~7.0M samples |
| `n512_sm` | Cross-method comparison at equal training budget, n=512 | `baseline_512_sm`, `repa_l{0,4,9}_512_sm` | 1 (baseline 500K, REPA 750K) | ~3.0M samples |

**Sample-matching arithmetic** (from `evaluation/proteina/lib/checkpoints.py`):
- n=128: baseline fixed bs=24 (800K × 24 = 19.2M); REPA bs switched 24→80 at step 220K (220K × 24 + 180K × 80 = 19.68M). Both ~19.5M at their respective eval steps.
- n=256: all runs bs switched 12→24 at step 220K (220K × 12 + 180K × 24 = 6.96M ≈ 7M at step 400K).
- n=512: fixed bs throughout (baseline bs=6 at 500K = 3.0M; REPA bs=4 at 750K = 3.0M).

**Checkpoint versions**: n128 and n256 use the corrected `_per_residue` checkpoints (projector depth=3, per-residue REPA averaging, post-April-2026 audit). The n512_sm runs point to the older `_v2` checkpoints (pre-audit, projector depth=2) — cross-size comparisons between n512_sm and n128/n256 are therefore not fully apples-to-apples on the REPA configuration.

---

## Data flow

```
src/proteina/configs/experiment_config/training/
    Training configs define run_name_, dataset, batch size, REPA settings.

/rds/user/sr2173/hpc-work/proteina/store/<run_dir>/checkpoints/
    Trained EMA checkpoints (chk_epoch=*_step=*-EMA.ckpt).

evaluation/proteina/lib/checkpoints.py  (RUN_SCHEDULES, GEN_RUN_CONFIGS)
    Single source of truth: maps run name → (store dir, is_repa, layer, steps).
    Also imports into the representation sweep.

evaluation/proteina/generation/sweep_config.yaml
    Named profiles: which runs, which steps, seed, designability N.

hpc-scripts/proteina/evaluation/generation/run_sweep.sh
    SLURM array job. Each task:
      1. Resolves checkpoint path via RUN_SCHEDULES.
      2. Calls evaluate.py → generates PDBs, computes FID + fold score.
      3. Appends one JSON line to results/<profile>/sweep_results.jsonl.
    Done-set logic: already-completed (run, step) pairs are skipped on re-run.

eval_output/inference_inference_fid_60m_*_sweep_*_step_*/
    Raw outputs per checkpoint:
      samples_fid/          — generated PDB files (200 per run for n128/n512_sm,
                              240 for n256; set by the inference config)
      tensors/              — atom37 feature tensors used for FID
      results_*_fid.csv     — one row with all _res_* metric columns

results/<profile>/sweep_results.jsonl   ← primary data store
    One JSON line per completed (run, step). Error lines (CUDA failures etc.)
    are written with an "error" key and skipped by all consumers.
    On duplicate (run, step) the last successful entry wins.

results/<profile>/sweep_results.{json,csv,md}
    Consolidated views rebuilt from the JSONL by --consolidate_only.
    json = list form (used by plot_sample_matched.py).
    csv/md = human-readable tables.

figures/fig_grid_sample_matched.png
    Bar chart across all three sample-matched profiles (n128/n256/n512_sm).
    Rows = model size, columns = metric (FID, fold score, designability, scRMSD).
    Built by: python evaluation/proteina/generation/scripts/plot_sample_matched.py
```

---

## Novelty (training-set comparison)

Novelty (max TM-score against training-set centroids; <0.5 ⇒ novel fold) is
auto-computed when `centroid_path` is set in the sweep profile (default in
`_defaults` since 2026-04-27, points at `centroids_pdb.pt`). See
[`centroids/README.md`](centroids/README.md) for centroid-panel construction,
caveats, and rebuild instructions.

Adds columns: `_res_novelty_rate`, `_res_novelty_max_tm_{mean,median}`,
`_res_novelty_n`.

---

## Designability (secondary metric, backfilled separately)

Designability (ProteinMPNN 8 seqs → ESMFold → scRMSD < 2 Å) is expensive and
runs as a separate step after FID generation. The `eval_designability_only.sh`
script re-reads the PDBs already in `samples_fid/` and merges the
`_res_designability_rate`, `_res_scRMSD_*`, `_res_plddt_*` columns into the
existing `results_*_fid.csv` without re-running generation or FID.

The plot script picks these up via `_load_per_run_csv_metrics()`, which reads
the `eval_output/` CSV as a secondary source for any metric missing from the
JSONL.

**Current status** (as of 2026-04-26):
- n128: designability complete (N=100 per run).
- n256: designability complete (N=100 per run).
- n512_sm: designability **not yet run** — shows "pending" in the plot.

**To backfill n512_sm designability:**
```bash
# Dry-run to confirm task list
bash hpc-scripts/proteina/evaluation/generation/eval_designability_only.sh --list

# Submit (tasks 8-11 are the n512_sm runs; check --list output to confirm indices)
sbatch --array=8-11 hpc-scripts/proteina/evaluation/generation/eval_designability_only.sh

# After completion, consolidate + replot
python evaluation/proteina/generation/scripts/run_sweep.py --config n512_sm --consolidate_only
python evaluation/proteina/generation/scripts/plot_sample_matched.py
```

**Wall-time estimates** (A100, N=100):
- n=128: ~30–40 min per task
- n=256: ~70 min per task
- n=512_sm: ~70–90 min per task (estimated; not yet measured)

---

## Common commands

```bash
# --- Running a sweep ---

# Dry-run: print task index table, confirm --array range
python evaluation/proteina/generation/scripts/run_sweep.py --config n128 --dry_run

# Submit sample-matched sweep (4 tasks each)
sbatch --array=0-3 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n128
sbatch --array=0-3 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n256
sbatch --array=0-3 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n512_sm

# Backfill a single failed run within a profile
sbatch --array=1-1 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n128 --runs repa_l0_128

# Override seed
EVAL_SEED=123 sbatch --array=0-3 hpc-scripts/proteina/evaluation/generation/run_sweep.sh --config n128

# --- After a sweep ---

# Rebuild csv/md from JSONL without re-running
python evaluation/proteina/generation/scripts/run_sweep.py --config n128 --consolidate_only

# Regenerate the sample-matched figure
python evaluation/proteina/generation/scripts/plot_sample_matched.py
```

---

## Results directory structure

```
results/
  n128/
    sweep_results.jsonl   ← append-only; one line per completed task
    sweep_results.json    ← list form (rebuilt by --consolidate_only)
    sweep_results.csv     ← tabular form
    sweep_results.md      ← markdown table
  n256/
    sweep_results.{jsonl,json,csv,md}
  n512_sm/
    sweep_results.{jsonl,json,csv,md}
  n512_convergence/
    sweep_results.{jsonl,json,csv,md}
  pdb/                    ← legacy pre-sweep CSVs (seed=5); not used by plot
  README.md               ← this file

figures/
  fig_grid_sample_matched.png
  fig_grid_n512_convergence_*.png
```

The `pdb/` subdirectory contains older per-run CSVs generated before the sweep
framework existed (seed=5, run via `eval_fid.sh` directly). They are not read
by any current script and can be ignored.
