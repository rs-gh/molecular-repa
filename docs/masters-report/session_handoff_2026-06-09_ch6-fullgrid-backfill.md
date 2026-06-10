# Session handoff — Ch6 full-variant-grid backfill (2026-06-09)

## Goal (user directive)
Make the Ch6 generation tables show **all 6 PDB n=256 variants** (incl. the
random-encoder control), not a hand-picked 2. PDB headline variant is always
**MPNN-L9** (what's plotted everywhere). Generate **all** the data we can —
700k (priority), 1M, then 400k — so we can compare and pick the step later
(user's bias: 700k as the mid-training point). Avoid variant-specific claims
masquerading as REPA-wide ones; always want the random control + all PDB
variants where data exists.

The 6 PDB n=256 variants: **baseline, GearNet-L4, GearNet-L9, MPNN-L4,
MPNN-L9, random-L4** (random-init GearNet).

## What was launched (SLURM, 2026-06-09)
**ACCOUNT HISTORY:** First launched on `LIO-CHARM-SL2-GPU` (30279830-834) — blocked
`AssocGrpGRESMinutes` (budget exhausted). Resubmitted on `computerlab-sl2-gpu`
(30338289-30338301) — blocked `AssocMaxJobsLimit` (account-wide job cap, other
users saturating the pool). Finally resubmitted on **`computerlab-sl3-gpu` / QOS
`gpu2`** (current, below) — pending reason `Priority` (healthy, just queued).
SL3 = free tier, no budget wall, lower priority (gpu2 priority 1000 vs gpu1 5000),
12h wall (our tasks ~2.5h). The script hardcodes `#SBATCH -A LIO-CHARM-SL2-GPU`;
override on the CLI with `-A computerlab-sl3-gpu --qos=gpu2`.

**CURRENT live jobs (SL3):** **30338698, 30338699, 30338700, 30338701, 30338703**
on `computerlab-sl3-gpu`, one array per sampler tag, 4-cell array each:

| Job | tag | flags |
|-----|-----|-------|
| 30338698 | ode      | `--sampling_mode vf` |
| 30338699 | sde_n0.0 | `--sampling_mode sc --sc_scale_noise 0.0` |
| 30338700 | sde_n0.35| `--sampling_mode sc --sc_scale_noise 0.35` |
| 30338701 | sde_n0.5 | `--sampling_mode sc --sc_scale_noise 0.5` |
| 30338703 | sde_n1.0 | `--sampling_mode sc --sc_scale_noise 1.0` |

Profile: **`n256_pdb_sampler_ablation_fullgrid`** (in
`evaluation/proteina/generation/sweep_config.yaml`). 4 (run,step) cells:
- `repa_mpnn_l9_256_per_residue_step700k`  ← headline, the priority cell
- `repa_l9_256_per_residue_bs24_2gpu_step700k`
- `repa_l9_256_per_residue_bs24_2gpu_step400k`
- `repa_l4_256_per_residue_random_bs24_2gpu_step1000k`

Output dir (same as existing sampler ablation, dedup merges):
`results/variance/n256_sampler_ablation/sweep_results.jsonl`.

**Effective new work = 14 gap cells** (the rest dedup-skip):
- 700k: MPNN-L9 ×5 tags (ode,0,0.35,0.5,1.0) + GN-L9 ×2 (0.35,0.5)
- 1M:   random-L4 ×5 (ode,0,0.35,0.5,1.0)  [sde_n0.45 already present, 1 seed]
- 400k: GN-L9 ×2 (0.35,0.5)

~1.5–2.5h/task on A100. `gamma=0.45` (sde_n0.45) is the default and already
present at all these cells from the convergence sweeps (3 seeds), so it is NOT
re-run. Non-0.45 tags are single-seed (rep_idx=0, seed 42) — same standard as
the existing sampler/ODE tables.

## Infra committed this session (commit on main, 2026-06-09)
- `evaluation/proteina/lib/checkpoints.py`: added
  `repa_mpnn_l9_256_per_residue_step700k` to BOTH `RUN_SCHEDULES` and
  `GEN_RUN_CONFIGS`. (The 700k EMA ckpt existed on disk; the registry skipped
  400k→800k for this variant. The dry-run caught the missing GEN_RUN_CONFIGS
  entry — both are needed.)
- `sweep_config.yaml`: the `n256_pdb_sampler_ablation_fullgrid` profile +
  per-tag submission recipe in the comment above it.

## How to launch / re-launch (gate bypass)
The sbatch smoke-gate races the live jsonl appends (see memory
`feedback_smoketest_gate_jsonl_race`). Bypass recipe used:
```
git update-index --assume-unchanged \
  evaluation/proteina/generation/results/variance/n256_sampler_ablation/sweep_results.jsonl
git update-index --assume-unchanged \
  evaluation/proteina/generation/results/variance/n256_sampler_ablation/sweep_results.clean.jsonl
export SKIP_SMOKE_GATE=1
SH=hpc-scripts/proteina/evaluation/generation/run_sweep.sh
sbatch --array=0-3 $SH --config n256_pdb_sampler_ablation_fullgrid --sampling_mode vf
sbatch --array=0-3 $SH --config n256_pdb_sampler_ablation_fullgrid --sampling_mode sc --sc_scale_noise 0.0
sbatch --array=0-3 $SH --config n256_pdb_sampler_ablation_fullgrid --sampling_mode sc --sc_scale_noise 0.35
sbatch --array=0-3 $SH --config n256_pdb_sampler_ablation_fullgrid --sampling_mode sc --sc_scale_noise 0.5
sbatch --array=0-3 $SH --config n256_pdb_sampler_ablation_fullgrid --sampling_mode sc --sc_scale_noise 1.0
# then restore tracking:
git update-index --no-assume-unchanged <both jsonl paths>
```
Dry-run first (no GPU): `python evaluation/proteina/generation/scripts/run_sweep.py --config n256_pdb_sampler_ablation_fullgrid --dry_run` — should show 4 tasks all `[OK]`.

## TODO when jobs land (the actual table work)
1. **Re-clean the jsonl**: `python evaluation/proteina/generation/scripts/clean_variance_jsonl.py`
   on `results/variance/n256_sampler_ablation/` (memory: plots/tables read
   `.clean.jsonl`; stale clean silently shows single-seed). Then `jsonl_to_tsv.py`
   if paper TSVs are involved.
2. **Verify coverage closed**: re-run the coverage audit (script logic in this
   session's transcript) — confirm all 6 variants have all 6 sampler tags at
   700k and 1M (random-L4 only to 1M; it trained to 1600k so 1M is fine).
3. **Beta-stratified pwTM for random-L4** (concentration table 6.8): random-L4
   has NO beta data. Its generated PDBs exist in eval_output at 700k
   (`..._random_bs24_2gpu_step700k_..._sde_n0.45_...`). Add random-L4 to the
   CASES list in `scripts/paper/exp_beta_stratified_diversity.py` and re-run
   (post-processing, no GPU) → writes
   `results/variance/beta_stratified_diversity.json`. Then the concentration
   table can show random.
4. **Populate the tables** (`docs/masters-report/tables/`), standardizing on a
   step but generating both 700k and 1M so we can compare:
   - `table_sampler.tex` (6.6): all 6 variants, MPNN-L9 as headline. Currently
     shows only MPNN-L4 + GN-L4 at 700k (data-availability artifact we're fixing).
   - `table_ode.tex` (6.7): add the new ODE cells for all variants. Currently
     has a daggered MPNN-L9@1M row; with 700k ODE for MPNN-L9 landing, that
     dagger can go and everything sits at one step.
   - `table_designable_diversity.tex` + `table_concentration.tex` (6.8): add
     random where data now exists.
   - **Metric note (load-bearing)**: for designable-subset diversity use **pwTM**
     (n-independent), NOT #clusters (#clusters scales with the designable count
     and gives the wrong answer — confirmed via
     `exp_wholeset_vs_designable_diversity.py` docstring). pwTM data lives in
     `results/variance/wholeset_vs_designable_diversity.json`.

## Context on the Ch6 prose (already committed this session)
§6.2 was restructured: §6.2.3 headline acceleration → §6.2.4 "REPA's gains are
robust to sampler noise" (sampler + ODE) → §6.2.5 "REPA consistently dominates
baseline on whole-set, but not designable-subset metrics" (incl. the
designability–diversity callback) → "What we do not claim". The designability–
diversity trade-off was corrected to the literature definition (designability
number vs designable-subset diversity); "designability–fidelity" was demoted
from a characteristic trade-off to an empirical frontier. Report compiles clean
at ~14.7k words.

## Monitoring
`squeue -u sr2173` — jobs 30279830–834. Logs:
`/rds/user/sr2173/hpc-work/proteina/logs/gen-sweep-<jobid>_<arrayidx>.{out,err}`.
