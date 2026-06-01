# Session handoff — multi-seed bands for Fig 6.2 (2026-06-01)

Task: re-run the representation evals at **3 seeds** for the checkpoints plotted in
Fig 6.2 (rep quality: IF top-1, dihedral MAE, CATH-A over training) and draw a
**min/max band**, mirroring the generation convergence figures (6.1/6.4). Full plan:
`/home/sr2173/.claude/plans/actually-can-you-kick-piped-pearl.md`. Broader report
state: [session_handoff_2026-05-31.md](session_handoff_2026-05-31.md).

## TL;DR — where we are

All **code/config is committed and pushed** (`origin/main` @ `f92fb3c`). The
**smoke test is queued but blocked by a cluster maintenance reservation** on `intr`
(`ReqNodeNotAvail, Reserved for maintenance`), so nothing has run yet. Next action
once the smoke clears + verifies: **launch the 4 full arrays**, then consolidate +
regenerate the figure.

Seeds: **42** (already in the CSVs) + **1042, 2042** (new).

## What is committed (newest first)

- `f92fb3c` docs: tracker — multi-seed rep sweep + AFDB-random-control priority.
- `3d95655` **eval(representation): make probe sweep seed-aware** — the load-bearing
  fixes (see "smoke caught two bugs" below). `pretrain_probe_sweep.py`,
  `sweep_config.yaml`, `run_pretrained_probe_array.sh`.
- `aba0e5d` docs: report — fig02/fig03 + Ch6 prose + appendix todo (parallel session).
- `77c303b` eval(representation): clean-manifest builder + sweep config + batch
  manifests (parallel session; includes the 4 new-seed train manifests + 4 profiles).

**Build artifacts present on disk** (the 4 new train manifests, n=5000, verified
~2–4 % key overlap with seed-42, i.e. genuinely independent draws):
`results/paper/n256_convergence_cleantrain_pdb/batch_manifest_train_clean_v2_s{1042,2042}.json`
and `results/paper/n256_xclean_afdb_pdb/batch_manifest_train_v2_s{1042,2042}.json`.

## Design (so the numbers mean something)

- **Full re-draw, restricted probes.** Each seed gets a fresh uniform 5,000-chain
  **train** manifest; the **eval set is held fixed at seed 42** (so the band reflects
  train-sample + probe-fit variance against a constant eval set). Restricted to the
  probes each regime feeds the figure: **xclean → inverse_folding,dihedral**;
  **cleantrain → cath**.
- 4 profiles in `sweep_config.yaml`: `paper_n256_if_dih_xclean_s{1042,2042}`,
  `paper_n256_cath_cleantrain_s{1042,2042}`. Each lists the **49** plotted
  `(family,step)` tokens, sets `probe_seed`/`manifest_seed` to the seed and
  `eval_manifest_seed: 42`.

## ⚠️ The smoke caught two bugs (both fixed in `3d95655`)

1. **Eval-manifest seed pinning.** `--manifest_seed` was applied to the *eval*
   manifest too, but `eval_clean_v2`/`eval_v1` are pinned to seed 42 →
   `build_or_load_manifest` raised. Fix: new `--eval_manifest_seed` (default =
   `manifest_seed`); profiles set it to 42.
2. **Resume-skip.** `DoneSet.from_jsonl` read all shards (incl. seed-42 rows) and its
   key excluded `probe_seed` → a new-seed run would skip every checkpoint as "done".
   Fix: `from_jsonl(path, probe_seed=...)` filters to the current seed (legacy rows = 42).
3. (Also) launcher shard tag is now **seed-scoped** (`<run>_s<seed>`) so the two seeds
   can run concurrently without appending to one shard file.

## Jobs

- Smoke (latest): **`29955966`** — `paper_n256_if_dih_xclean_s1042`, `--array=0`
  (1 checkpoint), on `intr`. PENDING (maintenance reservation). Logs:
  `/rds/user/sr2173/hpc-work/proteina/logs/repa-array-29955966_0.{out,err}`.
- (Earlier failed smoke `29948796` exposed bug #1.)

## NEXT STEPS (in order)

### 1. Verify the smoke once it runs
```bash
cd /home/sr2173/git/molecular-repa
sacct -j 29955966 --format=JobID,State,Elapsed,ExitCode
# new-seed rows landed?
grep -c '"probe_seed": 1042' evaluation/proteina/representation/results/paper/n256_xclean_afdb_pdb/pretrained_sweep_results.baseline_256_bs24_2gpu_step100k_s1042.jsonl
# consolidate keeps BOTH seeds for that (run,step):
source .venv/bin/activate
python evaluation/proteina/representation/scripts/paper/pretrain_probe_sweep.py \
  --config paper_n256_if_dih_xclean_s1042 --consolidate_only
python - <<'PY'
import csv
rows=[r for r in csv.DictReader(open("evaluation/proteina/representation/results/paper/n256_xclean_afdb_pdb/pretrained_sweep_results.csv"))
      if r["run"].startswith("baseline_256_bs24_2gpu_step100k") and r["probe_kind"]=="inverse_folding"]
print("distinct probe_seed:", sorted({r["probe_seed"] for r in rows}))  # expect {'42','1042'}
PY
```

### 2. Launch the 4 full arrays (after smoke verifies)
`intr` caps submit at 1 job → **override `--qos` off intr** (use the non-capped GPU
qos, e.g. `gpu1.5`, which the training/non-array jobs use). `%8` throttles concurrency.
Run-name collisions are not a concern (eval is read-only on checkpoints).
```bash
cd /home/sr2173/git/molecular-repa
W=hpc-scripts/proteina/evaluation/representation/run_pretrained_probe_array.sh
for cfg in paper_n256_if_dih_xclean_s1042 paper_n256_if_dih_xclean_s2042 \
           paper_n256_cath_cleantrain_s1042 paper_n256_cath_cleantrain_s2042; do
  sbatch --qos=gpu1.5 --array=0-48%8 "$W" --config "$cfg"
done
```
~196 checkpoint-evals total (49×2 regimes×2 new seeds), ~30–40 min each; ~half a day
to a day wall with the throttle + queue. Watch: `squeue -u sr2173 | grep repa-array`.

### 3. Consolidate + regenerate the figure (after all arrays finish)
```bash
source .venv/bin/activate
python evaluation/proteina/representation/scripts/paper/pretrain_probe_sweep.py --config paper_n256_if_dih_xclean_s1042 --consolidate_only
python evaluation/proteina/representation/scripts/paper/pretrain_probe_sweep.py --config paper_n256_cath_cleantrain_s1042 --consolidate_only
# sanity: every plotted (family,step) should have 3 probe_seed values in each CSV
python docs/masters-report/figures/scripts/fig2_representation.py   # bands now appear
```
Then update the fig02 caption to note "3 seeds; shaded band = min/max"
(`docs/masters-report/figures/fig02_representation.tex`) and recompile
(`report-draft.tex`; upquote stub recipe in the 05-31 handoff; expect 43 pp, 0 undefined).

## Plot status
`fig2_representation.py` already draws the band (`best_layer` returns (mean,min,max)
across `probe_seed`; `plot_traj_panel` does `fill_between`). With only seed-42 data it
degenerates to single lines — so the committed PNG is unchanged until the new seeds land.

## Gotchas / notes
- Manifests are version-pinned and seed-validated; new seeds needed new versions
  (`build_clean_manifests_v2.py --seed N --figure-only`). Don't pass `--manifest_seed`
  against an existing version tag — it raises.
- Maintenance reservation currently blocks `intr` (and likely the partition). Submits
  will queue until it clears.
- `consolidate` dedup key now includes `probe_seed`; re-consolidating existing dirs is
  safe (legacy rows = seed 42).
- This is the FIGURE only; multi-seed error bars on `table_rep_quality`, and a separate
  multi-seed pass for the **AFDB random control**, are tracked in
  `appendix_and_cleanup_todo.md`.
