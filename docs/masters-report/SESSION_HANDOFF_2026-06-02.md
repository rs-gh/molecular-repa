# Session handoff — Ch6 (Proteina) claim-scope pass + appendix robustness + AFDB n1000 re-eval

**Date:** 2026-06-02
**Working file:** `docs/masters-report/report-draft.tex` (Ch6 = "REPA on protein backbones"; only two real sections: §6.1 Experimental setup, §6.2 Results → all result subsections are **6.2.x**, NOT 6.x).

---

## 1. Goal of this session
1. Tighten **claim scope** across Ch6 results — the user noticed we slide between "best variant" and "all variants" without signalling which. Fix the ambiguous ones with minimal text; make the scope of every claim explicit.
2. Where a claim could be made more robust, **add supporting data** (appendix tables), build what we can now, and grow the appendix running list.
3. Kick off any **data-collection jobs** needed to firm up the above.

Writing-style constraints (user): **no em dashes joining clauses; short single-clause sentences; short captions.** Say "PDB-trained"/"AFDB-trained" not bare "PDB"/"AFDB".

---

## 2. DONE and in the document (compiles clean, 2 passes)

### Text fixes (claim-scope)
- **0a** §6.1 intro headline: bold kept verbatim (user wants the eye-catching line); added one qualifier sentence after it — AFDB designability is the saturated-baseline exception.
- **0b** Fig 6.1 caption (`figures/fig01_fid_des_convergence.tex`): now "REPA accelerates FID in both regimes, and designability on experimental (PDB) data" + AFDB-des saturation note.
- **A** §6.2.4 "fold concentration… baseline grows out of it" → scoped to **PDB** (AFDB baseline was never concentrated; the general claim was false).
- **B** §6.2.3 rep↔gen correlation → "In our **PDB-trained** models…" (Table 6.9 is PDB-only).
- **C** §6.2.3 routing sentence → split; pointed fold-dist → Table 6.5, designability → Table 6.4; "larger **acceleration** in designability" (pins framing; GN-L4 wins the 700K snapshot but MPNN wins acceleration).
- **D** §6.2.3 trimmed a duplicate "AFDB designability is the exception" to a callback.

### New appendix tables (built from real data, scripted, wired, compiling)
- **P4 — CKNNA matrix.** `tables/scripts/make_cknna_matrix.py` → `table_cknna_matrix.tex` → appendix `\label{app:cknna}`; callback at §6.2.2. Shows every REPA variant lifts CKNNA to all 3 encoders incl. non-targets (e.g. GN-L9 raises ESM2 0.0027→0.040). Source: `evaluation/proteina/alignment/results/cknna_matrix_per_residue.jsonl`.
- **R-a — AFDB rep↔gen correlation.** `tables/scripts/make_genrep_corr_afdb.py` → `table_genrep_corr_afdb.tex` → appendix `\label{app:genrep-afdb}`; callback at §6.2.3. n=32 AFDB ckpts. **Encoder-matched**: IF/dih→Des (partial +0.63/+0.44), CATH-A→FID (+0.46), off-axis weak/neg. Reinforces the routing thesis.

> Note: the 3 new table scripts were reformatted by a linter mid-session (cosmetic only). The `make_*.py` are reproducible; rerun any to regenerate its `.tex`.

### Claim-scope register (for the Ch7 discussion)
Commented `CLAIM SCOPE REGISTER` at the top of §6.2 Results in `report-draft.tex`. Tags each headline claim R1–R11 on **two axes**: variant-axis (A=universal / B=routing / C=ceilustrative) + secondary axis (regime/γ/step). Convention: **bold topic sentence names the variant-axis scope; body names the secondary axis.** R10 vs R11 are a deliberate opposite-scope pair. The Ch7 discussion should cite claims by R-id. See memory `project_claim_scope_register.md`.

---

## 3. BUILT but HELD (do NOT `\input` yet)
- **P3 — AFDB rep-quality table.** `tables/scripts/make_rep_quality_afdb.py` → `table_rep_quality_afdb.tex` (`\label{tab:proteina-rep-afdb}`). **Not wired into the document.**
  - **Finding (real):** on AFDB the **fold routing replicates** (GearNet→CATH), but **per-residue routing does NOT** — IF is flat and dihedral MAE *rises* under every REPA variant (incl. MPNN), the opposite of PDB. Caption already states this honestly.
  - **Why held:** rests on only 2–3 n1000 checkpoints/variant in the 700K–1.2M window (baseline dihedral itself swings 32→39 between two ckpts). Too noisy to commit the per-residue claim. GN-L9 and the random control are absent at n1000.
  - **Unblocked by §4 below.** The script auto-includes new variants/checkpoints as n1000 data lands (window is a `WINDOW=(700,1200)` constant; can widen later).

---

## 4. LIVE JOB — AFDB n1000 probe re-eval (the thing to keep tabs on)

**Why:** AFDB checkpoints exist on /rds to 1.0–1.8M, but the n1000 probe eval is behind (sweep default is `n_train=5000`). 11 AFDB checkpoints have n5000 rows but no n1000. This blocks P3's GN-L9 row and a non-noisy per-residue read.

**What I did:**
- Appended an additive profile to `evaluation/proteina/representation/sweep_config.yaml` (at EOF):
  **`paper_n256_cath_if_dih_xclean_pdb_afdb_n1000`** — `n_train: 1000`, **`train_manifest_version: train_v1`** (the n=1000 manifest; NOT ext4's `train_v2` which is the n=5000 manifest — see smoke note below), `eval_clean_v2`, `n_eval 62`, output `results/paper/n256_xclean_pdb_afdb`. The 11 runs (array index order):
  ```
  0 repa_l9_afdb_256_step700k     <- GN-L9, the critical one (smoke target)
  1 repa_l9_afdb_256_step800k
  2 repa_l9_afdb_256_step900k
  3 repa_l9_afdb_256_step1000k
  4 repa_l4_afdb_256_step1300k
  5 baseline_afdb_256_step1700k
  6 baseline_afdb_256_step1800k
  7 repa_l4_afdb_256_random_step100k
  8 repa_l4_afdb_256_random_step200k
  9 repa_l4_afdb_256_random_step400k
  10 repa_l4_afdb_256_random_step500k
  ```
- **Smoke history (3 tries, each caught a real config bug cheaply on intr):**
  1. `29995670` FAILED 31s — profile had `train_v2` (n=5000 manifest); needs **`train_v1`** (n=1000). Fixed in profile.
  2. `29996890` FAILED 18s — launched `--lmdb_dataset afdb`; the eval/probe set for `xclean_pdb_afdb` is the **PDB cross-database** set, not AFDB. Both train_v1 (1000 keys) and eval_clean_v2 (62 keys) are 100% in the **PDB** LMDB. The `--lmdb_dataset` flag selects the *probe proteins*, NOT the model's training data (the model is still the AFDB checkpoint).
  3. `29996985` COMPLETED exit 0 but wrote **0 n1000 rows** — "all layers cached, skip". **DoneSet keys on `(run, step, layer, t_tag)`, NOT n_train** ([pretrain_probe_sweep.py:1220](../../evaluation/proteina/representation/scripts/paper/pretrain_probe_sweep.py#L1220)). The existing n5000 rows in the main dir made it skip the n1000 fit.
  4. **Fix:** profile `output_dir` changed to **`results/paper/_n1000_compare/n256_xclean_pdb_afdb`** (fresh dir = empty DoneSet), and `batch_manifest_{train_v1,eval_clean_v2}.json` pre-copied in so the protein sets stay byte-identical to existing n1000 rows. **Re-smoke `29997100`** (fresh dir, `--lmdb_dataset pdb`, will re-extract features so slower). Watcher `bfpjft3lo`. ← current.
  **This mirrors the PDB `_n1000_compare/` convention — that dir exists precisely to dodge the n_train-blind DoneSet.**
  5. **Smoke `29997100` PASSED** — fresh-dir shard has 50 real n1000 rows (cath 30 / dih 10 / IF 10), genuine probe fits in the log. Setup is correct.
  6. **Remaining 10 launched: job `29997244`** on `--qos=gpu1 --array=1-10%5`. Note: `intr` has `MaxSubmitPU=1` so it CANNOT run an array — override to `--qos=gpu1` (no submit cap) for the full array; keep `intr` for single-task smokes only. Watcher `biq778r9a` waits for all 10, runs `--consolidate_only`, reports per-checkpoint counts.

**TO FINISH (next session):**
1. Check the smoke result (re-smoke job = **29996890**):
   ```bash
   sacct -j 29996890 --format=JobID,State,ExitCode,Elapsed -n
   # success = State COMPLETED, and a new row appears:
   python3 -c "import csv;p='evaluation/proteina/representation/results/paper/n256_xclean_pdb_afdb/pretrained_sweep_results.csv';print(sum(1 for r in csv.DictReader(open(p)) if r['run']=='repa_l9_afdb_256_step700k' and str(r.get('n_train'))=='1000'))"
   # logs: /rds/user/sr2173/hpc-work/proteina/logs/repa-array-29996890_0.{out,err}
   ```
2. If the smoke is clean, **launch the remaining 10** (note `--lmdb_dataset pdb`, NOT afdb):
   ```bash
   cd /home/sr2173/git/molecular-repa
   sbatch --array=1-10 hpc-scripts/proteina/evaluation/representation/run_pretrained_probe_array.sh \
       --config paper_n256_cath_if_dih_xclean_pdb_afdb_n1000 --lmdb_dataset pdb
   ```
   (DoneSet dedup makes it safe to instead launch `--array=0-10`. Add `%4` to throttle if desired.)
3. When all tasks done, **consolidate shards into the CSV** (per the launcher header):
   ```bash
   python evaluation/proteina/representation/scripts/paper/pretrain_probe_sweep.py \
       --config paper_n256_cath_if_dih_xclean_pdb_afdb_n1000 --consolidate_only
   ```
4. **First** make the two AFDB table scripts read BOTH dirs (the new rows land in `_n1000_compare/n256_xclean_pdb_afdb`, not the main dir). In `make_rep_quality_afdb.py` and `make_genrep_corr_afdb.py`, change `REP_CSV` to read the main CSV **and** `results/paper/_n1000_compare/n256_xclean_pdb_afdb/pretrained_sweep_results.csv` (union the rows; filter n_train=1000). This is exactly what `make_genrep_corr.py` already does for PDB (it lists both the regime dir and `_n1000_compare`). Then regenerate and re-examine the per-residue story:
   ```bash
   python docs/masters-report/tables/scripts/make_rep_quality_afdb.py   # GN-L9 row should now appear
   python docs/masters-report/tables/scripts/make_genrep_corr_afdb.py
   ```
   If, with GN-L9 + more in-window checkpoints, the per-residue dihedral-degradation persists → it's real; wire P3 in with that framing. If it washes out → revise caption, then wire in.
5. Then `\input{tables/table_rep_quality_afdb}` into a new appendix section (mirror `app:genrep-afdb`), add a callback at §6.2.1 (the "report the AFDB results in Appendix" promise, ~l.1398). Recompile (2 passes).

**Gotcha that bit us:** SLURM JobName strips the encoder type. Job 29956271 named `...repa-l9...` is GearNet-L9; 29956267 named `...repa-l9...` is **MPNN**-L9. Always read `run_name_`/`repa.encoder.type` from the job log, not the JobName.

---

## 5. Appendix running list (in `report-draft.tex`, top of `\chapter{Technical details}`)
Promissory "see Appendix" refs in Ch6 that still need a target, + robustness adds. **Data confirmed to exist for all.**
- **P1** full encoder×depth grid (ref §6.1 models)
- **P2** L4 trained-vs-random control (ref §6.1 models)
- **P3** AFDB rep-quality — BUILT-but-HELD (see §3, §4 above)
- **P4** CKNNA matrix — **DONE**
- **P5** training-dynamics (λ, bs, projector depth) + n≤128 scale (ref §6.2.7)
- **R-a** AFDB rep↔gen corr — **DONE**
- **R-b** sampler γ×variant grid — earns the broad §6.2.5 claim (currently scoped to one representative variant). Source: `docs/research/proteina_sampler_regime_audit_2026-05-28.md`.

Suggested next build order: finish P3 (after §4), then R-b, then P5/P1/P2.

---

## 6. Context / caveats
- AFDB training runs **29956267–271** still live (the 4 canonical 2gpu per-residue families + 1 PDB). So AFDB convergence tables are a **moving snapshot**; a later top-up n1000 sweep will catch newer checkpoints (additive, no rework).
- AFDB CATH at n1000 lives only on the cross-DB **blinded** set (no `cleantrain_afdb` sweep). Captions note this.
- Don't quote absolute scores cross-DB; AFDB designability is proxy-inflated (ProteinMPNN→ESMFold shares AF2 lineage). See memories `project_afdb_designability_proxy_alignment.md`, `project_repa_evidence_framing.md`.
- Standard launch hygiene: smoke→intr→full; match `run_name_` vs exp_config before any training submit; `--qos=intr` for short diagnostics.

## 7. Key paths
- Report: `docs/masters-report/report-draft.tex` · tables: `docs/masters-report/tables/` · table scripts: `.../tables/scripts/`
- Rep-eval: `evaluation/proteina/representation/` (`sweep_config.yaml`, `scripts/paper/pretrain_probe_sweep.py`)
- Launcher: `hpc-scripts/proteina/evaluation/representation/run_pretrained_probe_array.sh`
- Rep results CSV: `evaluation/proteina/representation/results/paper/n256_xclean_pdb_afdb/pretrained_sweep_results.csv`
- Gen (AFDB): `evaluation/proteina/generation/results/paper/n256_convergence_afdb/sweep_results.clean.jsonl`
- Memory: `project_claim_scope_register.md` (has the register, jobs note, P3 finding, this work's state)
