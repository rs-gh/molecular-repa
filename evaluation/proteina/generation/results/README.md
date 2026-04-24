# Proteina generation-quality results

## Data pipeline

Raw inference outputs live in `eval_output/` at the repo root. Each subdirectory
(e.g. `inference_fid_60m_baseline/`) contains:

- `samples_fid/` — generated PDB files
- `tensors/` — precomputed atom37 feature tensors
- `results_<name>_fid.csv` — FID and feature-JSD metrics (produced by `evaluate.py`)

The CSVs are copied here for analysis, renamed for brevity:

```
eval_output/inference_fid_60m_baseline/results_inference_fid_60m_baseline_fid.csv
  → evaluation/proteina/generation/results/pdb/fid/inference_fid_60m_baseline.csv

eval_output/inference_fid_60m_repa/results_inference_fid_60m_repa_fid.csv
  → evaluation/proteina/generation/results/pdb/fid/inference_fid_60m_repa.csv

(same pattern for repa_layer0, repa_layer9, etc.)
```

Subdirectories in `eval_output/` with a `_840k` suffix have samples generated but
metrics not yet computed — they will not have corresponding CSVs here until
post-processing is run.

## Designability

Designability (ProteinMPNN 8 seqs → ESMFold → scRMSD/TM/pLDDT) is run as part of
the sweep. Default subset is **N=100** PDBs per checkpoint (set in
[sweep_config.yaml](../sweep_config.yaml)); override per-run with
`--designability_subset 500` for headline numbers.

Output columns in `sweep_results.csv` / `results_*_fid.csv`:
`_res_designability_rate` (scRMSD < 2 Å), `_res_scRMSD_mean`/`_median`,
`_res_designability_n`, plus pLDDT and TM-score variants.

### Smoke test first (always)
```
sbatch hpc-scripts/proteina/smoke_tests/smoke_designability.sh
```
Expect `SMOKE PASS: designability pipeline OK`. Hard-fails on NaN (ProteinMPNN
or ESMFold silently broken).

### Backfilling existing sweep outputs
For the 12 sample-matched checkpoints already generated in `eval_output/`
(`inference_inference_fid_60m_*_lite_sweep_*_step_*`), designability columns
can be merged into the existing `results_*_fid.csv` without re-running
generation or FID:

```
bash hpc-scripts/proteina/evaluation/generation/eval_designability_only.sh --list
sbatch --array=0-11 hpc-scripts/proteina/evaluation/generation/eval_designability_only.sh
# override size for a single task:
DESIG_N=500 sbatch --array=5-5 hpc-scripts/proteina/evaluation/generation/eval_designability_only.sh
```

After backfill, regenerate the sweep CSV/MD summaries:
```
python evaluation/proteina/generation/scripts/run_sweep.py --config n128 --consolidate_only
```
(and similarly for n256, n512_sm).

### Dependencies (already staged on RDS)
- ESMFold: HuggingFace `facebook/esmfold_v1`, cached at
  `/rds/user/sr2173/hpc-work/proteina/hf_cache/` (~3 GB).
  `HF_HUB_DISABLE_XET=1` + `TMPDIR` on RDS required — xet ENOSPC otherwise.
- ProteinMPNN CA weights (`v_48_002/010/020.pt`) auto-downloaded to
  `/rds/user/sr2173/hpc-work/proteina/ProteinMPNN/ca_model_weights/` on first run.
