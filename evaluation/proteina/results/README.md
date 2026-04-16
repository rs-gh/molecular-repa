# Proteina Evaluation Results

## Data pipeline

Raw inference outputs live in `eval_output/` at the repo root. Each subdirectory
(e.g. `inference_fid_60m_baseline/`) contains:

- `samples_fid/` — generated PDB files
- `tensors/` — precomputed atom37 feature tensors
- `results_<name>_fid.csv` — FID and feature-JSD metrics (produced by `evaluate.py`)

The CSVs are copied here for analysis, renamed for brevity:

```
eval_output/inference_fid_60m_baseline/results_inference_fid_60m_baseline_fid.csv
  → evaluation/proteina/results/pdb/fid/inference_fid_60m_baseline.csv

eval_output/inference_fid_60m_repa/results_inference_fid_60m_repa_fid.csv
  → evaluation/proteina/results/pdb/fid/inference_fid_60m_repa.csv

(same pattern for repa_layer0, repa_layer9, etc.)
```

Subdirectories in `eval_output/` with a `_840k` suffix have samples generated but
metrics not yet computed — they will not have corresponding CSVs here until
post-processing is run.
