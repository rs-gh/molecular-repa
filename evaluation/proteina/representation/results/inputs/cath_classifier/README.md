# `inputs/cath_classifier/` — CATH-classifier artifact (shared input)

This is **not a sweep result** — it's an artifact consumed by other code.

## What's in here

| File | Purpose |
|---|---|
| `cath_gearnet_T.pkl` | Pickled bundle: linear logistic-regression head on GearNet embeddings → CATH T-level, plus train metadata / class distribution / sidecar JSON sklearn provenance. |
| `cath_gearnet_T.json` | Metadata sidecar for the pickle (sklearn version, hash, training params). |
| `batch_manifest_cath_train_labelled_v1.json` | Reproducible manifest of train.lmdb proteins (with CATH labels at T-level) used to fit the classifier. |
| `batch_manifest_cath_eval_labelled_v1.json` | Reproducible manifest of val.lmdb proteins used for the classifier's held-out CATH accuracy. |

## Who consumes this

- [evaluation/proteina/generation/scripts/evaluate.py](../../../../generation/scripts/evaluate.py)
  via `compute_cath_metrics` — reads `cath_gearnet_T.pkl` to score
  generated structures by CATH-T fold assignment as part of the generation
  evaluation pipeline.

## How it was built

```bash
sbatch hpc-scripts/proteina/evaluation/representation/build_cath_classifier.sh \
    --n_train 5000 --n_eval 500
```

Driven by
[scripts/paper/build_cath_classifier.py](../../../scripts/paper/build_cath_classifier.py).

Rebuild only when the CATH label vocabulary or the GearNet weights change.
