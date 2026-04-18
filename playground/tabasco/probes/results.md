# Tabasco Representation Quality Probes

Per-atom atom-type classification (P3) and per-molecule descriptor regression (P4).

## P3 — Atom-type classification

| source | dim | accuracy | macro-F1 | n_train | n_test | classes |
|---|---:|---:|---:|---:|---:|---:|
| chemeleon_frozen | 2048 | 1.0000 | 1.0000 | 20994 | 5249 | 7 |
| mace_frozen | 192 | 0.9971 | 0.8457 | 20994 | 5249 | 7 |
| dummy_frozen | 256 | 0.6961 | 0.1206 | 20994 | 5249 | 7 |
| ckpt:baseline | 128 | 0.9992 | 0.9931 | 20994 | 5249 | 7 |
| ckpt:chemeleon_additive_same | 128 | 0.9998 | 0.9969 | 20994 | 5249 | 7 |
| ckpt:chemeleon_tradeoff_same | 128 | 0.9977 | 0.9771 | 20994 | 5249 | 7 |
| ckpt:chemeleon_additive_fused | 128 | 1.0000 | 1.0000 | 20994 | 5249 | 7 |
| ckpt:chemeleon_tradeoff_fused | 128 | 0.9983 | 0.9905 | 20994 | 5249 | 7 |
| ckpt:mace_additive | 128 | 0.9994 | 0.9908 | 20994 | 5249 | 7 |
| ckpt:mace_tradeoff | 128 | 0.9994 | 0.9855 | 20994 | 5249 | 7 |

## P4 — RDKit descriptor regression (Pearson r / R²)

| source | MolWt  | MolLogP  | NumRings  | NumRotatableBonds  |
|---|---:|---:|---:|---:|
| chemeleon_frozen | +0.997/+0.993 | +0.994/+0.988 | +0.998/+0.995 | +0.995/+0.989 |
| mace_frozen | +0.316/+0.022 | +0.663/+0.307 | +0.361/+0.081 | +0.032/-0.014 |
| dummy_frozen | +0.908/+0.822 | +0.404/+0.160 | +0.661/+0.432 | +0.595/+0.345 |
| ckpt:baseline | +0.959/+0.920 | +0.845/+0.713 | +0.864/+0.742 | +0.799/+0.636 |
| ckpt:chemeleon_additive_same | +0.969/+0.938 | +0.822/+0.672 | +0.867/+0.746 | +0.778/+0.600 |
| ckpt:chemeleon_tradeoff_same | +0.965/+0.930 | +0.755/+0.565 | +0.801/+0.638 | +0.732/+0.532 |
| ckpt:chemeleon_additive_fused | +0.972/+0.943 | +0.824/+0.669 | +0.861/+0.736 | +0.778/+0.606 |
| ckpt:chemeleon_tradeoff_fused | +0.946/+0.894 | +0.803/+0.633 | +0.862/+0.733 | +0.775/+0.597 |
| ckpt:mace_additive | +0.967/+0.935 | +0.840/+0.703 | +0.842/+0.704 | +0.767/+0.582 |
| ckpt:mace_tradeoff | +0.977/+0.955 | +0.854/+0.730 | +0.856/+0.732 | +0.816/+0.666 |
