# Proteina Probe Sweep — peak-layer summary

Primary metric: linear-head P@L/5 at the best layer for each (run, step, t).

`P@L/5-mlp` column shows the MLP-head number at the same best-layer choice,
to flag cases where nonlinear decodability is inflating estimates.

| run | step | t | best_layer | P@L/5 (linear) | P@L/5 (mlp) | CATH-acc | CATH-classes |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 10000 | 1.00 | 0 | — | 0.887 | nan | 0 |
| baseline | 20000 | 1.00 | 1 | — | 0.906 | nan | 0 |
| baseline | 40000 | 1.00 | 3 | — | 0.921 | nan | 0 |
| baseline | 80000 | 1.00 | 3 | — | 0.933 | nan | 0 |
| baseline | 150000 | 1.00 | 0 | — | 0.935 | nan | 0 |
| baseline | 250000 | 1.00 | 0 | — | 0.944 | nan | 0 |
| baseline | 350000 | 1.00 | 0 | — | 0.943 | nan | 0 |
| baseline | 450000 | 1.00 | 0 | — | 0.953 | nan | 0 |
| baseline | 550000 | 1.00 | 0 | — | 0.949 | nan | 0 |
| baseline | 650000 | 1.00 | 0 | — | 0.948 | nan | 0 |
| baseline | 740000 | 1.00 | 0 | — | 0.943 | nan | 0 |
| gearnet | 0 | 1.00 | -1 | — | 0.674 | nan | 0 |
| pretrained_dfs_60m | 0 | 1.00 | 0 | — | 0.958 | nan | 0 |
| repa_l4 | 10000 | 1.00 | 0 | — | 0.873 | nan | 0 |
| repa_l4 | 20000 | 1.00 | 0 | — | 0.890 | nan | 0 |
| repa_l4 | 40000 | 1.00 | 0 | — | 0.905 | nan | 0 |
| repa_l4 | 80000 | 1.00 | 0 | — | 0.925 | nan | 0 |
| repa_l4 | 150000 | 1.00 | 0 | — | 0.934 | nan | 0 |
| repa_l4 | 250000 | 1.00 | 2 | — | 0.951 | nan | 0 |
| repa_l4 | 350000 | 1.00 | 1 | — | 0.937 | nan | 0 |
| repa_l4 | 450000 | 1.00 | 1 | — | 0.945 | nan | 0 |
| repa_l4 | 550000 | 1.00 | 0 | — | 0.938 | nan | 0 |
| repa_l4 | 650000 | 1.00 | 0 | — | 0.945 | nan | 0 |
| repa_l4 | 750000 | 1.00 | 1 | — | 0.948 | nan | 0 |
| repa_l4 | 840000 | 1.00 | 0 | — | 0.943 | nan | 0 |
