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
| baseline_512_sm | 450000 | 0.50 | 4 | 0.227 | 0.210 | 0.556 | 7 |
| baseline_512_sm | 450000 | 0.75 | 6 | 0.621 | 0.630 | 0.556 | 7 |
| baseline_512_sm | 450000 | 1.00 | 0 | 0.955 | 0.954 | 0.444 | 7 |
| distance_only | 0 | 1.00 | -101 | — | 0.007 | nan | 0 |
| gearnet | 0 | 1.00 | -1 | — | 0.674 | nan | 0 |
| pretrained_dfs_60m | 0 | 1.00 | 0 | — | 0.958 | nan | 0 |
| random_gauss | 0 | 1.00 | -2 | 0.023 | 0.022 | 0.333 | 7 |
| random_rank | 0 | 1.00 | -100 | — | 0.015 | nan | 0 |
| repa_l0_512_sm | 830000 | 0.50 | 6 | 0.243 | 0.245 | 0.889 | 7 |
| repa_l0_512_sm | 830000 | 0.75 | 3 | 0.641 | 0.675 | 1.000 | 7 |
| repa_l0_512_sm | 830000 | 1.00 | 2 | 0.947 | 0.949 | 0.556 | 7 |
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
| repa_l4_512_sm | 840000 | 0.50 | 7 | 0.253 | 0.239 | 0.889 | 7 |
| repa_l4_512_sm | 840000 | 0.75 | 6 | 0.605 | 0.675 | 1.000 | 7 |
| repa_l4_512_sm | 840000 | 1.00 | 0 | 0.934 | 0.943 | 0.556 | 7 |
| repa_l9_512_sm | 840000 | 0.50 | 9 | 0.257 | 0.274 | 1.000 | 7 |
| repa_l9_512_sm | 840000 | 0.75 | 2 | 0.611 | 0.609 | 0.667 | 7 |
| repa_l9_512_sm | 840000 | 1.00 | 0 | 0.949 | 0.946 | 0.444 | 7 |
| seq_onehot | 0 | 1.00 | -3 | 0.051 | 0.057 | 0.222 | 7 |
