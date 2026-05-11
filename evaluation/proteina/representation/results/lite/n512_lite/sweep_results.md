# Proteina Probe Sweep — peak-layer summary

Primary metric: linear-head P@L/5 at the best layer for each (run, step, t).

`P@L/5-mlp` column shows the MLP-head number at the same best-layer choice,
to flag cases where nonlinear decodability is inflating estimates.

| run | step | t | best_layer | P@L/5 (linear) | P@L/5 (mlp) | CATH-acc | CATH-classes |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_512_sm | 500000 | 0.50 | 5 | 0.205 | 0.198 | 0.600 | 9 |
| baseline_512_sm | 500000 | 0.75 | 5 | 0.607 | 0.620 | 0.700 | 9 |
| baseline_512_sm | 500000 | 1.00 | 0 | 0.894 | 0.899 | 0.300 | 9 |
| distance_only | 0 | 1.00 | -101 | — | 0.024 | nan | 0 |
| gearnet | 0 | 1.00 | -1 | 0.602 | 0.630 | 1.000 | 9 |
| random_gauss | 0 | 1.00 | -2 | 0.022 | 0.017 | 0.300 | 9 |
| random_rank | 0 | 1.00 | -100 | — | 0.018 | nan | 0 |
| repa_l0_512_sm | 750000 | 0.50 | 5 | 0.239 | 0.262 | 0.900 | 9 |
| repa_l0_512_sm | 750000 | 0.75 | 3 | 0.603 | 0.622 | 1.000 | 9 |
| repa_l0_512_sm | 750000 | 1.00 | 2 | 0.903 | 0.898 | 0.600 | 9 |
| repa_l4_512_sm | 750000 | 0.50 | 9 | 0.195 | 0.203 | 1.000 | 9 |
| repa_l4_512_sm | 750000 | 0.75 | 6 | 0.534 | 0.592 | 0.900 | 9 |
| repa_l4_512_sm | 750000 | 1.00 | 0 | 0.896 | 0.895 | 0.500 | 9 |
| repa_l9_512_sm | 750000 | 0.50 | 3 | 0.231 | 0.201 | 0.700 | 9 |
| repa_l9_512_sm | 750000 | 0.75 | 3 | 0.573 | 0.586 | 0.900 | 9 |
| repa_l9_512_sm | 750000 | 1.00 | 0 | 0.903 | 0.902 | 0.500 | 9 |
| seq_onehot | 0 | 1.00 | -3 | 0.038 | 0.047 | 0.200 | 9 |
