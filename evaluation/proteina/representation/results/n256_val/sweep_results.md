# Proteina Probe Sweep — peak-layer summary

Primary metric: linear-head P@L/5 at the best layer for each (run, step, t).

`P@L/5-mlp` column shows the MLP-head number at the same best-layer choice,
to flag cases where nonlinear decodability is inflating estimates.

| run | step | t | best_layer | P@L/5 (linear) | P@L/5 (mlp) | CATH-acc | CATH-classes |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_256 | 400000 | 0.50 | 4 | 0.251 | 0.238 | 0.600 | 9 |
| baseline_256 | 400000 | 0.75 | 3 | 0.605 | 0.610 | 0.900 | 9 |
| baseline_256 | 400000 | 1.00 | 0 | 0.898 | 0.891 | 0.200 | 9 |
| distance_only | 0 | 1.00 | -101 | — | 0.024 | nan | 0 |
| gearnet | 0 | 1.00 | -1 | 0.602 | 0.617 | 1.000 | 9 |
| random_gauss | 0 | 1.00 | -2 | 0.022 | 0.017 | 0.300 | 9 |
| random_rank | 0 | 1.00 | -100 | — | 0.018 | nan | 0 |
| repa_l0_256 | 400000 | 0.50 | 3 | 0.253 | 0.256 | 0.800 | 9 |
| repa_l0_256 | 400000 | 0.75 | 3 | 0.579 | 0.642 | 0.900 | 9 |
| repa_l0_256 | 400000 | 1.00 | 1 | 0.867 | 0.870 | 0.700 | 9 |
| repa_l4_256 | 400000 | 0.50 | 8 | 0.263 | 0.281 | 1.000 | 9 |
| repa_l4_256 | 400000 | 0.75 | 8 | 0.539 | 0.557 | 1.000 | 9 |
| repa_l4_256 | 400000 | 1.00 | 0 | 0.890 | 0.904 | 0.500 | 9 |
| repa_l9_256 | 400000 | 0.50 | 7 | 0.284 | 0.286 | 1.000 | 9 |
| repa_l9_256 | 400000 | 0.75 | 5 | 0.564 | 0.660 | 1.000 | 9 |
| repa_l9_256 | 400000 | 1.00 | 0 | 0.880 | 0.888 | 0.300 | 9 |
| seq_onehot | 0 | 1.00 | -3 | 0.038 | 0.047 | 0.200 | 9 |
