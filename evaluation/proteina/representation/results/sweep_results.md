# Proteina Probe Sweep — peak-layer summary

Primary metric: linear-head P@L/5 at the best layer for each (run, step, t).

`P@L/5-mlp` column shows the MLP-head number at the same best-layer choice,
to flag cases where nonlinear decodability is inflating estimates.

| run | step | t | best_layer | P@L/5 (linear) | P@L/5 (mlp) | CATH-acc | CATH-classes |
|---|---:|---:|---:|---:|---:|---:|---:|
| distance_only | 0 | 1.00 | -101 | — | 0.024 | nan | 0 |
| esm_repa_l0_128 | 87500 | 0.50 | 4 | 0.201 | 0.198 | 0.800 | 9 |
| esm_repa_l0_128 | 87500 | 0.75 | 3 | 0.530 | 0.525 | 0.800 | 9 |
| esm_repa_l0_128 | 87500 | 1.00 | 2 | 0.864 | 0.875 | 0.600 | 9 |
| esm_repa_l4_128 | 248500 | 0.50 | 4 | 0.236 | 0.241 | 0.600 | 9 |
| esm_repa_l4_128 | 248500 | 0.75 | 5 | 0.607 | 0.623 | 0.900 | 9 |
| esm_repa_l4_128 | 248500 | 1.00 | 0 | 0.865 | 0.874 | 0.600 | 9 |
| esm_repa_l9_128 | 266000 | 0.50 | 9 | 0.210 | 0.226 | 0.700 | 9 |
| esm_repa_l9_128 | 266000 | 0.75 | 3 | 0.554 | 0.644 | 1.000 | 9 |
| esm_repa_l9_128 | 266000 | 1.00 | 0 | 0.823 | 0.826 | 0.500 | 9 |
| gearnet | 0 | 1.00 | -1 | 0.602 | 0.629 | 1.000 | 9 |
| random_gauss | 0 | 1.00 | -2 | 0.022 | 0.017 | 0.300 | 9 |
| random_rank | 0 | 1.00 | -100 | — | 0.018 | nan | 0 |
| seq_onehot | 0 | 1.00 | -3 | 0.038 | 0.047 | 0.200 | 9 |
