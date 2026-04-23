# Proteina Probe Sweep — peak-layer summary

Primary metric: linear-head P@L/5 at the best layer for each (run, step, t).

`P@L/5-mlp` column shows the MLP-head number at the same best-layer choice,
to flag cases where nonlinear decodability is inflating estimates.

| run | step | t | best_layer | P@L/5 (linear) | P@L/5 (mlp) | CATH-acc | CATH-classes |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_128 | 800000 | 0.50 | 4 | 0.092 | 0.053 | nan | 0 |
| baseline_128 | 800000 | 1.00 | 2 | 0.874 | 0.987 | nan | 0 |
| gearnet | 0 | 1.00 | -1 | 0.510 | 0.621 | nan | 0 |
| random_rank | 0 | 1.00 | -100 | — | 0.037 | nan | 0 |
