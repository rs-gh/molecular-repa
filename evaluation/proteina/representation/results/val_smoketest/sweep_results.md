# Proteina Probe Sweep — peak-layer summary

Primary metric: linear-head P@L/5 at the best layer for each (run, step, t).

`P@L/5-mlp` column shows the MLP-head number at the same best-layer choice,
to flag cases where nonlinear decodability is inflating estimates.

| run | step | t | best_layer | P@L/5 (linear) | P@L/5 (mlp) | CATH-acc | CATH-classes |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_128 | 800000 | 1.00 | 0 | 0.774 | 0.774 | 0.400 | 5 |
| gearnet | 0 | 1.00 | -1 | 0.538 | 0.570 | 0.800 | 5 |
| random_rank | 0 | 1.00 | -100 | — | 0.029 | nan | 0 |
