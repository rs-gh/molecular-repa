# Gen vs Rep correlations (n=128 convergence)

Rep metrics are reduced across layers per checkpoint (max for accuracies, min for MAE). All rep rows are at t=1.0.

`partial_spearman_*_ctrl_step` is the Spearman partial correlation controlling for training step.

Variants: `cath_if_dih` = original (leaky) probe sweep; `cleantrain` = probe-side cleaned PDB val (high-n); `xclean` = doubly-clean cross-DB val.


## cath_if_dih / AFDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 22 | 0.339 | 0.123 | 0.261 | 0.240 | 0.150 | 0.505 |
| cath_C_top1 | _res_AFDB_FID | 22 | 0.236 | 0.290 | 0.172 | 0.443 | 0.123 | 0.584 |
| cath_C_top1 | _res_designability_rate | 22 | 0.037 | 0.869 | 0.008 | 0.973 | 0.371 | 0.089 |
| cath_A_top1 | _res_PDB_FID | 22 | 0.478 | 0.024 | 0.447 | 0.037 | 0.425 | 0.049 |
| cath_A_top1 | _res_AFDB_FID | 22 | 0.466 | 0.029 | 0.439 | 0.041 | 0.426 | 0.048 |
| cath_A_top1 | _res_designability_rate | 22 | 0.272 | 0.221 | 0.238 | 0.286 | 0.410 | 0.058 |
| cath_T_top1 | _res_PDB_FID | 22 | 0.522 | 0.013 | 0.444 | 0.039 | 0.441 | 0.040 |
| cath_T_top1 | _res_AFDB_FID | 22 | 0.507 | 0.016 | 0.454 | 0.034 | 0.448 | 0.036 |
| cath_T_top1 | _res_designability_rate | 22 | 0.259 | 0.245 | 0.302 | 0.172 | 0.432 | 0.045 |
| if_top1_acc | _res_PDB_FID | 22 | 0.768 | 2.98e-05 | 0.708 | 2.27e-04 | 0.779 | 1.94e-05 |
| if_top1_acc | _res_AFDB_FID | 22 | 0.845 | 7.30e-07 | 0.783 | 1.67e-05 | 0.810 | 4.83e-06 |
| if_top1_acc | _res_designability_rate | 22 | 0.563 | 0.006 | 0.420 | 0.052 | 0.442 | 0.039 |
| dih_mae_total_deg | _res_PDB_FID | 22 | -0.793 | 1.06e-05 | -0.897 | 1.58e-08 | -0.903 | 8.54e-09 |
| dih_mae_total_deg | _res_AFDB_FID | 22 | -0.820 | 3.00e-06 | -0.931 | 3.33e-10 | -0.930 | 4.07e-10 |
| dih_mae_total_deg | _res_designability_rate | 22 | -0.348 | 0.113 | -0.216 | 0.335 | -0.375 | 0.086 |

## cath_if_dih / AFDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 18 | 0.209 | 0.404 | 0.114 | 0.654 | 0.109 | 0.666 |
| cath_C_top1 | _res_AFDB_FID | 18 | 0.139 | 0.581 | 0.036 | 0.887 | 0.031 | 0.903 |
| cath_C_top1 | _res_designability_rate | 18 | -0.015 | 0.952 | 0.121 | 0.631 | 0.210 | 0.403 |
| cath_A_top1 | _res_PDB_FID | 18 | 0.496 | 0.036 | 0.437 | 0.070 | 0.596 | 0.009 |
| cath_A_top1 | _res_AFDB_FID | 18 | 0.556 | 0.017 | 0.491 | 0.039 | 0.574 | 0.013 |
| cath_A_top1 | _res_designability_rate | 18 | 0.374 | 0.126 | 0.314 | 0.204 | 0.162 | 0.520 |
| cath_T_top1 | _res_PDB_FID | 18 | 0.587 | 0.010 | 0.396 | 0.103 | 0.662 | 0.003 |
| cath_T_top1 | _res_AFDB_FID | 18 | 0.637 | 0.004 | 0.466 | 0.051 | 0.623 | 0.006 |
| cath_T_top1 | _res_designability_rate | 18 | 0.386 | 0.114 | 0.441 | 0.067 | 0.184 | 0.465 |
| if_top1_acc | _res_PDB_FID | 18 | 0.739 | 4.56e-04 | 0.548 | 0.019 | 0.744 | 4.00e-04 |
| if_top1_acc | _res_AFDB_FID | 18 | 0.836 | 1.58e-05 | 0.646 | 0.004 | 0.755 | 2.91e-04 |
| if_top1_acc | _res_designability_rate | 18 | 0.505 | 0.032 | 0.331 | 0.180 | 0.148 | 0.557 |
| dih_mae_total_deg | _res_PDB_FID | 18 | -0.779 | 1.41e-04 | -0.872 | 2.48e-06 | -0.903 | 2.79e-07 |
| dih_mae_total_deg | _res_AFDB_FID | 18 | -0.842 | 1.19e-05 | -0.921 | 5.80e-08 | -0.925 | 3.95e-08 |
| dih_mae_total_deg | _res_designability_rate | 18 | -0.234 | 0.351 | -0.079 | 0.756 | -0.196 | 0.437 |

## cath_if_dih / PDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 135 | -0.479 | 4.03e-09 | -0.424 | 3.03e-07 | -0.418 | 4.50e-07 |
| cath_C_top1 | _res_AFDB_FID | 135 | -0.455 | 2.92e-08 | -0.404 | 1.18e-06 | -0.398 | 1.70e-06 |
| cath_C_top1 | _res_designability_rate | 135 | 0.168 | 0.051 | 0.049 | 0.575 | 0.072 | 0.408 |
| cath_A_top1 | _res_PDB_FID | 135 | -0.380 | 5.57e-06 | -0.340 | 5.48e-05 | -0.372 | 8.65e-06 |
| cath_A_top1 | _res_AFDB_FID | 135 | -0.347 | 3.76e-05 | -0.320 | 1.53e-04 | -0.358 | 2.06e-05 |
| cath_A_top1 | _res_designability_rate | 135 | 0.221 | 0.010 | 0.058 | 0.501 | 0.037 | 0.673 |
| cath_T_top1 | _res_PDB_FID | 135 | -0.316 | 1.92e-04 | -0.268 | 0.002 | -0.311 | 2.37e-04 |
| cath_T_top1 | _res_AFDB_FID | 135 | -0.279 | 0.001 | -0.246 | 0.004 | -0.295 | 5.21e-04 |
| cath_T_top1 | _res_designability_rate | 135 | 0.224 | 0.009 | 0.072 | 0.404 | 0.039 | 0.656 |
| if_top1_acc | _res_PDB_FID | 135 | -0.593 | 3.65e-14 | -0.594 | 3.30e-14 | -0.600 | 1.38e-14 |
| if_top1_acc | _res_AFDB_FID | 135 | -0.556 | 2.54e-12 | -0.568 | 6.85e-13 | -0.577 | 2.32e-13 |
| if_top1_acc | _res_designability_rate | 135 | 0.339 | 5.66e-05 | 0.278 | 0.001 | 0.299 | 4.19e-04 |
| dih_mae_total_deg | _res_PDB_FID | 135 | 0.644 | 3.46e-17 | 0.535 | 2.41e-11 | 0.495 | 1.03e-09 |
| dih_mae_total_deg | _res_AFDB_FID | 135 | 0.649 | 1.70e-17 | 0.538 | 1.63e-11 | 0.488 | 1.99e-09 |
| dih_mae_total_deg | _res_designability_rate | 135 | -0.137 | 0.114 | -0.229 | 0.008 | -0.404 | 1.14e-06 |

## cath_if_dih / PDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 115 | -0.637 | 1.92e-14 | -0.551 | 1.74e-10 | -0.497 | 1.65e-08 |
| cath_C_top1 | _res_AFDB_FID | 115 | -0.624 | 9.37e-14 | -0.531 | 1.01e-09 | -0.468 | 1.30e-07 |
| cath_C_top1 | _res_designability_rate | 115 | -0.138 | 0.141 | -0.170 | 0.069 | -0.234 | 0.012 |
| cath_A_top1 | _res_PDB_FID | 115 | -0.550 | 1.92e-10 | -0.495 | 1.89e-08 | -0.483 | 4.55e-08 |
| cath_A_top1 | _res_AFDB_FID | 115 | -0.527 | 1.42e-09 | -0.471 | 1.06e-07 | -0.460 | 2.39e-07 |
| cath_A_top1 | _res_designability_rate | 115 | -0.141 | 0.133 | -0.176 | 0.060 | -0.200 | 0.032 |
| cath_T_top1 | _res_PDB_FID | 115 | -0.478 | 6.39e-08 | -0.417 | 3.48e-06 | -0.415 | 3.97e-06 |
| cath_T_top1 | _res_AFDB_FID | 115 | -0.451 | 4.10e-07 | -0.391 | 1.57e-05 | -0.389 | 1.71e-05 |
| cath_T_top1 | _res_designability_rate | 115 | -0.124 | 0.188 | -0.171 | 0.068 | -0.187 | 0.046 |
| if_top1_acc | _res_PDB_FID | 115 | -0.703 | 2.05e-18 | -0.671 | 2.21e-16 | -0.632 | 3.67e-14 |
| if_top1_acc | _res_AFDB_FID | 115 | -0.677 | 1.04e-16 | -0.645 | 6.87e-15 | -0.599 | 1.53e-12 |
| if_top1_acc | _res_designability_rate | 115 | 0.114 | 0.226 | 0.142 | 0.131 | 0.099 | 0.290 |
| dih_mae_total_deg | _res_PDB_FID | 115 | 0.693 | 9.40e-18 | 0.598 | 1.69e-12 | 0.528 | 1.28e-09 |
| dih_mae_total_deg | _res_AFDB_FID | 115 | 0.702 | 2.19e-18 | 0.608 | 6.00e-13 | 0.516 | 3.46e-09 |
| dih_mae_total_deg | _res_designability_rate | 115 | -0.157 | 0.094 | -0.246 | 0.008 | -0.196 | 0.036 |

## cleantrain / PDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 135 | -0.404 | 1.18e-06 | -0.407 | 9.58e-07 | -0.392 | 2.50e-06 |
| cath_C_top1 | _res_AFDB_FID | 135 | -0.392 | 2.60e-06 | -0.403 | 1.26e-06 | -0.387 | 3.43e-06 |
| cath_C_top1 | _res_designability_rate | 135 | 0.071 | 0.415 | 0.031 | 0.725 | 0.065 | 0.457 |
| cath_A_top1 | _res_PDB_FID | 135 | -0.397 | 1.82e-06 | -0.345 | 4.27e-05 | -0.373 | 8.17e-06 |
| cath_A_top1 | _res_AFDB_FID | 135 | -0.376 | 7.13e-06 | -0.327 | 1.08e-04 | -0.360 | 1.76e-05 |
| cath_A_top1 | _res_designability_rate | 135 | 0.157 | 0.068 | 0.091 | 0.291 | 0.075 | 0.389 |
| cath_T_top1 | _res_PDB_FID | 135 | -0.318 | 1.69e-04 | -0.294 | 5.39e-04 | -0.353 | 2.69e-05 |
| cath_T_top1 | _res_AFDB_FID | 135 | -0.290 | 6.42e-04 | -0.268 | 0.002 | -0.334 | 7.33e-05 |
| cath_T_top1 | _res_designability_rate | 135 | 0.133 | 0.123 | 0.073 | 0.398 | 0.027 | 0.760 |
| if_top1_acc | _res_PDB_FID | 135 | -0.594 | 2.93e-14 | -0.562 | 1.30e-12 | -0.569 | 5.95e-13 |
| if_top1_acc | _res_AFDB_FID | 135 | -0.569 | 6.31e-13 | -0.540 | 1.42e-11 | -0.549 | 5.30e-12 |
| if_top1_acc | _res_designability_rate | 135 | 0.264 | 0.002 | 0.264 | 0.002 | 0.284 | 8.44e-04 |
| dih_mae_total_deg | _res_PDB_FID | 135 | 0.593 | 3.55e-14 | 0.457 | 2.51e-08 | 0.415 | 5.67e-07 |
| dih_mae_total_deg | _res_AFDB_FID | 135 | 0.602 | 1.11e-14 | 0.458 | 2.25e-08 | 0.409 | 8.41e-07 |
| dih_mae_total_deg | _res_designability_rate | 135 | -0.040 | 0.643 | -0.204 | 0.017 | -0.316 | 1.93e-04 |

## cleantrain / PDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 115 | -0.511 | 5.25e-09 | -0.488 | 3.06e-08 | -0.445 | 6.23e-07 |
| cath_C_top1 | _res_AFDB_FID | 115 | -0.505 | 8.70e-09 | -0.480 | 5.55e-08 | -0.433 | 1.33e-06 |
| cath_C_top1 | _res_designability_rate | 115 | -0.101 | 0.282 | -0.094 | 0.316 | -0.137 | 0.145 |
| cath_A_top1 | _res_PDB_FID | 115 | -0.520 | 2.51e-09 | -0.460 | 2.34e-07 | -0.443 | 7.21e-07 |
| cath_A_top1 | _res_AFDB_FID | 115 | -0.508 | 6.88e-09 | -0.439 | 9.14e-07 | -0.421 | 2.75e-06 |
| cath_A_top1 | _res_designability_rate | 115 | -0.145 | 0.122 | -0.142 | 0.130 | -0.167 | 0.075 |
| cath_T_top1 | _res_PDB_FID | 115 | -0.431 | 1.48e-06 | -0.408 | 6.07e-06 | -0.441 | 8.17e-07 |
| cath_T_top1 | _res_AFDB_FID | 115 | -0.413 | 4.51e-06 | -0.377 | 3.22e-05 | -0.415 | 3.96e-06 |
| cath_T_top1 | _res_designability_rate | 115 | -0.184 | 0.049 | -0.164 | 0.080 | -0.163 | 0.082 |
| if_top1_acc | _res_PDB_FID | 115 | -0.661 | 9.33e-16 | -0.609 | 5.44e-13 | -0.590 | 4.05e-12 |
| if_top1_acc | _res_AFDB_FID | 115 | -0.643 | 9.59e-15 | -0.586 | 6.25e-12 | -0.566 | 4.37e-11 |
| if_top1_acc | _res_designability_rate | 115 | 0.148 | 0.114 | 0.182 | 0.052 | 0.158 | 0.092 |
| dih_mae_total_deg | _res_PDB_FID | 115 | 0.624 | 8.93e-14 | 0.505 | 8.72e-09 | 0.433 | 1.31e-06 |
| dih_mae_total_deg | _res_AFDB_FID | 115 | 0.634 | 2.84e-14 | 0.509 | 6.40e-09 | 0.429 | 1.71e-06 |
| dih_mae_total_deg | _res_designability_rate | 115 | -0.167 | 0.075 | -0.281 | 0.002 | -0.244 | 0.009 |

## xclean / PDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 135 | 0.027 | 0.753 | 0.022 | 0.796 | 0.005 | 0.958 |
| cath_C_top1 | _res_AFDB_FID | 135 | 0.046 | 0.593 | 0.026 | 0.767 | 0.006 | 0.948 |
| cath_C_top1 | _res_designability_rate | 135 | 0.178 | 0.039 | 0.048 | 0.581 | 0.029 | 0.738 |
| cath_A_top1 | _res_PDB_FID | 135 | -0.390 | 2.91e-06 | -0.346 | 3.98e-05 | -0.298 | 4.42e-04 |
| cath_A_top1 | _res_AFDB_FID | 135 | -0.371 | 9.17e-06 | -0.321 | 1.45e-04 | -0.264 | 0.002 |
| cath_A_top1 | _res_designability_rate | 135 | -0.190 | 0.028 | -0.230 | 0.007 | -0.167 | 0.053 |
| cath_T_top1 | _res_PDB_FID | 135 | -0.327 | 1.08e-04 | -0.334 | 7.63e-05 | -0.376 | 6.82e-06 |
| cath_T_top1 | _res_AFDB_FID | 135 | -0.291 | 6.16e-04 | -0.316 | 1.88e-04 | -0.365 | 1.36e-05 |
| cath_T_top1 | _res_designability_rate | 135 | 0.070 | 0.421 | -0.041 | 0.636 | -0.077 | 0.375 |
| if_top1_acc | _res_PDB_FID | 135 | -0.507 | 3.64e-10 | -0.521 | 9.44e-11 | -0.557 | 2.38e-12 |
| if_top1_acc | _res_AFDB_FID | 135 | -0.472 | 7.63e-09 | -0.502 | 5.41e-10 | -0.544 | 8.97e-12 |
| if_top1_acc | _res_designability_rate | 135 | 0.477 | 5.14e-09 | 0.420 | 3.97e-07 | 0.414 | 5.80e-07 |
| dih_mae_total_deg | _res_PDB_FID | 135 | 0.491 | 1.50e-09 | 0.389 | 3.17e-06 | 0.369 | 1.09e-05 |
| dih_mae_total_deg | _res_AFDB_FID | 135 | 0.495 | 1.04e-09 | 0.387 | 3.49e-06 | 0.365 | 1.32e-05 |
| dih_mae_total_deg | _res_designability_rate | 135 | -0.162 | 0.060 | -0.329 | 9.79e-05 | -0.384 | 4.28e-06 |

## xclean / PDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 115 | -0.110 | 0.241 | -0.086 | 0.363 | -0.081 | 0.391 |
| cath_C_top1 | _res_AFDB_FID | 115 | -0.098 | 0.296 | -0.075 | 0.426 | -0.069 | 0.461 |
| cath_C_top1 | _res_designability_rate | 115 | -0.113 | 0.230 | -0.130 | 0.165 | -0.136 | 0.146 |
| cath_A_top1 | _res_PDB_FID | 115 | -0.430 | 1.65e-06 | -0.427 | 1.91e-06 | -0.387 | 1.92e-05 |
| cath_A_top1 | _res_AFDB_FID | 115 | -0.412 | 4.81e-06 | -0.398 | 1.07e-05 | -0.352 | 1.13e-04 |
| cath_A_top1 | _res_designability_rate | 115 | -0.133 | 0.158 | -0.140 | 0.135 | -0.178 | 0.057 |
| cath_T_top1 | _res_PDB_FID | 115 | -0.433 | 1.33e-06 | -0.439 | 9.19e-07 | -0.489 | 2.90e-08 |
| cath_T_top1 | _res_AFDB_FID | 115 | -0.402 | 8.32e-06 | -0.417 | 3.50e-06 | -0.475 | 8.17e-08 |
| cath_T_top1 | _res_designability_rate | 115 | -0.147 | 0.117 | -0.209 | 0.025 | -0.203 | 0.030 |
| if_top1_acc | _res_PDB_FID | 115 | -0.659 | 1.19e-15 | -0.655 | 2.12e-15 | -0.621 | 1.27e-13 |
| if_top1_acc | _res_AFDB_FID | 115 | -0.636 | 2.33e-14 | -0.632 | 3.58e-14 | -0.595 | 2.46e-12 |
| if_top1_acc | _res_designability_rate | 115 | 0.214 | 0.022 | 0.235 | 0.012 | 0.203 | 0.029 |
| dih_mae_total_deg | _res_PDB_FID | 115 | 0.524 | 1.81e-09 | 0.393 | 1.37e-05 | 0.313 | 6.46e-04 |
| dih_mae_total_deg | _res_AFDB_FID | 115 | 0.534 | 8.03e-10 | 0.392 | 1.50e-05 | 0.302 | 0.001 |
| dih_mae_total_deg | _res_designability_rate | 115 | -0.088 | 0.348 | -0.271 | 0.003 | -0.235 | 0.011 |
