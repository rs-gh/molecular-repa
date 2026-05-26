# Gen vs Rep correlations (n=256 convergence)

Rep metrics are reduced across layers per checkpoint (max for accuracies, min for MAE). All rep rows are at t=1.0.

`partial_spearman_*_ctrl_step` is the Spearman partial correlation controlling for training step.

Variants: `cath_if_dih` = original (leaky) probe sweep; `cleantrain` = probe-side cleaned PDB val (high-n); `xclean` = doubly-clean cross-DB val.


## cath_if_dih / AFDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 86 | -0.568 | 1.17e-08 | -0.674 | 1.16e-12 | -0.728 | 1.96e-15 |
| cath_C_top1 | _res_AFDB_FID | 86 | -0.509 | 5.61e-07 | -0.655 | 8.02e-12 | -0.708 | 2.41e-14 |
| cath_C_top1 | _res_designability_rate | 86 | 0.521 | 2.65e-07 | 0.330 | 0.002 | 0.269 | 0.012 |
| cath_A_top1 | _res_PDB_FID | 86 | -0.485 | 2.26e-06 | -0.652 | 1.05e-11 | -0.732 | 1.21e-15 |
| cath_A_top1 | _res_AFDB_FID | 86 | -0.428 | 3.87e-05 | -0.639 | 3.47e-11 | -0.719 | 6.45e-15 |
| cath_A_top1 | _res_designability_rate | 86 | 0.564 | 1.57e-08 | 0.348 | 0.001 | 0.287 | 0.007 |
| cath_T_top1 | _res_PDB_FID | 86 | -0.411 | 8.37e-05 | -0.672 | 1.45e-12 | -0.755 | 4.31e-17 |
| cath_T_top1 | _res_AFDB_FID | 86 | -0.341 | 0.001 | -0.659 | 5.05e-12 | -0.743 | 2.59e-16 |
| cath_T_top1 | _res_designability_rate | 86 | 0.622 | 1.58e-10 | 0.371 | 4.37e-04 | 0.315 | 0.003 |
| if_top1_acc | _res_PDB_FID | 86 | 0.300 | 0.005 | 0.433 | 3.10e-05 | 0.520 | 2.91e-07 |
| if_top1_acc | _res_AFDB_FID | 86 | 0.277 | 0.010 | 0.461 | 7.95e-06 | 0.549 | 4.53e-08 |
| if_top1_acc | _res_designability_rate | 86 | 0.375 | 3.73e-04 | 0.145 | 0.184 | 0.063 | 0.562 |
| dih_mae_total_deg | _res_PDB_FID | 86 | -0.471 | 4.64e-06 | -0.436 | 2.65e-05 | -0.500 | 9.75e-07 |
| dih_mae_total_deg | _res_AFDB_FID | 86 | -0.487 | 1.97e-06 | -0.464 | 6.82e-06 | -0.527 | 1.81e-07 |
| dih_mae_total_deg | _res_designability_rate | 86 | -0.414 | 7.57e-05 | -0.078 | 0.473 | -0.005 | 0.961 |

## cath_if_dih / AFDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 75 | -0.571 | 9.10e-08 | -0.644 | 4.71e-10 | -0.716 | 4.99e-13 |
| cath_C_top1 | _res_AFDB_FID | 75 | -0.564 | 1.39e-07 | -0.653 | 2.18e-10 | -0.709 | 1.08e-12 |
| cath_C_top1 | _res_designability_rate | 75 | 0.210 | 0.070 | 0.237 | 0.040 | 0.258 | 0.025 |
| cath_A_top1 | _res_PDB_FID | 75 | -0.473 | 1.83e-05 | -0.636 | 8.77e-10 | -0.731 | 9.15e-14 |
| cath_A_top1 | _res_AFDB_FID | 75 | -0.480 | 1.29e-05 | -0.646 | 4.03e-10 | -0.722 | 2.80e-13 |
| cath_A_top1 | _res_designability_rate | 75 | 0.236 | 0.041 | 0.226 | 0.051 | 0.254 | 0.028 |
| cath_T_top1 | _res_PDB_FID | 75 | -0.452 | 4.78e-05 | -0.655 | 1.90e-10 | -0.755 | 5.09e-15 |
| cath_T_top1 | _res_AFDB_FID | 75 | -0.452 | 4.63e-05 | -0.663 | 9.45e-11 | -0.743 | 2.30e-14 |
| cath_T_top1 | _res_designability_rate | 75 | 0.262 | 0.023 | 0.270 | 0.019 | 0.303 | 0.008 |
| if_top1_acc | _res_PDB_FID | 75 | 0.391 | 5.20e-04 | 0.553 | 2.69e-07 | 0.559 | 1.87e-07 |
| if_top1_acc | _res_AFDB_FID | 75 | 0.309 | 0.007 | 0.535 | 7.43e-07 | 0.556 | 2.20e-07 |
| if_top1_acc | _res_designability_rate | 75 | 0.004 | 0.975 | -0.099 | 0.399 | -0.096 | 0.410 |
| dih_mae_total_deg | _res_PDB_FID | 75 | -0.669 | 5.21e-11 | -0.561 | 1.65e-07 | -0.555 | 2.35e-07 |
| dih_mae_total_deg | _res_AFDB_FID | 75 | -0.615 | 4.43e-09 | -0.545 | 4.32e-07 | -0.547 | 3.75e-07 |
| dih_mae_total_deg | _res_designability_rate | 75 | 0.153 | 0.189 | 0.198 | 0.089 | 0.198 | 0.089 |

## cath_if_dih / PDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 129 | -0.375 | 1.22e-05 | -0.239 | 0.006 | -0.203 | 0.021 |
| cath_C_top1 | _res_AFDB_FID | 129 | -0.315 | 2.81e-04 | -0.199 | 0.024 | -0.144 | 0.104 |
| cath_C_top1 | _res_designability_rate | 129 | 0.444 | 1.32e-07 | 0.229 | 0.009 | 0.187 | 0.034 |
| cath_A_top1 | _res_PDB_FID | 129 | -0.410 | 1.44e-06 | -0.319 | 2.32e-04 | -0.181 | 0.040 |
| cath_A_top1 | _res_AFDB_FID | 129 | -0.348 | 5.25e-05 | -0.272 | 0.002 | -0.124 | 0.161 |
| cath_A_top1 | _res_designability_rate | 129 | 0.474 | 1.45e-08 | 0.355 | 3.69e-05 | 0.238 | 0.007 |
| cath_T_top1 | _res_PDB_FID | 129 | -0.362 | 2.50e-05 | -0.311 | 3.32e-04 | -0.152 | 0.085 |
| cath_T_top1 | _res_AFDB_FID | 129 | -0.300 | 5.55e-04 | -0.261 | 0.003 | -0.092 | 0.299 |
| cath_T_top1 | _res_designability_rate | 129 | 0.424 | 5.44e-07 | 0.329 | 1.38e-04 | 0.182 | 0.039 |
| if_top1_acc | _res_PDB_FID | 129 | -0.473 | 1.52e-08 | -0.435 | 2.57e-07 | -0.211 | 0.016 |
| if_top1_acc | _res_AFDB_FID | 129 | -0.403 | 2.17e-06 | -0.392 | 4.25e-06 | -0.167 | 0.059 |
| if_top1_acc | _res_designability_rate | 129 | 0.615 | 8.52e-15 | 0.557 | 7.16e-12 | 0.410 | 1.43e-06 |
| dih_mae_total_deg | _res_PDB_FID | 129 | 0.332 | 1.20e-04 | 0.330 | 1.33e-04 | 0.275 | 0.002 |
| dih_mae_total_deg | _res_AFDB_FID | 129 | 0.270 | 0.002 | 0.302 | 5.10e-04 | 0.230 | 0.009 |
| dih_mae_total_deg | _res_designability_rate | 129 | -0.442 | 1.52e-07 | -0.394 | 3.84e-06 | -0.368 | 1.79e-05 |

## cath_if_dih / PDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 111 | -0.280 | 0.003 | -0.178 | 0.061 | -0.206 | 0.030 |
| cath_C_top1 | _res_AFDB_FID | 111 | -0.202 | 0.034 | -0.143 | 0.135 | -0.154 | 0.107 |
| cath_C_top1 | _res_designability_rate | 111 | 0.399 | 1.46e-05 | 0.123 | 0.198 | 0.136 | 0.156 |
| cath_A_top1 | _res_PDB_FID | 111 | -0.268 | 0.004 | -0.170 | 0.074 | -0.145 | 0.129 |
| cath_A_top1 | _res_AFDB_FID | 111 | -0.189 | 0.047 | -0.128 | 0.182 | -0.095 | 0.323 |
| cath_A_top1 | _res_designability_rate | 111 | 0.336 | 3.07e-04 | 0.171 | 0.073 | 0.145 | 0.128 |
| cath_T_top1 | _res_PDB_FID | 111 | -0.234 | 0.014 | -0.181 | 0.057 | -0.134 | 0.162 |
| cath_T_top1 | _res_AFDB_FID | 111 | -0.154 | 0.107 | -0.133 | 0.164 | -0.080 | 0.402 |
| cath_T_top1 | _res_designability_rate | 111 | 0.285 | 0.002 | 0.157 | 0.099 | 0.105 | 0.273 |
| if_top1_acc | _res_PDB_FID | 111 | -0.266 | 0.005 | -0.278 | 0.003 | -0.167 | 0.079 |
| if_top1_acc | _res_AFDB_FID | 111 | -0.161 | 0.092 | -0.235 | 0.013 | -0.129 | 0.177 |
| if_top1_acc | _res_designability_rate | 111 | 0.483 | 7.85e-08 | 0.423 | 3.79e-06 | 0.354 | 1.41e-04 |
| dih_mae_total_deg | _res_PDB_FID | 111 | 0.142 | 0.138 | 0.114 | 0.235 | 0.199 | 0.037 |
| dih_mae_total_deg | _res_AFDB_FID | 111 | 0.054 | 0.573 | 0.093 | 0.333 | 0.155 | 0.104 |
| dih_mae_total_deg | _res_designability_rate | 111 | -0.329 | 4.29e-04 | -0.177 | 0.063 | -0.270 | 0.004 |

## cleantrain / PDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 129 | -0.412 | 1.20e-06 | -0.389 | 5.27e-06 | -0.264 | 0.002 |
| cath_C_top1 | _res_AFDB_FID | 129 | -0.350 | 4.76e-05 | -0.342 | 7.39e-05 | -0.201 | 0.023 |
| cath_C_top1 | _res_designability_rate | 129 | 0.481 | 7.75e-09 | 0.374 | 1.29e-05 | 0.242 | 0.006 |
| cath_A_top1 | _res_PDB_FID | 129 | -0.445 | 1.24e-07 | -0.392 | 4.31e-06 | -0.235 | 0.007 |
| cath_A_top1 | _res_AFDB_FID | 129 | -0.386 | 6.13e-06 | -0.343 | 6.96e-05 | -0.173 | 0.050 |
| cath_A_top1 | _res_designability_rate | 129 | 0.475 | 1.26e-08 | 0.409 | 1.47e-06 | 0.263 | 0.003 |
| cath_T_top1 | _res_PDB_FID | 129 | -0.413 | 1.15e-06 | -0.392 | 4.23e-06 | -0.239 | 0.006 |
| cath_T_top1 | _res_AFDB_FID | 129 | -0.366 | 2.02e-05 | -0.346 | 5.99e-05 | -0.180 | 0.041 |
| cath_T_top1 | _res_designability_rate | 129 | 0.420 | 7.35e-07 | 0.397 | 3.19e-06 | 0.248 | 0.005 |
| if_top1_acc | _res_PDB_FID | 129 | -0.398 | 2.98e-06 | -0.423 | 6.10e-07 | -0.261 | 0.003 |
| if_top1_acc | _res_AFDB_FID | 129 | -0.333 | 1.13e-04 | -0.390 | 5.04e-06 | -0.222 | 0.012 |
| if_top1_acc | _res_designability_rate | 129 | 0.517 | 3.61e-10 | 0.507 | 8.70e-10 | 0.393 | 4.02e-06 |
| dih_mae_total_deg | _res_PDB_FID | 129 | 0.192 | 0.029 | 0.191 | 0.030 | 0.257 | 0.003 |
| dih_mae_total_deg | _res_AFDB_FID | 129 | 0.142 | 0.108 | 0.186 | 0.034 | 0.231 | 0.008 |
| dih_mae_total_deg | _res_designability_rate | 129 | -0.238 | 0.007 | -0.203 | 0.021 | -0.269 | 0.002 |

## cleantrain / PDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 111 | -0.366 | 7.67e-05 | -0.296 | 0.002 | -0.246 | 0.009 |
| cath_C_top1 | _res_AFDB_FID | 111 | -0.278 | 0.003 | -0.246 | 0.009 | -0.187 | 0.049 |
| cath_C_top1 | _res_designability_rate | 111 | 0.443 | 1.12e-06 | 0.237 | 0.012 | 0.173 | 0.069 |
| cath_A_top1 | _res_PDB_FID | 111 | -0.351 | 1.57e-04 | -0.266 | 0.005 | -0.211 | 0.026 |
| cath_A_top1 | _res_AFDB_FID | 111 | -0.270 | 0.004 | -0.217 | 0.022 | -0.154 | 0.107 |
| cath_A_top1 | _res_designability_rate | 111 | 0.353 | 1.42e-04 | 0.245 | 0.009 | 0.185 | 0.052 |
| cath_T_top1 | _res_PDB_FID | 111 | -0.318 | 6.73e-04 | -0.268 | 0.004 | -0.224 | 0.018 |
| cath_T_top1 | _res_AFDB_FID | 111 | -0.254 | 0.007 | -0.223 | 0.019 | -0.170 | 0.074 |
| cath_T_top1 | _res_designability_rate | 111 | 0.285 | 0.002 | 0.219 | 0.021 | 0.163 | 0.088 |
| if_top1_acc | _res_PDB_FID | 111 | -0.269 | 0.004 | -0.295 | 0.002 | -0.229 | 0.016 |
| if_top1_acc | _res_AFDB_FID | 111 | -0.173 | 0.069 | -0.267 | 0.005 | -0.199 | 0.036 |
| if_top1_acc | _res_designability_rate | 111 | 0.403 | 1.17e-05 | 0.375 | 4.96e-05 | 0.330 | 4.06e-04 |
| dih_mae_total_deg | _res_PDB_FID | 111 | 0.147 | 0.124 | 0.155 | 0.103 | 0.240 | 0.011 |
| dih_mae_total_deg | _res_AFDB_FID | 111 | 0.077 | 0.424 | 0.150 | 0.116 | 0.213 | 0.025 |
| dih_mae_total_deg | _res_designability_rate | 111 | -0.204 | 0.032 | -0.136 | 0.155 | -0.210 | 0.027 |

## xclean / AFDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 86 | -0.503 | 8.06e-07 | -0.548 | 4.79e-08 | -0.580 | 4.83e-09 |
| cath_C_top1 | _res_AFDB_FID | 86 | -0.473 | 4.24e-06 | -0.515 | 3.82e-07 | -0.545 | 5.75e-08 |
| cath_C_top1 | _res_designability_rate | 86 | 0.224 | 0.038 | 0.128 | 0.239 | 0.036 | 0.741 |
| cath_A_top1 | _res_PDB_FID | 86 | -0.546 | 5.33e-08 | -0.571 | 9.66e-09 | -0.598 | 1.20e-09 |
| cath_A_top1 | _res_AFDB_FID | 86 | -0.530 | 1.55e-07 | -0.569 | 1.06e-08 | -0.598 | 1.17e-09 |
| cath_A_top1 | _res_designability_rate | 86 | 0.204 | 0.059 | 0.106 | 0.333 | 0.017 | 0.876 |
| cath_T_top1 | _res_PDB_FID | 86 | -0.357 | 7.41e-04 | -0.335 | 0.002 | -0.342 | 0.001 |
| cath_T_top1 | _res_AFDB_FID | 86 | -0.328 | 0.002 | -0.294 | 0.006 | -0.296 | 0.006 |
| cath_T_top1 | _res_designability_rate | 86 | 0.377 | 3.41e-04 | 0.353 | 8.70e-04 | 0.295 | 0.006 |
| if_top1_acc | _res_PDB_FID | 86 | -0.213 | 0.049 | -0.044 | 0.688 | -0.047 | 0.670 |
| if_top1_acc | _res_AFDB_FID | 86 | -0.079 | 0.470 | 0.033 | 0.762 | 0.031 | 0.777 |
| if_top1_acc | _res_designability_rate | 86 | 0.584 | 3.49e-09 | 0.452 | 1.25e-05 | 0.468 | 5.42e-06 |
| dih_mae_total_deg | _res_PDB_FID | 86 | 0.181 | 0.096 | 0.118 | 0.279 | 0.176 | 0.105 |
| dih_mae_total_deg | _res_AFDB_FID | 86 | 0.057 | 0.604 | 0.056 | 0.610 | 0.103 | 0.343 |
| dih_mae_total_deg | _res_designability_rate | 86 | -0.187 | 0.084 | -0.145 | 0.183 | -0.278 | 0.010 |

## xclean / AFDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 75 | -0.395 | 4.56e-04 | -0.483 | 1.15e-05 | -0.565 | 1.27e-07 |
| cath_C_top1 | _res_AFDB_FID | 75 | -0.390 | 5.51e-04 | -0.477 | 1.55e-05 | -0.539 | 6.11e-07 |
| cath_C_top1 | _res_designability_rate | 75 | 0.073 | 0.532 | 0.047 | 0.689 | 0.061 | 0.603 |
| cath_A_top1 | _res_PDB_FID | 75 | -0.460 | 3.30e-05 | -0.515 | 2.33e-06 | -0.566 | 1.19e-07 |
| cath_A_top1 | _res_AFDB_FID | 75 | -0.462 | 3.06e-05 | -0.533 | 8.48e-07 | -0.571 | 8.67e-08 |
| cath_A_top1 | _res_designability_rate | 75 | -0.012 | 0.921 | -0.044 | 0.706 | -0.039 | 0.741 |
| cath_T_top1 | _res_PDB_FID | 75 | -0.295 | 0.010 | -0.225 | 0.052 | -0.272 | 0.018 |
| cath_T_top1 | _res_AFDB_FID | 75 | -0.303 | 0.008 | -0.221 | 0.057 | -0.252 | 0.029 |
| cath_T_top1 | _res_designability_rate | 75 | 0.184 | 0.114 | 0.231 | 0.046 | 0.251 | 0.030 |
| if_top1_acc | _res_PDB_FID | 75 | -0.194 | 0.096 | -0.016 | 0.891 | 0.010 | 0.932 |
| if_top1_acc | _res_AFDB_FID | 75 | -0.091 | 0.437 | 0.031 | 0.789 | 0.048 | 0.683 |
| if_top1_acc | _res_designability_rate | 75 | 0.342 | 0.003 | 0.346 | 0.002 | 0.351 | 0.002 |
| dih_mae_total_deg | _res_PDB_FID | 75 | 0.207 | 0.074 | 0.156 | 0.182 | 0.119 | 0.309 |
| dih_mae_total_deg | _res_AFDB_FID | 75 | 0.084 | 0.472 | 0.090 | 0.441 | 0.069 | 0.558 |
| dih_mae_total_deg | _res_designability_rate | 75 | -0.254 | 0.028 | -0.218 | 0.061 | -0.243 | 0.036 |

## xclean / PDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 129 | -0.308 | 3.78e-04 | -0.162 | 0.066 | -0.137 | 0.121 |
| cath_C_top1 | _res_AFDB_FID | 129 | -0.242 | 0.006 | -0.123 | 0.165 | -0.080 | 0.367 |
| cath_C_top1 | _res_designability_rate | 129 | 0.377 | 1.09e-05 | 0.169 | 0.056 | 0.146 | 0.099 |
| cath_A_top1 | _res_PDB_FID | 129 | -0.488 | 4.29e-09 | -0.519 | 2.98e-10 | -0.235 | 0.007 |
| cath_A_top1 | _res_AFDB_FID | 129 | -0.425 | 5.14e-07 | -0.454 | 6.60e-08 | -0.162 | 0.067 |
| cath_A_top1 | _res_designability_rate | 129 | 0.532 | 8.54e-11 | 0.538 | 4.76e-11 | 0.274 | 0.002 |
| cath_T_top1 | _res_PDB_FID | 129 | -0.509 | 7.41e-10 | -0.471 | 1.70e-08 | -0.170 | 0.054 |
| cath_T_top1 | _res_AFDB_FID | 129 | -0.459 | 4.46e-08 | -0.410 | 1.39e-06 | -0.106 | 0.230 |
| cath_T_top1 | _res_designability_rate | 129 | 0.546 | 2.28e-11 | 0.518 | 3.27e-10 | 0.255 | 0.004 |
| if_top1_acc | _res_PDB_FID | 129 | -0.553 | 1.12e-11 | -0.526 | 1.48e-10 | -0.331 | 1.26e-04 |
| if_top1_acc | _res_AFDB_FID | 129 | -0.484 | 6.17e-09 | -0.484 | 6.00e-09 | -0.280 | 0.001 |
| if_top1_acc | _res_designability_rate | 129 | 0.691 | 1.23e-19 | 0.648 | 1.00e-16 | 0.531 | 9.72e-11 |
| dih_mae_total_deg | _res_PDB_FID | 129 | 0.476 | 1.16e-08 | 0.474 | 1.39e-08 | 0.406 | 1.84e-06 |
| dih_mae_total_deg | _res_AFDB_FID | 129 | 0.421 | 6.99e-07 | 0.448 | 9.98e-08 | 0.360 | 2.72e-05 |
| dih_mae_total_deg | _res_designability_rate | 129 | -0.543 | 2.90e-11 | -0.518 | 3.18e-10 | -0.470 | 1.87e-08 |

## xclean / PDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 111 | -0.220 | 0.021 | -0.130 | 0.175 | -0.145 | 0.130 |
| cath_C_top1 | _res_AFDB_FID | 111 | -0.141 | 0.140 | -0.089 | 0.353 | -0.090 | 0.346 |
| cath_C_top1 | _res_designability_rate | 111 | 0.324 | 5.25e-04 | 0.110 | 0.249 | 0.119 | 0.213 |
| cath_A_top1 | _res_PDB_FID | 111 | -0.430 | 2.53e-06 | -0.405 | 1.04e-05 | -0.230 | 0.015 |
| cath_A_top1 | _res_AFDB_FID | 111 | -0.336 | 3.14e-04 | -0.336 | 3.16e-04 | -0.167 | 0.081 |
| cath_A_top1 | _res_designability_rate | 111 | 0.423 | 3.78e-06 | 0.395 | 1.74e-05 | 0.224 | 0.018 |
| cath_T_top1 | _res_PDB_FID | 111 | -0.289 | 0.002 | -0.273 | 0.004 | -0.117 | 0.223 |
| cath_T_top1 | _res_AFDB_FID | 111 | -0.219 | 0.021 | -0.206 | 0.030 | -0.057 | 0.551 |
| cath_T_top1 | _res_designability_rate | 111 | 0.339 | 2.72e-04 | 0.311 | 8.80e-04 | 0.172 | 0.071 |
| if_top1_acc | _res_PDB_FID | 111 | -0.394 | 1.85e-05 | -0.365 | 8.05e-05 | -0.288 | 0.002 |
| if_top1_acc | _res_AFDB_FID | 111 | -0.289 | 0.002 | -0.325 | 4.91e-04 | -0.245 | 0.010 |
| if_top1_acc | _res_designability_rate | 111 | 0.572 | 5.59e-11 | 0.517 | 6.05e-09 | 0.480 | 9.65e-08 |
| dih_mae_total_deg | _res_PDB_FID | 111 | 0.310 | 9.16e-04 | 0.334 | 3.45e-04 | 0.359 | 1.07e-04 |
| dih_mae_total_deg | _res_AFDB_FID | 111 | 0.233 | 0.014 | 0.307 | 0.001 | 0.314 | 7.96e-04 |
| dih_mae_total_deg | _res_designability_rate | 111 | -0.407 | 9.13e-06 | -0.379 | 4.14e-05 | -0.410 | 7.80e-06 |
