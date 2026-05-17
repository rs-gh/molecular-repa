# Gen vs Rep correlations (n=256 convergence)

Rep metrics are reduced across layers per checkpoint (max for accuracies, min for MAE). All rep rows are at t=1.0.

`partial_spearman_*_ctrl_step` is the Spearman partial correlation controlling for training step.


## AFDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 14 | 0.438 | 0.117 | 0.374 | 0.187 | 0.211 | 0.470 |
| cath_C_top1 | _res_AFDB_FID | 14 | 0.377 | 0.183 | 0.263 | 0.364 | 0.145 | 0.622 |
| cath_C_top1 | _res_designability_rate | 14 | 0.056 | 0.848 | 0.000 | 1.000 | 0.389 | 0.169 |
| cath_A_top1 | _res_PDB_FID | 14 | 0.538 | 0.047 | 0.584 | 0.028 | 0.508 | 0.064 |
| cath_A_top1 | _res_AFDB_FID | 14 | 0.538 | 0.047 | 0.524 | 0.054 | 0.481 | 0.082 |
| cath_A_top1 | _res_designability_rate | 14 | 0.281 | 0.331 | 0.388 | 0.170 | 0.740 | 0.002 |
| cath_T_top1 | _res_PDB_FID | 14 | 0.535 | 0.049 | 0.619 | 0.018 | 0.549 | 0.042 |
| cath_T_top1 | _res_AFDB_FID | 14 | 0.526 | 0.053 | 0.546 | 0.043 | 0.513 | 0.061 |
| cath_T_top1 | _res_designability_rate | 14 | 0.220 | 0.450 | 0.246 | 0.397 | 0.628 | 0.016 |
| if_top1_acc | _res_PDB_FID | 14 | 0.804 | 5.28e-04 | 0.895 | 1.58e-05 | 0.890 | 2.00e-05 |
| if_top1_acc | _res_AFDB_FID | 14 | 0.851 | 1.14e-04 | 0.890 | 2.00e-05 | 0.883 | 2.82e-05 |
| if_top1_acc | _res_designability_rate | 14 | 0.468 | 0.092 | 0.438 | 0.117 | 0.607 | 0.021 |
| dih_mae_total_deg | _res_PDB_FID | 14 | -0.813 | 4.11e-04 | -0.938 | 6.86e-07 | -0.933 | 1.13e-06 |
| dih_mae_total_deg | _res_AFDB_FID | 14 | -0.825 | 2.78e-04 | -0.916 | 4.08e-06 | -0.929 | 1.61e-06 |
| dih_mae_total_deg | _res_designability_rate | 14 | -0.350 | 0.220 | -0.205 | 0.483 | -0.464 | 0.095 |

## AFDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 10 | 0.342 | 0.333 | 0.182 | 0.615 | -0.033 | 0.927 |
| cath_C_top1 | _res_AFDB_FID | 10 | 0.288 | 0.420 | 0.031 | 0.931 | -0.172 | 0.635 |
| cath_C_top1 | _res_designability_rate | 10 | -0.231 | 0.521 | -0.126 | 0.729 | 0.439 | 0.204 |
| cath_A_top1 | _res_PDB_FID | 10 | 0.631 | 0.051 | 0.669 | 0.035 | 0.758 | 0.011 |
| cath_A_top1 | _res_AFDB_FID | 10 | 0.692 | 0.027 | 0.717 | 0.020 | 0.776 | 0.008 |
| cath_A_top1 | _res_designability_rate | 10 | 0.310 | 0.383 | 0.366 | 0.298 | 0.790 | 0.007 |
| cath_T_top1 | _res_PDB_FID | 10 | 0.649 | 0.042 | 0.717 | 0.020 | 0.783 | 0.007 |
| cath_T_top1 | _res_AFDB_FID | 10 | 0.676 | 0.032 | 0.705 | 0.023 | 0.737 | 0.015 |
| cath_T_top1 | _res_designability_rate | 10 | 0.155 | 0.669 | 0.277 | 0.438 | 0.717 | 0.020 |
| if_top1_acc | _res_PDB_FID | 10 | 0.782 | 0.008 | 0.867 | 0.001 | 0.920 | 1.59e-04 |
| if_top1_acc | _res_AFDB_FID | 10 | 0.859 | 0.001 | 0.903 | 3.44e-04 | 0.930 | 9.53e-05 |
| if_top1_acc | _res_designability_rate | 10 | 0.291 | 0.415 | 0.195 | 0.590 | 0.700 | 0.024 |
| dih_mae_total_deg | _res_PDB_FID | 10 | -0.810 | 0.004 | -0.964 | 7.32e-06 | -0.972 | 2.55e-06 |
| dih_mae_total_deg | _res_AFDB_FID | 10 | -0.856 | 0.002 | -0.976 | 1.47e-06 | -0.974 | 2.03e-06 |
| dih_mae_total_deg | _res_designability_rate | 10 | -0.082 | 0.822 | 0.079 | 0.828 | -0.510 | 0.132 |

## PDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 19 | -0.423 | 0.071 | -0.356 | 0.134 | -0.396 | 0.093 |
| cath_C_top1 | _res_AFDB_FID | 19 | -0.391 | 0.098 | -0.329 | 0.169 | -0.367 | 0.122 |
| cath_C_top1 | _res_designability_rate | 19 | 0.091 | 0.712 | -0.078 | 0.752 | -0.086 | 0.727 |
| cath_A_top1 | _res_PDB_FID | 19 | -0.271 | 0.262 | -0.316 | 0.188 | -0.405 | 0.086 |
| cath_A_top1 | _res_AFDB_FID | 19 | -0.221 | 0.362 | -0.287 | 0.233 | -0.374 | 0.115 |
| cath_A_top1 | _res_designability_rate | 19 | 0.104 | 0.673 | -0.060 | 0.808 | -0.084 | 0.734 |
| cath_T_top1 | _res_PDB_FID | 19 | -0.251 | 0.300 | -0.294 | 0.222 | -0.409 | 0.082 |
| cath_T_top1 | _res_AFDB_FID | 19 | -0.202 | 0.408 | -0.267 | 0.270 | -0.378 | 0.110 |
| cath_T_top1 | _res_designability_rate | 19 | 0.080 | 0.745 | -0.054 | 0.827 | -0.085 | 0.728 |
| if_top1_acc | _res_PDB_FID | 19 | -0.222 | 0.360 | -0.182 | 0.455 | -0.220 | 0.365 |
| if_top1_acc | _res_AFDB_FID | 19 | -0.184 | 0.451 | -0.135 | 0.581 | -0.170 | 0.487 |
| if_top1_acc | _res_designability_rate | 19 | 0.038 | 0.879 | -0.020 | 0.935 | -0.030 | 0.904 |
| dih_mae_total_deg | _res_PDB_FID | 19 | 0.167 | 0.496 | 0.165 | 0.500 | 0.073 | 0.765 |
| dih_mae_total_deg | _res_AFDB_FID | 19 | 0.160 | 0.513 | 0.123 | 0.616 | 0.027 | 0.912 |
| dih_mae_total_deg | _res_designability_rate | 19 | -0.024 | 0.922 | -0.144 | 0.556 | -0.190 | 0.436 |

## PDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 16 | -0.566 | 0.022 | -0.478 | 0.061 | -0.435 | 0.093 |
| cath_C_top1 | _res_AFDB_FID | 16 | -0.540 | 0.031 | -0.455 | 0.077 | -0.409 | 0.115 |
| cath_C_top1 | _res_designability_rate | 16 | -0.340 | 0.198 | -0.447 | 0.083 | -0.624 | 0.010 |
| cath_A_top1 | _res_PDB_FID | 16 | -0.412 | 0.113 | -0.446 | 0.084 | -0.428 | 0.098 |
| cath_A_top1 | _res_AFDB_FID | 16 | -0.366 | 0.163 | -0.413 | 0.112 | -0.394 | 0.131 |
| cath_A_top1 | _res_designability_rate | 16 | -0.346 | 0.189 | -0.417 | 0.108 | -0.525 | 0.037 |
| cath_T_top1 | _res_PDB_FID | 16 | -0.384 | 0.142 | -0.438 | 0.090 | -0.431 | 0.096 |
| cath_T_top1 | _res_AFDB_FID | 16 | -0.339 | 0.199 | -0.406 | 0.119 | -0.397 | 0.128 |
| cath_T_top1 | _res_designability_rate | 16 | -0.375 | 0.152 | -0.451 | 0.080 | -0.543 | 0.030 |
| if_top1_acc | _res_PDB_FID | 16 | -0.275 | 0.303 | -0.247 | 0.356 | -0.254 | 0.342 |
| if_top1_acc | _res_AFDB_FID | 16 | -0.241 | 0.368 | -0.206 | 0.444 | -0.211 | 0.433 |
| if_top1_acc | _res_designability_rate | 16 | -0.188 | 0.485 | -0.165 | 0.541 | -0.190 | 0.482 |
| dih_mae_total_deg | _res_PDB_FID | 16 | 0.151 | 0.576 | 0.200 | 0.458 | 0.088 | 0.745 |
| dih_mae_total_deg | _res_AFDB_FID | 16 | 0.145 | 0.593 | 0.162 | 0.549 | 0.046 | 0.867 |
| dih_mae_total_deg | _res_designability_rate | 16 | -0.079 | 0.772 | -0.233 | 0.385 | -0.086 | 0.752 |
