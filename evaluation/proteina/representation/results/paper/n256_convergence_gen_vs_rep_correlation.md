# Gen vs Rep correlations (n=256 convergence)

Rep metrics are reduced across layers per checkpoint (max for accuracies, min for MAE). All rep rows are at t=1.0.

`partial_spearman_*_ctrl_step` is the Spearman partial correlation controlling for training step.


## AFDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_T_top1 | _res_PDB_FID | 32 | -0.384 | 0.030 | -0.688 | 1.36e-05 | -0.744 | 1.07e-06 |
| cath_T_top1 | _res_AFDB_FID | 32 | -0.334 | 0.062 | -0.681 | 1.79e-05 | -0.738 | 1.46e-06 |
| cath_T_top1 | _res_designability_rate | 32 | 0.668 | 2.91e-05 | 0.398 | 0.024 | 0.257 | 0.156 |
| if_top1_acc | _res_PDB_FID | 32 | 0.326 | 0.069 | 0.404 | 0.022 | 0.537 | 0.002 |
| if_top1_acc | _res_AFDB_FID | 32 | 0.297 | 0.099 | 0.420 | 0.017 | 0.552 | 0.001 |
| if_top1_acc | _res_designability_rate | 32 | 0.503 | 0.003 | 0.384 | 0.030 | 0.262 | 0.148 |
| dih_mae_total_deg | _res_PDB_FID | 32 | -0.506 | 0.003 | -0.430 | 0.014 | -0.534 | 0.002 |
| dih_mae_total_deg | _res_AFDB_FID | 32 | -0.506 | 0.003 | -0.450 | 0.010 | -0.555 | 9.89e-04 |
| dih_mae_total_deg | _res_designability_rate | 32 | -0.427 | 0.015 | -0.236 | 0.193 | -0.109 | 0.553 |

## AFDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_T_top1 | _res_PDB_FID | 27 | -0.425 | 0.027 | -0.694 | 5.96e-05 | -0.772 | 2.44e-06 |
| cath_T_top1 | _res_AFDB_FID | 27 | -0.440 | 0.022 | -0.702 | 4.41e-05 | -0.760 | 4.30e-06 |
| cath_T_top1 | _res_designability_rate | 27 | 0.307 | 0.119 | 0.314 | 0.111 | 0.274 | 0.167 |
| if_top1_acc | _res_PDB_FID | 27 | 0.416 | 0.031 | 0.501 | 0.008 | 0.533 | 0.004 |
| if_top1_acc | _res_AFDB_FID | 27 | 0.340 | 0.083 | 0.477 | 0.012 | 0.530 | 0.005 |
| if_top1_acc | _res_designability_rate | 27 | 0.262 | 0.188 | 0.209 | 0.294 | 0.154 | 0.442 |
| dih_mae_total_deg | _res_PDB_FID | 27 | -0.706 | 3.92e-05 | -0.557 | 0.003 | -0.563 | 0.002 |
| dih_mae_total_deg | _res_AFDB_FID | 27 | -0.655 | 2.12e-04 | -0.544 | 0.003 | -0.559 | 0.002 |
| dih_mae_total_deg | _res_designability_rate | 27 | 0.016 | 0.935 | 0.006 | 0.975 | 0.046 | 0.819 |

## PDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_T_top1 | _res_PDB_FID | 33 | -0.378 | 0.030 | -0.328 | 0.062 | -0.138 | 0.444 |
| cath_T_top1 | _res_AFDB_FID | 33 | -0.311 | 0.078 | -0.249 | 0.162 | -0.044 | 0.808 |
| cath_T_top1 | _res_designability_rate | 33 | 0.534 | 0.001 | 0.477 | 0.005 | 0.378 | 0.030 |
| if_top1_acc | _res_PDB_FID | 33 | -0.487 | 0.004 | -0.429 | 0.013 | -0.161 | 0.370 |
| if_top1_acc | _res_AFDB_FID | 33 | -0.420 | 0.015 | -0.385 | 0.027 | -0.125 | 0.487 |
| if_top1_acc | _res_designability_rate | 33 | 0.675 | 1.66e-05 | 0.686 | 1.04e-05 | 0.598 | 2.35e-04 |
| dih_mae_total_deg | _res_PDB_FID | 33 | 0.347 | 0.048 | 0.410 | 0.018 | 0.299 | 0.091 |
| dih_mae_total_deg | _res_AFDB_FID | 33 | 0.287 | 0.106 | 0.366 | 0.036 | 0.237 | 0.184 |
| dih_mae_total_deg | _res_designability_rate | 33 | -0.495 | 0.003 | -0.633 | 7.76e-05 | -0.672 | 1.82e-05 |

## PDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_T_top1 | _res_PDB_FID | 27 | -0.285 | 0.150 | -0.190 | 0.342 | -0.136 | 0.498 |
| cath_T_top1 | _res_AFDB_FID | 27 | -0.196 | 0.327 | -0.102 | 0.614 | -0.045 | 0.825 |
| cath_T_top1 | _res_designability_rate | 27 | 0.465 | 0.015 | 0.338 | 0.085 | 0.319 | 0.105 |
| if_top1_acc | _res_PDB_FID | 27 | -0.321 | 0.102 | -0.256 | 0.198 | -0.130 | 0.519 |
| if_top1_acc | _res_AFDB_FID | 27 | -0.220 | 0.271 | -0.214 | 0.283 | -0.106 | 0.599 |
| if_top1_acc | _res_designability_rate | 27 | 0.592 | 0.001 | 0.614 | 6.62e-04 | 0.580 | 0.002 |
| dih_mae_total_deg | _res_PDB_FID | 27 | 0.194 | 0.332 | 0.222 | 0.265 | 0.251 | 0.206 |
| dih_mae_total_deg | _res_AFDB_FID | 27 | 0.112 | 0.580 | 0.173 | 0.387 | 0.184 | 0.359 |
| dih_mae_total_deg | _res_designability_rate | 27 | -0.429 | 0.026 | -0.538 | 0.004 | -0.662 | 1.70e-04 |
