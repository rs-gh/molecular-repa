# Gen vs Rep correlations (n=128 convergence)

Rep metrics are reduced across layers per checkpoint (max for accuracies, min for MAE). All rep rows are at t=1.0.

`partial_spearman_*_ctrl_step` is the Spearman partial correlation controlling for training step.


## AFDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 16 | 0.370 | 0.159 | 0.333 | 0.207 | 0.134 | 0.621 |
| cath_C_top1 | _res_AFDB_FID | 16 | 0.301 | 0.257 | 0.263 | 0.325 | 0.116 | 0.668 |
| cath_C_top1 | _res_designability_rate | 16 | 0.229 | 0.394 | 0.212 | 0.430 | 0.567 | 0.022 |
| cath_A_top1 | _res_PDB_FID | 16 | 0.578 | 0.019 | 0.631 | 0.009 | 0.552 | 0.027 |
| cath_A_top1 | _res_AFDB_FID | 16 | 0.565 | 0.023 | 0.600 | 0.014 | 0.553 | 0.026 |
| cath_A_top1 | _res_designability_rate | 16 | 0.402 | 0.122 | 0.473 | 0.064 | 0.769 | 4.97e-04 |
| cath_T_top1 | _res_PDB_FID | 16 | 0.567 | 0.022 | 0.660 | 0.005 | 0.587 | 0.017 |
| cath_T_top1 | _res_AFDB_FID | 16 | 0.550 | 0.027 | 0.612 | 0.012 | 0.569 | 0.022 |
| cath_T_top1 | _res_designability_rate | 16 | 0.323 | 0.223 | 0.377 | 0.150 | 0.680 | 0.004 |
| if_top1_acc | _res_PDB_FID | 16 | 0.842 | 4.24e-05 | 0.912 | 8.82e-07 | 0.909 | 1.09e-06 |
| if_top1_acc | _res_AFDB_FID | 16 | 0.880 | 6.91e-06 | 0.903 | 1.68e-06 | 0.895 | 2.80e-06 |
| if_top1_acc | _res_designability_rate | 16 | 0.481 | 0.059 | 0.458 | 0.075 | 0.596 | 0.015 |
| dih_mae_total_deg | _res_PDB_FID | 16 | -0.821 | 9.73e-05 | -0.935 | 1.07e-07 | -0.924 | 3.11e-07 |
| dih_mae_total_deg | _res_AFDB_FID | 16 | -0.838 | 5.15e-05 | -0.909 | 1.10e-06 | -0.908 | 1.17e-06 |
| dih_mae_total_deg | _res_designability_rate | 16 | -0.340 | 0.197 | -0.202 | 0.454 | -0.393 | 0.132 |

## AFDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 12 | 0.229 | 0.473 | 0.116 | 0.719 | -0.047 | 0.884 |
| cath_C_top1 | _res_AFDB_FID | 12 | 0.172 | 0.592 | 0.044 | 0.893 | -0.094 | 0.771 |
| cath_C_top1 | _res_designability_rate | 12 | 0.073 | 0.821 | 0.211 | 0.510 | 0.603 | 0.038 |
| cath_A_top1 | _res_PDB_FID | 12 | 0.672 | 0.017 | 0.705 | 0.010 | 0.767 | 0.004 |
| cath_A_top1 | _res_AFDB_FID | 12 | 0.712 | 0.009 | 0.765 | 0.004 | 0.801 | 0.002 |
| cath_A_top1 | _res_designability_rate | 12 | 0.448 | 0.144 | 0.487 | 0.108 | 0.739 | 0.006 |
| cath_T_top1 | _res_PDB_FID | 12 | 0.696 | 0.012 | 0.747 | 0.005 | 0.826 | 9.22e-04 |
| cath_T_top1 | _res_AFDB_FID | 12 | 0.712 | 0.009 | 0.754 | 0.005 | 0.798 | 0.002 |
| cath_T_top1 | _res_designability_rate | 12 | 0.302 | 0.341 | 0.448 | 0.144 | 0.665 | 0.018 |
| if_top1_acc | _res_PDB_FID | 12 | 0.833 | 7.64e-04 | 0.888 | 1.14e-04 | 0.925 | 1.60e-05 |
| if_top1_acc | _res_AFDB_FID | 12 | 0.893 | 9.10e-05 | 0.895 | 8.37e-05 | 0.907 | 4.73e-05 |
| if_top1_acc | _res_designability_rate | 12 | 0.330 | 0.295 | 0.266 | 0.403 | 0.558 | 0.059 |
| dih_mae_total_deg | _res_PDB_FID | 12 | -0.821 | 0.001 | -0.937 | 6.99e-06 | -0.940 | 5.40e-06 |
| dih_mae_total_deg | _res_AFDB_FID | 12 | -0.864 | 2.95e-04 | -0.923 | 1.86e-05 | -0.913 | 3.40e-05 |
| dih_mae_total_deg | _res_designability_rate | 12 | -0.083 | 0.798 | 0.067 | 0.837 | -0.237 | 0.459 |

## PDB — all checkpoints

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 23 | -0.488 | 0.018 | -0.423 | 0.045 | -0.436 | 0.038 |
| cath_C_top1 | _res_AFDB_FID | 23 | -0.475 | 0.022 | -0.418 | 0.047 | -0.432 | 0.039 |
| cath_C_top1 | _res_designability_rate | 23 | 0.149 | 0.498 | 0.006 | 0.978 | -0.006 | 0.978 |
| cath_A_top1 | _res_PDB_FID | 23 | -0.378 | 0.075 | -0.330 | 0.125 | -0.349 | 0.103 |
| cath_A_top1 | _res_AFDB_FID | 23 | -0.358 | 0.094 | -0.325 | 0.130 | -0.346 | 0.106 |
| cath_A_top1 | _res_designability_rate | 23 | 0.156 | 0.476 | -0.067 | 0.762 | -0.088 | 0.690 |
| cath_T_top1 | _res_PDB_FID | 23 | -0.335 | 0.118 | -0.253 | 0.245 | -0.280 | 0.195 |
| cath_T_top1 | _res_AFDB_FID | 23 | -0.316 | 0.142 | -0.253 | 0.244 | -0.283 | 0.191 |
| cath_T_top1 | _res_designability_rate | 23 | 0.122 | 0.580 | -0.069 | 0.756 | -0.100 | 0.650 |
| if_top1_acc | _res_PDB_FID | 23 | -0.613 | 0.002 | -0.623 | 0.002 | -0.652 | 7.55e-04 |
| if_top1_acc | _res_AFDB_FID | 23 | -0.583 | 0.003 | -0.627 | 0.001 | -0.659 | 6.31e-04 |
| if_top1_acc | _res_designability_rate | 23 | 0.288 | 0.183 | 0.138 | 0.531 | 0.117 | 0.595 |
| dih_mae_total_deg | _res_PDB_FID | 23 | 0.617 | 0.002 | 0.470 | 0.024 | 0.459 | 0.027 |
| dih_mae_total_deg | _res_AFDB_FID | 23 | 0.617 | 0.002 | 0.468 | 0.024 | 0.455 | 0.029 |
| dih_mae_total_deg | _res_designability_rate | 23 | -0.279 | 0.197 | -0.287 | 0.184 | -0.356 | 0.095 |

## PDB — step >= 200000

| rep_metric | gen_metric | n | pearson_r | pearson_p | spearman_r | spearman_p | partial_spearman_r_ctrl_step | partial_spearman_p_ctrl_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cath_C_top1 | _res_PDB_FID | 19 | -0.667 | 0.002 | -0.523 | 0.022 | -0.500 | 0.029 |
| cath_C_top1 | _res_AFDB_FID | 19 | -0.658 | 0.002 | -0.528 | 0.020 | -0.502 | 0.028 |
| cath_C_top1 | _res_designability_rate | 19 | -0.252 | 0.298 | -0.363 | 0.127 | -0.533 | 0.019 |
| cath_A_top1 | _res_PDB_FID | 19 | -0.572 | 0.010 | -0.492 | 0.033 | -0.478 | 0.038 |
| cath_A_top1 | _res_AFDB_FID | 19 | -0.554 | 0.014 | -0.493 | 0.032 | -0.478 | 0.038 |
| cath_A_top1 | _res_designability_rate | 19 | -0.257 | 0.288 | -0.411 | 0.081 | -0.519 | 0.023 |
| cath_T_top1 | _res_PDB_FID | 19 | -0.511 | 0.025 | -0.411 | 0.081 | -0.402 | 0.088 |
| cath_T_top1 | _res_AFDB_FID | 19 | -0.494 | 0.032 | -0.420 | 0.074 | -0.411 | 0.080 |
| cath_T_top1 | _res_designability_rate | 19 | -0.279 | 0.247 | -0.435 | 0.063 | -0.520 | 0.023 |
| if_top1_acc | _res_PDB_FID | 19 | -0.757 | 1.77e-04 | -0.728 | 4.09e-04 | -0.722 | 4.84e-04 |
| if_top1_acc | _res_AFDB_FID | 19 | -0.734 | 3.45e-04 | -0.751 | 2.12e-04 | -0.745 | 2.49e-04 |
| if_top1_acc | _res_designability_rate | 19 | -0.058 | 0.814 | -0.135 | 0.581 | -0.211 | 0.385 |
| dih_mae_total_deg | _res_PDB_FID | 19 | 0.694 | 9.77e-04 | 0.565 | 0.012 | 0.563 | 0.012 |
| dih_mae_total_deg | _res_AFDB_FID | 19 | 0.696 | 9.34e-04 | 0.579 | 0.009 | 0.572 | 0.011 |
| dih_mae_total_deg | _res_designability_rate | 19 | -0.213 | 0.382 | -0.266 | 0.270 | -0.057 | 0.818 |
