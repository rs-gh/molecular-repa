# n=128 cliff check (γ=0.45, multi-seed aggregated)


## PDB

| run group | trajectory (step → #clust mean) | verdict | drop |
|---|---|---|---|
| baseline_128_bs80 | 100k→28.2, 200k→34.6, 300k→23, 400k→25.4, 500k→13.8, 600k→19.4, 700k→16.4, 800k→20.4, 900k→24.4 | soft_decline | 20.8/0.6 |
| repa_l4_128_bs80 | 100k→26.7, 200k→35, 300k→6.3, 400k→13.3, 500k→13.3, 600k→10.7 | cliff | 28.7/0.82 |
| repa_l4_128_random | 100k→31.3, 200k→63.3, 300k→40, 400k→15.7, 500k→32.3, 600k→41 | soft_decline | 47.7/0.75 |
| repa_l9_128_bs80 | 100k→36, 200k→39.3, 300k→27, 400k→18.7, 500k→32.3 | soft_decline | 20.7/0.53 |
| repa_mpnn_l4_128_bs80 | 100k→29, 200k→44.5, 300k→41.7, 400k→32.3 | soft_decline | 12.2/0.27 |
| repa_mpnn_l9_128_bs80_2gpu | 100k→40.8, 200k→75.4, 300k→67, 400k→60.2, 500k→49.6, 600k→46, 700k→55 | soft_decline | 29.4/0.39 |

## AFDB

| run group | trajectory (step → #clust mean) | verdict | drop |
|---|---|---|---|
| baseline_afdb_128_bs80 | 100k→54, 200k→58, 400k→53, 700k→37, 1000k→40, 1100k→35, 1200k→38 | soft_decline | 23/0.4 |
| repa_l4_afdb_128_bs80 | 100k→46, 200k→49, 400k→47, 500k→40.7, 600k→40.7 | stable | 8.3/0.17 |
| repa_mpnn_l4_afdb_128_bs80 | 100k→46, 200k→60, 400k→46, 700k→44, 1000k→30, 1100k→25 | cliff | 35/0.58 |
| repa_mpnn_l9_afdb_128_bs80_2gpu | 100k→39, 200k→29, 400k→32, 500k→22.7, 600k→23.7 | cliff | 16.3/0.42 |
