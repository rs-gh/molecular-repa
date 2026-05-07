# n=128 paper-protocol sweep — ablation table

Companion to `n128_paper_sweep.png`. Pool: 500 PDBs at L∈{50,75,100,125} × 125 for FID/fJSD/fS_T (N=500); designability on 50/L × 4 lengths (N=200); diversity on the designable subset. Rows whose `des N` < 175 have downstream metrics (designability, scRMSD, diversity) suppressed — they hit the PDB-index-shift bug and the FID family is the only safe column.

**N per metric:** PDB FID=N=500, AFDB FID=N=500, Fold Score (Topo)=N=500, PDB fJSD (Topo)=N=500, Designability=N=200, scRMSD mean (Å)=N=200, Diversity (clusters)=designable, Diversity (pairwise TM)=designable.

Best per metric within each ablation block is **bolded**.

| Run | Step | bs | des N | PDB FID (↓) | AFDB FID (↓) | Fold Score (Topo) (↑) | PDB fJSD (Topo) (↓) | Designability (↑) | scRMSD mean (Å) (↓) | Diversity (clusters) (↑) | Diversity (pairwise TM) (↓) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Layer ablation — (L0/L4/L9 vs baseline, bs=80)** |  |  |  |  |  |  |  |  |  |  |  |
| Baseline | 200K | 80 | 175 | **332.7** | **324.5** | 13.36 | 2.976 | 0.737 | 2.022 | 14.25 | 0.394 |
| REPA L0 | — | 80 | — | — | — | — | — | — | — | — | — |
| REPA L4 | 200K | 80 | 200 | 347.0 | 348.5 | **15.50** | **2.658** | **0.825** | **1.856** | **23.50** | 0.377 |
| REPA L9 | 63K | 80 | 200 | 541.4 | 545.7 | 10.83 | 3.414 | 0.055 | 7.632 | 10.00 | **0.134** |
| **Encoder ablation — (REPA L4 with 6 target encoders)** |  |  |  |  |  |  |  |  |  |  |  |
| Baseline | 200K | 80 | 175 | **332.7** | **324.5** | 13.36 | 2.976 | 0.737 | 2.022 | 14.25 | 0.394 |
| CA-GearNet | 200K | 80 | 200 | 347.0 | 348.5 | **15.50** | **2.658** | **0.825** | **1.856** | **23.50** | **0.377** |
| GearNet random | — | 80 | — | — | — | — | — | — | — | — | — |
| PW-Structure | — | 80 | — | — | — | — | — | — | — | — | — |
| PW-Torsional | 100K | 80 | 200 | 521.8 | 496.4 | 8.706 | 3.285 | 0.420 | 2.667 | 12.00 | 0.462 |
| ProteinMPNN | — | 80 | — | — | — | — | — | — | — | — | — |
| ESM2 | — | 80 | — | — | — | — | — | — | — | — | — |
| **Batch size + LR ablation — (bs ∈ {24,80} × lr ∈ {1×,3×} × ±REPA)** |  |  |  |  |  |  |  |  |  |  |  |
| BL bs24 200k | 200K | 24 | 200 | 315.6 | 318.3 | 27.68 | 2.329 | 0.050 | 7.230 | 2.500 | **0.136** |
| L4 bs24 200k | — | 24 | — | — | — | — | — | — | — | — | — |
| BL bs24 400k | 400K | 24 | 200 | **306.6** | **309.8** | **30.78** | **2.129** | 0.260 | 4.824 | 15.00 | 0.172 |
| L4 bs24 400k | — | 24 | — | — | — | — | — | — | — | — | — |
| BL bs80 200k | 200K | 80 | 175 | 332.7 | 324.5 | 13.36 | 2.976 | 0.737 | 2.022 | 14.25 | 0.394 |
| L4 bs80 200k | 200K | 80 | 200 | 347.0 | 348.5 | 15.50 | 2.658 | **0.825** | **1.856** | **23.50** | 0.377 |
| BL bs80 lr3× 200k | — | 80 | — | — | — | — | — | — | — | — | — |
| L4 bs80 lr3× last | — | 80 | — | — | — | — | — | — | — | — | — |
