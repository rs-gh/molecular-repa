# Claim 5 — T-D crossover analysis

Compiles β% / #clusters / pwTM step-matched across all configs, identifies the T-D crossover step (where REPA #clusters Δ flips + → −), and splits all-metric win fractions into before/after that step.

Δ = REPA − baseline, sign-corrected (✓ = REPA better). Crossover from #clusters trajectory.

---

## 1. T-D metric trajectories (Δ vs baseline at same step)

### N256_PDB

**#Clusters ↑**

| variant | 100K | 200K | 400K | 700K | 800K | 900K | 1000K | 1300K | 1600K |
|---|---|---|---|---|---|---|---|---|---|
| L4-GN | +4.00✓ |  | +18.0✓ | +1.00✓ | -8.00✗ | -91.0✗ | -27.3✗ |  |  |
| L9-GN | +0.00= |  | +38.0✓ | +27.0✓ | +6.00✓ | -70.0✗ | -44.3✗ |  |  |
| L4-rand | +0.00= |  | +0.67✓ | -23.0✗ |  |  |  |  |  |
| L4-MPNN | +4.67✓ |  | +30.3✓ | -4.00✗ | +38.3✓ | -30.0✗ | -61.7✗ | -15.0✗ | -44.0✗ |
| L9-MPNN | +0.00= |  | +39.7✓ | +33.0✓ | +32.3✓ | -30.7✗ | -50.7✗ |  |  |

**pwTM ↓**

| variant | 100K | 200K | 400K | 700K | 800K | 900K | 1000K | 1300K | 1600K |
|---|---|---|---|---|---|---|---|---|---|
| L4-GN |  |  | -0.12✗ | -0.11✗ | -0.18✗ | -0.21✗ | -0.01✗ |  |  |
| L9-GN |  |  | -0.03✗ | +0.02✓ | -0.05✗ | -0.09✗ | -0.13✗ |  |  |
| L4-rand |  |  | -0.03✗ | +0.10✓ |  |  |  |  |  |
| L4-MPNN |  |  | -0.18✗ | -0.08✗ | -0.11✗ | -0.13✗ | -0.07✗ | -0.08✗ | -0.08✗ |
| L9-MPNN |  |  | -0.03✗ | +0.01✓ | -0.12✗ | -0.12✗ | -0.15✗ |  |  |

**β% ↑**

| variant | 100K | 200K | 400K | 700K | 800K | 900K | 1000K | 1300K | 1600K |
|---|---|---|---|---|---|---|---|---|---|
| L4-GN |  | +2.76✓ | +8.56✓ | +0.42✓ | +6.33✓ | +11.9✓ | -7.01✗ |  |  |
| L9-GN |  | +1.06✓ | +6.66✓ | -3.92✗ | +3.64✓ | +10.1✓ | +9.24✓ |  |  |
| L4-rand |  | +0.50✓ | +4.08✓ | -11.3✗ |  |  |  |  |  |
| L4-MPNN |  | +4.05✓ | +17.8✓ | +1.14✓ | +5.65✓ | +9.18✓ | +1.67✓ | +10.2✓ | +4.33✓ |
| L9-MPNN |  | +1.66✓ | +10.0✓ | -6.92✗ | +7.58✓ | +11.9✓ | +4.04✓ |  |  |

**T-D crossover (from #clusters)** — last step REPA wins / first step REPA loses:

- L4-GN: between 700K and 800K
- L9-GN: between 800K and 900K
- L4-rand: between 400K and 700K
- L4-MPNN: between 800K and 700K
- L9-MPNN: between 800K and 900K

### N256_AFDB

**#Clusters ↑**

| variant | 100K | 200K | 400K | 700K | 1000K | 1300K |
|---|---|---|---|---|---|---|
| L4-GN | +12.7✓ | -19.0✗ | -33.7✗ | -35.3✗ | -39.0✗ |  |
| L9-GN | +15.0✓ | -47.3✗ | -60.7✗ | -50.3✗ |  |  |
| L9-MPNN | +80.3✓ | +21.3✓ | -19.7✗ | +21.0✓ | +6.00✓ | +36.7✓ |

**pwTM ↓**

| variant | 100K | 200K | 400K | 700K | 1000K | 1300K |
|---|---|---|---|---|---|---|
| L4-GN | +0.02✓ | +0.01✓ | -0.04✗ | -0.04✗ | -0.03✗ |  |
| L9-GN | +0.04✓ | -0.09✗ | -0.14✗ | -0.11✗ |  |  |
| L9-MPNN | -0.00✗ | +0.00✓ | -0.01✗ | +0.03✓ | +0.02✓ | +0.00✓ |

**β% ↑**

| variant | 100K | 200K | 400K | 700K | 1000K | 1300K |
|---|---|---|---|---|---|---|
| L4-GN | +4.39✓ | +3.75✓ | -2.53✗ | -1.72✗ | +0.95✓ |  |
| L9-GN | +8.88✓ | +8.96✓ | +2.93✓ | +3.88✓ |  |  |
| L9-MPNN | +2.07✓ | +4.96✓ | -2.75✗ | -3.53✗ | -3.01✗ | -1.87✗ |

**T-D crossover (from #clusters)** — last step REPA wins / first step REPA loses:

- L4-GN: between 100K and 200K
- L9-GN: between 100K and 200K
- L9-MPNN: between 1300K and 400K

### N128_PDB

**#Clusters ↑**

| variant | 100K | 200K | 300K | 400K | 500K | 600K | 700K |
|---|---|---|---|---|---|---|---|
| L4-GN | -1.53✗ | +0.40✓ | -16.7✗ | -12.1✗ | -0.47✗ | -8.73✗ |  |
| L9-GN | +7.80✓ | +4.73✓ | +4.00✓ | -6.73✗ | +18.5✓ |  |  |
| L4-rand | +3.13✓ | +28.7✓ | +17.0✓ | -9.73✗ | +18.5✓ | +21.6✓ |  |
| L4-MPNN | +0.80✓ | +9.90✓ | +18.7✓ | +6.93✓ |  |  |  |
| L9-MPNN | +12.6✓ | +40.8✓ | +44.0✓ | +34.8✓ | +35.8✓ | +26.6✓ | +38.6✓ |

**pwTM ↓**

| variant | 100K | 200K | 300K | 400K | 500K | 600K | 700K |
|---|---|---|---|---|---|---|---|
| L4-GN | -0.38✗ | -0.20✗ | -0.03✗ | -0.05✗ | -0.05✗ | -0.11✗ |  |
| L9-GN | -0.37✗ | +0.02✓ | +0.05✓ | +0.03✓ | +0.22✓ |  |  |
| L4-rand | +0.01✓ | +0.21✓ | +0.26✓ | +0.23✓ | +0.05✓ | +0.28✓ |  |
| L4-MPNN | -0.11✗ | -0.10✗ | +0.06✓ | +0.04✓ |  |  |  |
| L9-MPNN | -0.00✗ | +0.23✓ | +0.36✓ | +0.38✓ | +0.38✓ | +0.38✓ | +0.38✓ |

**β% ↑**

| variant | 100K | 200K | 300K | 400K | 500K | 600K | 700K |
|---|---|---|---|---|---|---|---|
| L4-GN | +19.4✓ | +6.55✓ | +10.7✓ | +0.82✓ | -8.20✗ | -2.72✗ |  |
| L9-GN | +12.9✓ | -3.61✗ | +4.07✓ | +11.0✓ | -4.07✗ |  |  |
| L4-rand | +2.49✓ | -13.1✗ | -3.31✗ | -11.9✗ | -4.54✗ | -12.4✗ |  |
| L4-MPNN | +10.8✓ | +6.78✓ | +3.45✓ | +1.40✓ |  |  |  |
| L9-MPNN | +0.33✓ | -9.77✗ | -8.19✗ | -3.92✗ | -11.0✗ | -21.0✗ | -15.5✗ |

**T-D crossover (from #clusters)** — last step REPA wins / first step REPA loses:

- L4-GN: between 200K and 300K
- L9-GN: between 500K and 400K
- L4-rand: between 600K and 400K
- L4-MPNN: never crosses (stays ✓ through 400K)
- L9-MPNN: never crosses (stays ✓ through 700K)

### N128_AFDB

**#Clusters ↑**

| variant | 100K | 200K | 400K | 700K | 1000K | 1100K |
|---|---|---|---|---|---|---|
| L4-GN | -8.00✗ | -9.00✗ | -6.00✗ |  |  |  |
| L4-MPNN | -8.00✗ | +2.00✓ | -7.00✗ | +7.00✓ | -10.0✗ | -10.0✗ |
| L9-MPNN | -15.0✗ | -29.0✗ | -21.0✗ |  |  |  |

**pwTM ↓**

| variant | 100K | 200K | 400K | 700K | 1000K | 1100K |
|---|---|---|---|---|---|---|
| L4-GN | -0.00✗ | -0.04✗ | -0.03✗ |  |  |  |
| L4-MPNN | -0.01✗ | -0.00✗ | -0.02✗ | -0.02✗ | -0.05✗ | -0.09✗ |
| L9-MPNN | -0.13✗ | -0.12✗ | -0.08✗ |  |  |  |

**β% ↑**

| variant | 100K | 200K | 400K | 700K | 1000K | 1100K |
|---|---|---|---|---|---|---|
| L4-GN | -6.93✗ | -6.46✗ | -4.22✗ |  |  |  |
| L4-MPNN | +5.76✓ | +2.34✓ | +2.08✓ | +3.31✓ | +2.76✓ | +3.71✓ |
| L9-MPNN | -0.96✗ | -1.74✗ | +0.04✓ |  |  |  |

**T-D crossover (from #clusters)** — last step REPA wins / first step REPA loses:

- L4-GN: insufficient data
- L4-MPNN: between 700K and 400K
- L9-MPNN: insufficient data


---
## 2. Regime-level crossover step (for before/after split)

| Regime | Crossover step | Basis |
|---|---|---|
| n256 PDB | **~850K** | L4-GN & L9-GN #clusters Δ flip between 700K (+) and 1000K (−) |
| n256 AFDB | **~150K** | L4-GN & L9-GN flip between 100K (+) and 200K (−); MPNN never flips |
| n128 PDB | none in range | most variants stay T-D-positive through 600-700K (baseline hasn't overtaken yet) |
| n128 AFDB | <100K | REPA T-D-negative from first ckpt |

**The crossover step is NOT universal** — it tracks when the *baseline's* #clusters growth overtakes REPA's plateau. PDB baseline keeps growing → crossover ~850K (n256) / not-yet (n128). AFDB baseline is high from early → crossover ~150K (n256) / immediate (n128).


---
## 3. Before/after-crossover win fractions for ALL metrics

For regimes with a clear crossover. `before` = steps ≤ crossover, `after` = steps > crossover. Each cell = (#REPA-wins / #comparisons).

### N256_PDB (crossover ~850K)

| Variant | window | FID-PDB | FID-AFDB | fJSD-A | fJSD-C | fS-A | Des% | scRMSD | pLDDT | ssJSD2D | β% | #Clust | pwTM | Nov-PDB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4-GN | before | 3/5 | 3/5 | 4/5 | 3/5 | 5/5 | 5/5 | 3/5 | 5/5 | 4/4 | 4/4 | 3/4 | 0/3 | 1/5 |
| L4-GN | after | 1/2 | 1/2 | 2/2 | 0/2 | 2/2 | 0/2 | 0/2 | 0/2 | 1/2 | 1/2 | 0/2 | 0/2 | 0/2 |
| | |  |  |  |  |  |  |  |  |  |  |  |  |  |
| L9-GN | before | 3/5 | 3/5 | 3/5 | 4/5 | 4/5 | 4/5 | 4/5 | 4/5 | 4/4 | 3/4 | 3/4 | 1/3 | 2/5 |
| L9-GN | after | 2/2 | 2/2 | 1/2 | 2/2 | 2/2 | 1/2 | 0/2 | 2/2 | 2/2 | 2/2 | 0/2 | 0/2 | 2/2 |
| | |  |  |  |  |  |  |  |  |  |  |  |  |  |
| L4-rand | before | 0/4 | 0/4 | 2/4 | 2/4 | 3/4 | 2/4 | 2/4 | 2/4 | 3/3 | 2/3 | 1/3 | 1/2 | 1/4 |
| L4-rand | after | — | — | — | — | — | — | — | — | — | — | — | — | — |
| | |  |  |  |  |  |  |  |  |  |  |  |  |  |
| L4-MPNN | before | 3/5 | 3/5 | 3/5 | 3/5 | 4/5 | 5/5 | 5/5 | 5/5 | 4/4 | 4/4 | 3/4 | 0/3 | 2/5 |
| L4-MPNN | after | 0/4 | 0/4 | 3/4 | 3/4 | 3/4 | 3/4 | 1/4 | 3/4 | 4/4 | 4/4 | 0/4 | 0/4 | 2/4 |
| | |  |  |  |  |  |  |  |  |  |  |  |  |  |
| L9-MPNN | before | 5/5 | 5/5 | 3/5 | 4/5 | 3/5 | 4/5 | 4/5 | 4/5 | 4/4 | 3/4 | 3/4 | 1/3 | 1/5 |
| L9-MPNN | after | 1/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 0/2 | 0/2 | 2/2 |
| | |  |  |  |  |  |  |  |  |  |  |  |  |  |

### N256_AFDB (crossover ~150K)

| Variant | window | FID-PDB | FID-AFDB | fJSD-A | fJSD-C | fS-A | Des% | scRMSD | pLDDT | ssJSD2D | β% | #Clust | pwTM | Nov-PDB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4-GN | before | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 |
| L4-GN | after | 4/4 | 4/4 | 3/4 | 3/4 | 4/4 | 4/4 | 0/4 | 4/4 | 4/4 | 2/4 | 0/4 | 1/4 | 3/4 |
| | |  |  |  |  |  |  |  |  |  |  |  |  |  |
| L9-GN | before | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| L9-GN | after | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| | |  |  |  |  |  |  |  |  |  |  |  |  |  |
| L9-MPNN | before | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 |
| L9-MPNN | after | 4/5 | 3/5 | 3/5 | 2/5 | 3/5 | 3/5 | 3/5 | 5/5 | 3/5 | 1/5 | 4/5 | 4/5 | 4/5 |
| | |  |  |  |  |  |  |  |  |  |  |  |  |  |
