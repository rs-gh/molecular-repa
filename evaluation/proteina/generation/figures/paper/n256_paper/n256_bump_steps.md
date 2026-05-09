# n=256 — confirmed bs=12→24 bump steps

Companion reference for `n256_paper_tables.md`. The bumps cluster around 2026-04-18 (SDPA `.contiguous()` fix unlocked higher bs on all in-flight 256 jobs). Each row's bump step is woven into the Notes column of the corresponding row in `n256_paper_tables.md`; this file is the source of truth.

**Method:** read `nsamples_processed` from each EMA ckpt across the run, compute per-100K-step delta, pin the bump to the segment where the delta transitions from ~12 to ~24. Verified 2026-05-08.

| Run | Bump @ step | Pre-bump samples/step | Post-bump samples/step |
| :--- | ---: | ---: | ---: |
| baseline_256 | ~322K | 12.00 | 24.00 (after step 400K) |
| repa_l0_256_per_residue | ~210K | 12.00 | 24.00 (after step 300K) |
| repa_l4_256_per_residue | ~269K | 12.00 | 24.00 (after step 400K) |
| repa_l9_256_per_residue | ~196K | 12.48 | 24.00 (after step 200K) |

**Why this matters.** Earlier estimates (assumed bump @ ~144K for all per_residue runs) over-counted samples by 0.6–1.5M. **L4 per_residue ep22 is the worst-affected**: estimated 7.87M, actual 6.37M — meaningfully smaller sample budget at the same wall-clock step. Per_sample variants (L0=7.52M @ 381.5K, L4=7.25M @ 400K, L9=7.56M @ 385K) read the same way; their per-run bumps (L0 ~143K, L9 ~145K) are noted inline in their respective rows of the main table.
