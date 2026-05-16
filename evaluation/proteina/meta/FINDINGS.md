# Meta-evaluation findings (2026-05-14)

Variance + FID-scaling diagnostics for the proteina generation/eval pipeline,
driven by `run_meta_sweep.py` + `sweep_config.yaml` in this directory.

Each variance row is one fresh `(seed, ckpt)` rep with its own PDB pool. The
`__seed<S>` suffix on `ckpt_label` is what forced regeneration — the standard
sweep reuses cached PDBs whenever `output_suffix` matches, so simply changing
`--seed` doesn't change the metric values. See `run_meta_sweep.py` for the
wrapper that makes per-rep `output_suffix` differ.

Profiles disable centroid novelty + foldseek for speed (not part of the
variance question and adds ~15 min/task at n=256).

Source JSONL:
- [results/meta/variance_n128_layer/sweep_results.jsonl](../generation/results/meta/variance_n128_layer/sweep_results.jsonl)
- [results/meta/variance_n256_layer/sweep_results.jsonl](../generation/results/meta/variance_n256_layer/sweep_results.jsonl)
- [results/meta/fid_scaling_n128/sweep_results.jsonl](../generation/results/meta/fid_scaling_n128/sweep_results.jsonl)
- [results/meta/fid_scaling_n256/sweep_results.jsonl](../generation/results/meta/fid_scaling_n256/sweep_results.jsonl)

---

## 1. Between-seed variance — n=128 paper protocol

500-PDB FID pool, 200 designability PDBs (50/length × 4 lengths {50,75,100,125}).
5 seeds (44-48).

### baseline_128_bs80_step200k (anchor, strong)

| Metric | μ | σ | CV | Range |
|---|---|---|---|---|
| FID (PDB) | 316.41 | 5.91 | **1.87%** | [309.14, 324.39] |
| fJSD_C | 0.2570 | 0.0177 | 6.89% | [0.2369, 0.2762] |
| Designability rate | 0.725 | 0.028 | 3.87% | [0.690, 0.765] |
| Diversity (clusters) | 72.2 | 6.98 | 9.67% | [67, 84] |
| Diversity (mean pTM) | 0.349 | 0.020 | 5.73% | [0.320, 0.370] |

### repa_l0_128_bs80_step200k (weak — desig ~0.38)

| Metric | μ | σ | CV | Range |
|---|---|---|---|---|
| FID (PDB) | 462.27 | 14.89 | 3.22% | [439.26, 477.35] |
| fJSD_C | 0.5571 | 0.0652 | 11.70% | [0.4968, 0.6605] |
| Designability rate | 0.377 | 0.046 | 12.14% | [0.320, 0.445] |
| Diversity (clusters) | 58.4 | 8.08 | 13.84% | [48, 70] |
| Diversity (mean pTM) | 0.252 | 0.011 | 4.20% | [0.238, 0.265] |

---

## 2. Between-seed variance — n=256 paper protocol

1125-PDB FID pool, 250 designability PDBs (50/length × 5 lengths
{50,100,150,200,250}). 5 seeds (44-48). One ckpt only: `baseline_256_ep21`.

| Metric | μ | σ | CV | Range |
|---|---|---|---|---|
| FID (PDB) | 387.35 | 6.75 | **1.74%** | [379.41, 394.70] |
| fJSD_C | 0.4274 | 0.0306 | 7.15% | [0.388, 0.465] |
| Designability rate | 0.130 | 0.015 | 11.26% | [0.104, 0.140] |
| Diversity (clusters) | 32.0 | 3.94 | 12.30% | [25, 34] |
| Diversity (mean pTM) | 0.148 | 0.008 | 5.29% | [0.136, 0.157] |

---

## 3. Headline takeaways

1. **FID is rock-solid across reps.** CV under 3.2% even on the weak repa_l0.
   Absolute σ is 6-15 on values of 300-475. **Differences smaller than ~15
   between checkpoints at fixed N are within seed-to-seed noise.** For the
   strong baseline at fixed protocol the σ floor is ~6.
2. **Designability is at the binomial floor.** For baseline_128_bs80 at
   N=200, the analytical SE = √(p(1-p)/N) = √(0.725·0.275/200) = 0.032 —
   empirical σ = 0.028. So tightening protocol won't tighten the metric;
   only growing N will.
3. **Diversity cluster count is the noisiest single number** (CV 10-14%) —
   absolute σ=4-8 clusters. Ranks between very-different ckpts should still
   survive, but small (<10-cluster) gaps are noise.
4. **Variance scales with quality / metric magnitude.** Weak repa_l0 has
   2.5× the FID σ and 1.6× the designability σ of the strong baseline. So
   when comparing weak models the "is this difference real" threshold should
   be much larger than for strong models.
5. **fJSD_C is meaningfully noisier than FID** (CV 7-12% vs 2-3%) — surprising
   given they share the GearNet embedding. Worth investigating; probably the
   JSD on the C-only marginal is more sample-sensitive than the full FID.

---

## 4. FID scaling — does the 500/1125-PDB FID converge?

One run per (target, N), seed=42 — quantifies the small-N bias of FID.

| Target | N | FID (PDB) | Compare to paper-N μ | Δ |
|---|---|---|---|---|
| baseline_128_bs80_step200k | **5000** | 287.38 | μ(N=500)=316.41 (σ=5.91) | **−29.03** (−4.9σ) |
| baseline_256_ep21 | **5625** | 375.39 | μ(N=1125)=387.35 (σ=6.75) | **−11.96** (−1.8σ) |

**FID drops substantially as N grows.** This is the well-known upward bias of
the sample-FID estimator — the covariance trace term converges from above
in 1/N. Both numbers are well below the paper-N seed-band (n=128 by 4.9σ).

**Implication:** FID values **are not comparable across N**. Cross-ckpt
ranking at fixed N is fine, but stay within one N when reporting absolute
numbers. If we want a less-biased absolute FID, going to N=5625 at n=256 buys
~12 units of reduction (~2σ) on top of just being more stable.

---

## 5. Timing — what the meta protocol actually costs

Measured from SLURM log start/end timestamps (`/rds/.../meta-sweep-*.out`):

| Profile | N (FID) | N (des) | Wall (min) | Notes |
|---|---|---|---|---|
| variance_n128_layer (10 tasks) | 500 | 200 | 46–84 (median 53) | Faster than 1h47 paper-protocol because foldseek + centroid disabled |
| variance_n256_layer (5 tasks) | 1125 | 250 | 120–145 | Same shape as paper sweeps; n=256 designability dominates |
| fid_scaling_n128 (1 task, FID-only) | 5000 | — | **74** | Linear gen scaling: ~9 min × 10 ≈ 90 min predicted; 74 actual |
| fid_scaling_n256 (1 task, FID-only) | 5625 | — | **158** | Confirms 27 min × 5 = 135 + FID compute ≈ 158 |

So scaling FID to N=5625 across the whole n=256 paper grid would cost
**~2h40/ckpt vs ~2h05 today (+35 min/ckpt)**, in exchange for a meaningfully
better-converged FID estimate.

---

## 6. Per-length designability cost model

Extracted from the per-PDB `Designability [k/N]` log timestamps in the
variance task-0 logs (`meta-sweep-29309222_0.out` for n=128,
`meta-sweep-29309263_0.out` for n=256). Each "Designability [k]" line marks
completion of one PDB through ProteinMPNN(8 seq) → ESMFold(×8) → scRMSD.

Per-PDB designability seconds, split by residue length (50 PDBs per block):

| Length | n=128 s/PDB | n=256 s/PDB | Avg |
|---|---|---|---|
| 50 | 9.06 | 7.38 | 8.2 |
| 75 | 10.20 | — | 10.2 |
| 100 | 13.86 | 15.00 | 14.4 |
| 125 | 19.18 | — | 19.2 |
| 150 | — | 18.92 | 18.9 |
| 200 | — | 27.26 | 27.3 |
| 250 | — | 38.98 | 39.0 |

**Power-law fit:** `t(L) ≈ a · L^k` → log-log slope `k ≈ 0.97`. Essentially
**linear in L** in the 50–250 range. Closed form for back-of-envelope budgets:
`t(L) ≈ 0.155 · L seconds per PDB`. (At larger L the ESMFold attention term
will reassert and push toward O(L²), so this is a 50–250 fit only.)

Designability scales linearly in `N_per_length` at fixed L — ESMFold processes
PDBs one at a time, no batch amortisation.

---

## 7. Wall-clock budget at paper-scale

Combining the FID-scaling generation timings with the per-length designability
model. "Paper-scale" here = generate N≈paper-sized pool for FID/fJSD, then
designability on 100/length subset of that pool (one PDB pool feeds both
metric families — designability does **not** regenerate).

Calculation inputs:
- Gen 5625 PDBs (n=256) + FID: 158 min ≈ 2h38 (measured)
- Gen 5000 PDBs (n=128) + FID: 74 min ≈ 1h14 (measured)
- Designability 100/length × {50,100,150,200,250} at n=256:
  100·(7.38+15.00+18.92+27.26+38.98) = 10,754 s ≈ **2h59**
- Designability 100/length × {50,75,100,125} at n=128:
  100·(9.06+10.20+13.86+19.18) = 5,230 s ≈ **1h27**

| Protocol | Gen + FID | Designability (100/len) | Total / ckpt |
|---|---|---|---|
| **n=128 paper-scale** (5000 FID, 400 des) | 1h14 | 1h27 | **~2h41** |
| **n=256 paper-scale** (5625 FID, 500 des) | 2h38 | 2h59 | **~5h37** |

Round to **3h** (n=128) / **6h** (n=256) per ckpt with comfortable margin.

For other N-per-length choices at n=256, designability scales linearly:
- 50/len = current protocol: 1h30
- 100/len = paper-scale interpretation: 3h00
- 200/len = Proteina App. F: 6h00
- 500/len = full paper: 14h55

At n=256 **L=200/250 dominate the designability bill** (~50% of wall). If you
want tighter FID without paying for long-length designability scaling, you
can keep 50/length for des while taking the FID pool to N=5625.

---

## 8. Open follow-ups

- **Repeat variance at n=256 with a weak ckpt** (e.g. one of the repa_l9
  early-epoch points) to confirm the "variance scales with quality"
  observation holds at n=256, not just n=128.
- **Decide whether to bump FID to N=5625 default** for the canonical n=256
  paper sweeps. The +35 min/ckpt is cheap relative to designability cost
  and would shrink the FID floor by ~50% (σ goes as √N).
- **fJSD_C noise floor (CV ~7-12%)** — worth understanding before relying
  on small fJSD_C deltas in cross-method comparisons.
- **Designability is binomial-floor-limited** — for any individual ckpt
  comparison where the gap is <0.05, N=200/250 is too small.
