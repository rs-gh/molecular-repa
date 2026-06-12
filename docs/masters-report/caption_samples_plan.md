# Plan — adding "samples per seed" to captions

**No edits made.** For your review. This builds on the seed annotations already
applied; it proposes adding the **sample count per seed** alongside them.

---

## Ground truth (from §Evaluation protocol — these are already stated there)

The sample count is **metric-dependent**, and most floats show >1 metric type,
so a float often has **two** sample counts:

| Metric type | Samples per seed | Floats using it |
|---|---|---|
| **Whole-set** (FPSD, fJSD-A, fS-A, ssJSD-W) | **1,125 backbones** (125 × 9 lengths) | gen figs/tables |
| **Designability + designable-subset** (Des, pwTM, ssJSD-D, β%) | **250 backbones** (50 × 5 lengths), subset = those passing | gen figs/tables |
| **Representation probe** | fit on **1,000 chains**; eval **~325 proteins / ~43k residues** (per-residue), **~3,190 chains** (CATH) | rep figs/tables |
| **CKNNA** | **10,000 residues**; bootstrap 50× @80% | alignment fig/table |
| **Tabasco gen** | **1,000 molecules** (already in caption); curves 100/epoch | Ch5 |

**Note:** all of this is *already* written out in the §Evaluation protocol
prose. Adding it to captions is partly redundant — but makes each float
self-contained. Your call on the tradeoff vs. caption length.

---

## The tension

For generation floats the addition is clean: `1,125 backbones/seed, 250 for
designability`. For **representation** floats it's bulky — two eval-set sizes
(~325 proteins / ~3,190 CATH chains) that aren't really "samples per seed" so
much as "eval-set size." So I recommend a **split decision**:

- **Option 1 (recommended): generation floats only.** Add the backbone count
  to the 11 generation floats (where "samples per seed" is most meaningful and
  compact). Leave representation/CKNNA floats to the protocol section.
- **Option 2: all floats.** Also add probe eval-set sizes and CKNNA residue
  count. More complete, but lengthens the rep captions noticeably.

Below I list every float and the exact addition under each option.

---

## Generation floats (both options add these)

Standard insert: `; 1{,}125 backbones/seed, 250 for designability` (or just
`250 backbones/seed` where the float is designability-only).

| Float | Metrics shown | Proposed sample clause |
|---|---|---|
| **Fig 6.1** proteina-fid | FPSD + Des | `1{,}125 backbones/seed, 250 for designability` |
| **Fig 6.4** proteina-genrep | FPSD + Des (vs CATH-A) | `1{,}125 backbones/seed, 250 for designability` |
| **Table 6.3** speedup *(auto-gen)* | Des + FPSD | `1{,}125 backbones/seed, 250 for designability` |
| **Table 6.4** sampler | FPSD + Des | `1{,}125 backbones/seed, 250 for designability` |
| **Table 6.5** genrep-corr *(auto-gen)* | FPSD + Des (corr) | `1{,}125 backbones/seed, 250 for designability` |
| **Table 6.6** proteina-13m | whole-set + designable | `1{,}125 backbones/seed, 250 for designability` |
| **Table 6.7** ode-floor | Des + FPSD | `1{,}125 backbones/seed, 250 for designability` |
| **Table 6.8** concentration | designable pwTM only | `250 backbones/seed (designable subset)` |
| **A.6** genrep-corr-afdb *(auto-gen)* | FPSD + Des (corr) | `1{,}125 backbones/seed, 250 for designability` |
| **A.7** ss-composition | designable subset | `250 backbones/seed (designable subset)` |
| **A.8** ode (appendix) | Des + FPSD + pwTM | `1{,}125 backbones/seed, 250 for designability` |

Example, Table 6.6 (current → proposed end of caption):
> …AFDB L4-MPNN, $n{=}1$.)
> →  …AFDB L4-MPNN, $n{=}1$; $1{,}125$ backbones/seed, $250$ for designability.)

---

## Representation + CKNNA floats (Option 2 only)

| Float | Metrics | Proposed sample clause |
|---|---|---|
| **Fig 6.2** proteina-rep | IF, dihedral, CATH-A | `probe eval $\sim$325 proteins / $\sim$3{,}190 CATH chains` |
| **Table 6.2** rep-quality | CATH + IF + dihedral | `probe eval $\sim$325 proteins / $\sim$3{,}190 CATH chains` |
| **A.3** rep-quality-full *(auto-gen)* | same | same |
| **A.4** rep-quality-afdb *(auto-gen)* | same (cross-db) | `probe eval $\sim$325 proteins (cross-db blinded)` |
| **Fig 6.3** proteina-cknna | per-residue CKNNA | `over $10{,}000$ residues` |
| **A.5** cknna-matrix *(auto-gen)* | CKNNA | `over $10{,}000$ residues` |

---

## Tabasco (Ch5) — already partly covered

| Float | Current | Action |
|---|---|---|
| **Fig 5.1** tabasco-curves | "100 molecules sampled per epoch" | already states it; no change |
| **Table 5.2** tabasco-gen | "1{,}000 sampled molecules" | already states it; no change |
| **Table 5.3** tabasco-probe | "held-out molecules" | could add probe-set size if I look it up |

---

## Mechanics (same as last time)

- **Auto-generated tables** (6.3, 6.5, A.3, A.4, A.5, A.6): I'll edit the `.tex`
  caption directly (preserving current data) **and** the generator caption
  string, to avoid pulling in the unreviewed data drift again.
- Inserted into the existing trailing parenthetical, semicolon-separated, same
  tight style.

---

## Questions

1. **Option 1 (generation floats only) or Option 2 (all floats)?** — I lean 1.
2. **Wording:** `1{,}125 backbones/seed, 250 for designability` OK, or tighter
   (e.g. `n_gen = 1{,}125 / 250`)?
3. **Table 5.3** (Tabasco probe): want me to look up and add its probe-set size,
   or leave it?
