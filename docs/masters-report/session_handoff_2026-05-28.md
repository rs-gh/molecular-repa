# Session handoff — Proteina chapter, figures + tables (2026-05-28)

Carrying-forward doc for resuming this work in a new session. Read this top-to-bottom; everything you need to pick up cleanly is here.

## TL;DR

We're building the **Proteina chapter** of the master's report (`docs/masters-report/report-draft.tex`, Ch~\ref{ch:proteina-study}). This session focused on **figures and tables** in preparation for a supervisor presentation. The presentation .tex is in scratch/; the figures and tables are now in `docs/masters-report/figures/` as locked artefacts that the chapter prose will reference.

Sequencing for the work: **figures done → tables done → text remaining**. We agreed to do text last.

## Where the work lives

| Path | Role |
|---|---|
| [docs/masters-report/report-draft.tex](report-draft.tex) | The actual chapter draft; Ch 6 (Proteina) has a detailed scaffold but no prose yet |
| [docs/masters-report/proteina-chapter-flow.md](proteina-chapter-flow.md) | **Canonical plan** — section-by-section topic sentences, claims, figure/table register, thread tagging |
| [docs/masters-report/figures/](figures/) | All locked figure PNGs + scripts in `scripts/` |
| [docs/masters-report/figures/tables_2026-05-28.tex](figures/tables_2026-05-28.tex) | Five tables as LaTeX (compiles to .pdf in same dir) |
| [docs/masters-report/scratch/presentation_2026-05-28.tex](scratch/presentation_2026-05-28.tex) | First-pass presentation doc with embedded figures (now stale; figures/tables it references have been refreshed) |
| [docs/research/data_gathering_handoff_2026-05-28.md](../research/data_gathering_handoff_2026-05-28.md) | Separate handoff for **data jobs** to run in another session (PDB L9 extension, baseline backfill γ=0.35/0.5, citation hunt) |
| [docs/research/proteina_sampler_regime_audit_2026-05-28.md](../research/proteina_sampler_regime_audit_2026-05-28.md) | Full audit of sampler-ablation findings; the source for the sampler-noise table |
| [docs/research/proteina_narratives.md](../research/proteina_narratives.md) | The living findings doc — overall claims and evidence |

## Five narrative threads (used as tags throughout)

- **T1** — REPA does mechanistic work (encoder-selectivity, not regularisation)
- **T2** — Multi-factorial reps + encoder routing (imaging-vs-molecular framing)
- **T3** — Regime asymmetry (PDB vs AFDB tracks baseline convergence)
- **T4** — T-D trade-off (designable concentration on encoder-preferred folds)
- **T5** — Sampler-robustness scaffold (REPA composes with the sampler dial)

## Locked conventions

**Styling** (in [figures/scripts/style.py](figures/scripts/style.py)):
- Color: baseline = blue, L9 = green, L4 = red, random = grey
- Marker: GearNet = circle (o), MPNN = triangle (^), baseline = square (s), random = x
- Multi-seed shaded ±1 SD bands where ≥2 seeds
- Direction arrows in panel titles (↑/↓); non-bold titles
- Training-step x-axes: log-scale with humanised ticks (100, 200, 400, 700, 1000, 1600 K)
- Fig 1 only: y-axis also log (FID convention from REPA paper) with pinned y-ticks {250, 300, 400, 500, 700, 1000}

**Variant selection rule** (trajectory plots): best learned + random control + baseline. Rank-order plots may show all 4 learned. Tables carry full suite.

**Best variant per dataset** (when picking "best" for a single-trajectory figure):
- **PDB**: REPA L9-MPNN — cleanest step-matched FID-PDB winner (5/5 sweep)
- **AFDB**: REPA L4-GearNet — durable winner (FID-AFDB 6.5–13× speedup, baseline never closes)

## Figures (5 locked + 1 outstanding)

All in [figures/](figures/), regenerable via `python3 figures/scripts/figN_*.py`.

| # | File | Status | Claim |
|---|---|---|---|
| 1 | `fig01_fid_convergence.png` | ✓ locked | Headline FID convergence (log-log), PDB+AFDB. Durable AFDB, transient PDB. |
| 2 | `fig02_representation.png` | ✓ locked | 3-panel decodability: IF · dihedral · CATH-A. FM-alone is flat; REPA installs structure; encoder rank-order is axis-specific (MPNN→local, GearNet→fold). |
| 3 | `fig03_alignment.png` | ✓ locked | 2-panel CKNNA: alignment to GearNet target + off-diagonal Platonic convergence to MPNN/ESM2. |
| 4 | `fig04_gen_vs_rep.png` | ✓ locked | Gen-vs-rep envelope with same-compute arrow at 400K connecting baseline → random → L9-MPNN. Title: "REPA delivers better generation quality at the same training compute." |
| 5 | `fig05_td_tradeoff.png` | ⚠ **user not convinced** — pwTM-designable vs pwTM-whole envelope; user said "I'm not convinced by this yet but let's work on the tables for now." Needs revisit. |
| 6 | (none yet) | optional | Sampler-ablation figure — table covers it for now; user has not asked for a figure |

**Fig 5 status**: After trying (a) 2x2 trajectory grid (fS-A + #Clust per dataset), (b) fS-A vs #Clust envelope, (c) pwTM-designable vs pwTM-whole envelope, the user wasn't convinced by (c). The data narrative they want shown is "REPA consistently increases fS-A but tends to decrease T-D" — with the encoder × dataset structure (GearNet shows the trade-off, MPNN-AFDB is the falsifier). Open alternatives on the table:
  1. Bar chart of Δ from baseline at a fixed step (Δ fS-A vs Δ #Clust per variant)
  2. Two stacked trajectory panels (fS-A top, #Clust bottom)
  3. Iterate further on the envelope

## Tables (5 locked, in [figures/tables_2026-05-28.tex](figures/tables_2026-05-28.tex))

| # | What | Notes |
|---|---|---|
| 1 | **700K snapshot (pre-trade-off)** | Supercolumns: Quality (Des, FID) · T-W (fJSD-A, fS-A, pwTM-whole) · T-D (#Clust, pwTM-des) · S-W (ssJSD2D) · S-D (ssJSD2D, E%des). Delta-from-baseline format; baseline row absolute, REPA rows Δ. |
| 2 | **1.0M snapshot (post-trade-off)** | Same columns. Shows every PDB REPA variant turning negative on #Clust (−27 to −62); AFDB L9-MPNN remains the +6 falsifier. The Table 1 → Table 2 contrast makes the temporal nature of the trade-off explicit. |
| 3 | **Speedup table** | AFDB-GearNet 6.5–13× durable; PDB transient ~1.3–1.5× to REPA's plateau. |
| 4 | **Sampler-noise robustness Δ table** | Per-γ Δ values for PDB L9-MPNN and AFDB L4-GN. Six γ columns: ODE, 0, 0.35, 0.45, 0.5, 1.0. |
| 5 | **Rep-quality rank-order** | Best-layer values per encoder; the axis split (MPNN→IF/dihedral, GearNet→CATH) is visible. |

**Conventions for the centerpiece (Tables 1 & 2)**:
- Baseline row: absolute values
- REPA rows: Δ = REPA − baseline (signed)
- `−` rendered as LaTeX `$-$` (don't use unicode minus U+2212 — it silently drops in default font encoding)
- Same metric name in S-W and S-D (e.g. "ssJSD2D" in both); supercolumn disambiguates whole vs designable
- Same for pwTM in T-W (whole) and T-D (designable)
- 10 metric columns + variant column; uses `\resizebox{\textwidth}{!}` to fit

## Data sources

- Convergence sweeps: `evaluation/proteina/generation/results/paper/n256_convergence_{pdb,afdb}/sweep_results.clean.jsonl` (3 seeds, γ=0.45)
- Sampler ablation: `evaluation/proteina/generation/results/variance/n256_{,_afdb}_sampler_ablation/sweep_results.clean.jsonl` (multi-γ, single seed mostly)
  - **PDB baseline missing γ=0.35/0.5** — recover from git commit `51fddb6` (see audit doc)
- Rep quality: `evaluation/proteina/representation/results/paper/n256_{convergence_cleantrain_pdb,xclean_afdb_pdb}/pretrained_sweep_results.csv` (cleantrain for CATH, xclean for IF + dihedral)
- CKNNA: `evaluation/proteina/alignment/results/cknna_matrix_per_residue.jsonl` (n256 PDB, step 1M, t=1.0)
- Whole-set vs designable pwTM: `evaluation/proteina/generation/results/variance/wholeset_vs_designable_diversity.json` (36 cells; from `exp_wholeset_vs_designable_diversity.py`)

## Metric key map (chapter plan supercolumns)

| Supercolumn | Display metrics | Data keys |
|---|---|---|
| Quality | Des, FID | `_res_designability_rate`, `_res_PDB_FID` / `_res_AFDB_FID` (in-distribution per dataset) |
| T-W | fJSD-A, fS-A, pwTM(whole) | `_res_PDB_fJSD_A` / `_res_AFDB_fJSD_A`, `_res_fS_A`, `whole_pwtm` (from json) |
| T-D | #Clust, pwTM(des) | `_res_diversity_clusters_total`, `_res_diversity_pairwise_tm_mean` |
| S-W | ssJSD2D(whole) | `_res_ss_jsd_pdb_2d` / `_res_ss_jsd_afdb_2d` |
| S-D | ssJSD2D(des), E%des | `_res_ss_jsd_pdb_designable_2d` / `_res_ss_jsd_afdb_designable_2d`, `_res_ss_frac_E_designable` |

Removed from earlier drafts: scRMSD, pLDDT (redundant with Des), NVIDIA-60M ceiling reference (same-architecture-more-training, misleading as "ceiling").

## Outstanding decisions

1. **Fig 5 (T-D trade-off) framing** — user not convinced by current pwTM-des vs pwTM-whole envelope. Three alternative paths listed above.
2. **Sampler-ablation Fig 6** — user asked earlier; still not built. Table 4 carries the story numerically.
3. **Length-extrapolation experiment** (n128-trained model on >128-residue proteins) — user flagged as interesting if time permits. NEEDS-DATA.
4. **Citations for proxy-data alignment caveat** — open research-debt item; handed off to a separate session via [data_gathering_handoff_2026-05-28.md](../research/data_gathering_handoff_2026-05-28.md).
5. **Title for Table 1 vs Table 2 contrast** — currently "700K snapshot" and "1.0M snapshot" but could be sharpened to e.g. "pre-trade-off" and "post-trade-off" in the section titles.

## Next-step menu

**A. Refine Fig 5** (the still-open visualisation)
  - Pick one of {bar-chart-of-Δ, stacked-trajectory, iterate-envelope} and build it.
  - User narrative to support: "REPA consistently increases fS-A but tends to decrease T-D; trade-off is encoder × dataset gated."

**B. Build sampler-ablation Fig 6** (if user wants a figure to accompany Table 4)
  - Probably: per-encoder Δ fJSD-A across γ, showing γ-invariance for learned encoders and near-chance for random.

**C. Start writing prose** (the third stage in our sequencing)
  - Port topic sentences from [proteina-chapter-flow.md](proteina-chapter-flow.md) §6.1–6.7 into the `.tex` Ch 6 scaffold.
  - Lead with **§6.2 representation** (cleanest evidence) per the locked order; the scaffold comments in the .tex have it all.

**D. Rebuild the presentation doc** ([scratch/presentation_2026-05-28.tex](scratch/presentation_2026-05-28.tex))
  - The figures/tables it references have been refreshed; need to update content + image paths.

**E. Iterate on Table 1 / Table 2** (the centerpiece)
  - User has been actively shaping these — expect more tweaks (column choice, caption sharpening).

## Other notes / gotchas

- **Unicode minus** (U+2212) silently drops in default LaTeX font encoding — always use `$-$` or `\textendash` if a minus needs to render. The tables had 82 invisible minuses earlier; all fixed now.
- **Multi-seed numbers**: most n256 PDB cells are 3-seed. Known exceptions: AFDB-L9-GN early steps (100–600K) single-seed; MPNN-L4-AFDB single-seed at most steps.
- **AFDB designability proxy caveat**: AFDB-trained models score 2–5× higher on designability at every γ because ProteinMPNN→ESMFold shares folding-model lineage with AFDB (AF2). This is documented in memory `project_afdb_designability_proxy_alignment` and Ch 3 §3.4 NotA. Don't quote cross-dataset designability headline numbers without this caveat.
- **`build_sampler_regime_robustness.py` doesn't auto-emit PDB tables** because PDB baseline lost γ=0.35/0.5 rows in the 2026-05-28 clean.jsonl refresh. Audit doc handles the work-around (recover from git `51fddb6`).
- **The chapter-flow doc Part 4 ("Sampler-audit integration") and Part 5 ("Computed numbers")** are the canonical reference for thread mapping and concrete Δ values. Read those before writing prose.

## Recompile recipes

```bash
# Regenerate any figure
cd docs/masters-report/figures/scripts
python3 fig1_headline_fid.py   # or fig2_*, fig3_*, fig4_*, fig5_*

# Recompile tables
cd docs/masters-report/figures
pdflatex tables_2026-05-28.tex

# Recompile the (stale) presentation
cd docs/masters-report/scratch
pdflatex presentation_2026-05-28.tex
```

To recompute the table numbers from scratch, the data-pulling logic is in this handoff's discussion above and matches the snippets used to populate the .tex tables. The 700K and 1.0M snapshots, speedup, sampler-Δ table, and rep rank-order all read from the data sources listed in §Data sources.
