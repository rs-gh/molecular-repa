# Session handoff — Proteína sampler ablation & REPA narrative work

Date: 2026-05-27. Hand-off after extended analysis session exploring REPA's distributional effects across encoder × dataset × layer × sampler combinations.

---

## TL;DR for picking up

The session built a comprehensive cross-variant analysis of REPA's effect on the Proteína generative model. Key conclusions:

1. **REPA isn't a single intervention** — it's a family parameterized by encoder × dataset. Different combinations push the model in different (sometimes opposite) directions.
2. **The "700K T-D cliff" is not universal** — appears when REPA loss saturates much earlier than FM loss. Specific to PDB-GearNet and PDB-MPNN; AFDB-L4-GearNet shows gradual convergence instead.
3. **The narrative needs to be configuration-conditional** — "REPA-GearNet on PDB does X" not "REPA does X".

The full living doc is at `docs/research/proteina_narratives.md`. **Read that first** — this handoff is the working-memory-snapshot, that doc is the persistent findings.

---

## What's done

### Plots and figures (paper-ready)

Located at `evaluation/proteina/generation/figures/paper/n256_sampler_ablation/`:

```
pdb/
  sampler_ablation_fid.png              FID/fJSD/fS family (PDB baseline vs REPA L9)
  sampler_ablation_des_quality.png      Des/scRMSD/pLDDT/Div quality metrics
  sampler_ablation_des_dist.png         Novelty/SS-JSD/H-E
  sampler_ablation_ss.png               SS-focused (H, E, H/E, SS-JSD) with des% annotations
  ss_vs_gamma.png                       SS-fraction vs sampler γ at latest ckpt
afdb/
  same 4 plots (baseline AFDB vs REPA L4 GearNet)
sampler_ablation_table.md               Supercolumn-grouped tables (PDB + AFDB) for paper
sampler_ablation_table.csv              Flat CSV with group-prefixed columns
```

All plots regenerable via:
- `evaluation/proteina/generation/scripts/paper/plot_sampler_ablation.py --dataset {pdb,afdb}`
- `evaluation/proteina/generation/scripts/paper/plot_ss_vs_gamma.py`
- `evaluation/proteina/generation/scripts/paper/build_sampler_ablation_table.py`

### Sweeps that were run

1. **n256_sampler_ablation** (PDB ablation):
   - Output: `results/variance/n256_sampler_ablation/sweep_results.clean.jsonl`
   - 5 samplers (ODE, γ=0.0, γ=0.35, γ=0.5, γ=1.0) × 13 ckpts (baseline_256_bs24_2gpu + repa_l9_256_per_residue) = 65 rows
   - γ=0.45 sourced from `results/paper/n256_convergence_pdb/sweep_results.clean.jsonl` (multi-rep, ×3)

2. **n256_afdb_sampler_ablation** (AFDB ablation):
   - Output: `results/variance/n256_afdb_sampler_ablation/sweep_results.clean.jsonl`
   - 5 samplers × 6 ckpts (baseline_afdb_256 + repa_l4_afdb_256, at {100K, 700K, latest}) = 30 rows
   - γ=0.45 from `results/paper/n256_convergence_afdb/sweep_results.clean.jsonl`

### Analyses completed

1. **Sampler ablation tables** — full 16-metric supercolumn-grouped tables for both PDB and AFDB, step-matched and step-mismatched variants.

2. **Sampler-regime robustness check** — confirmed which REPA findings hold across all 5 γ values. `ssJSD-2D` is the most robust REPA advantage; FID and fJSD-A robust at γ ∈ [0, 0.5].

3. **Cross-encoder comparison at AFDB γ=0.45** — Baseline vs L4-GearNet vs L9-GearNet vs L9-MPNN. Found MPNN-AFDB diverges from GearNet behavior on every dimension.

4. **β-stratified diversity (Experiment 1)** — ran `exp_beta_stratified_diversity.py` on 19 (model, ckpt) combinations across PDB+AFDB+GearNet+MPNN. Output: `results/variance/beta_stratified_diversity.json`. **Falsified the "sheets→fewer folds geometric explanation"** — at matched β content, REPA-GearNet still concentrates folds.

5. **SS-class trajectory across all variants** — ran `/tmp/ss_traj_all.py` (could be moved to scripts/paper). Output: `results/variance/ss_class_trajectory_all.json`. Decomposed effects into encoder/dataset/layer-specific patterns.

6. **H2 test** (compositional shift) — confirmed REPA's designable subset transitions α-dominated → balanced → β-dominated for PDB models. Opposite direction for MPNN-AFDB.

7. **H1 test** (loss saturation) — pulled wandb training curves for PDB-L9-GN (REPA saturates at 400K) and AFDB-L4-GN (REPA keeps improving through 1M). MPNN-L9-AFDB blocked on wandb 500s.

8. **PDB cross-encoder β-rich concentration matrix** — at γ=0.45 late training, GearNet-REPA concentrates β-rich on both datasets (pwTM 0.7–0.9), MPNN-REPA concentrates β-rich on PDB only.

---

## Key findings (with confidence levels)

### Strong, multi-dataset evidence

- **REPA accelerates whole-distribution metrics** (FID, fJSD-A, fJSD-C, ssJSD-2D). Holds across γ ∈ [0, 0.5] for both PDB and AFDB. AFDB advantage is durable through 1.7M+; PDB advantage narrows by 1.5M.
- **ssJSD-2D is the single most robust REPA advantage**. ✓ across every γ on both datasets.
- **REPA reaches good designability faster than baseline**. Acceleration effect — baseline catches up on PDB with enough training, doesn't on AFDB.
- **Random GearNet doesn't drive distributional shifts** (PDB-L4-rand control). Confirms the shifts are *learned-representation*-specific.

### Encoder × dataset specific

- **GearNet-REPA shifts designable subset toward β-rich** on PDB (β-rich % grows 0 → 35-47% over training). Modest β-shift on AFDB (β-rich % stays 8-23%).
- **MPNN-REPA shifts designable subset in opposite directions** on PDB (toward β-rich) vs AFDB (toward α-rich).
- **REPA concentrates β-rich folds (Exp 1 finding)** — strongly for GearNet on both datasets and MPNN on PDB. **MPNN-AFDB is the falsifier**: β-rich pwTM ≈ 0.13 (baseline-like, NOT concentrated).
- **T-D cliff at ~700K**: PDB-GearNet (L4, L9), PDB-MPNN (L4, L9), AFDB-L9-GearNet (limited data). NOT seen for AFDB-MPNN. AFDB-L4-GN shows gradual decline instead of cliff.

### Within-step apples-to-apples (not stepmissed)

- At PDB 700K vs 700K, REPA has MORE clusters than baseline. The "REPA reduces T-D" claim is misleading when stepmatched — it's about plateaus vs continued growth, not inherent.
- AFDB at 700K vs 700K: REPA L4 GearNet wins Des%, FID, fJSD on every metric; ties on SS balance.

### Bimodality observation (unverified)

REPA's scRMSD-mean is slightly higher than baseline despite higher Des%, suggesting bimodal sample distribution (clearly designable + clearly broken). Not yet verified with histograms.

---

## Open TODOs

### High priority

1. **MPNN-L9-AFDB wandb pull** — keeps hitting HTTP 500s. Want REPA-loss saturation profile to confirm "saturation timing → cliff dynamics" mechanism for the opposite-direction encoder. Retry when wandb is more responsive.

2. **CATH-A label inspection of REPA's concentrated β-rich folds** — Exp 1 confirms concentration, but which architecture(s)? Likely a single CATH-A class. Useful for the report — "REPA-GearNet converges to producing N copies of architecture X".

3. **Verify scRMSD bimodality claim** — pull per-sample scRMSD from `designability_index.csv` and plot histograms for baseline vs REPA at matched steps. Predict: REPA bimodal, baseline unimodal.

### Medium priority

4. **Run n256_afdb_sampler_ablation at more checkpoints** — currently only have 3 ckpts each for baseline and REPA L4. Limits step-matched comparisons and prevents AFDB T-D-cliff analysis.

5. **fJSD-A on designable subset** — we have fJSD-A on whole set. Adding `_designable` variant would fill the empty cell in the metric framework. Requires extending evaluate.py.

6. **Cross-check n128 vs n256** — does the 700K cliff hold at n=128? The user said defer this but it'd be a useful cross-scale check.

### Lower priority

7. **Run β-stratified diversity for MPNN-L4 on PDB** — would add another data point on encoder × depth interaction.

8. **REPA-L4-rand past 700K** — current ckpts end at 700K. If we trained longer, would the random control eventually also show concentration (suggesting it's any persistent regularization) or stay diverse (confirming learned-rep-specific)?

---

## Important paths and conventions

### Scripts created this session

- `evaluation/proteina/generation/scripts/paper/plot_sampler_ablation.py` (parameterized PDB/AFDB)
- `evaluation/proteina/generation/scripts/paper/plot_ss_vs_gamma.py` (PDB only currently)
- `evaluation/proteina/generation/scripts/paper/build_sampler_ablation_table.py`
- `evaluation/proteina/generation/scripts/paper/exp_beta_stratified_diversity.py`

### Ad-hoc analysis scripts (in /tmp, may be useful to commit)

- `/tmp/ss_traj_all.py` — SS-class trajectory across all model families. Move to `scripts/paper/exp_ss_class_trajectory.py` if useful for the report.
- `/tmp/h1_v3.py`, `/tmp/h1_afdb_gn.py`, `/tmp/h1_mpnn_afdb_v3.py` — wandb pull scripts. Move/refactor as needed.

### Data conventions

- All plot scripts read from `.clean.jsonl`, NOT raw `sweep_results.jsonl`. **Important**: per `feedback_clean_jsonl_regen.md`, always run `evaluation/proteina/generation/scripts/clean_variance_jsonl.py` after any sweep_results mutation before regenerating plots.

- Run name conventions for AFDB models:
  - `baseline_afdb_256` — baseline AFDB-trained
  - `repa_l4_afdb_256`, `repa_l9_afdb_256` — REPA with GearNet (default encoder, no "mpnn" in name)
  - `repa_mpnn_l4_afdb_256`, `repa_mpnn_l9_afdb_256` — REPA with MPNN encoder

- eval_output dir naming:
  - `inference_paper_inference_fid_60m_paper_sweep_{run_id}_step_{step}__{sampler_tag}__rep{N}/`
  - Contains `ss_cache/ss_fractions.npz` (per-sample (H,E,C) and pdb paths)
  - Contains `designability_index.csv` (per-sample designable flag, scRMSD, pLDDT)

### Memory used / extended

- `MEMORY.md` index entries didn't change but I may want to add new ones for:
  - "REPA effects are encoder × dataset conditional, not generic" (feedback_repa_encoder_specific.md)
  - "Cross-encoder β-rich concentration finding from Exp 1" (project_repa_beta_concentration.md)
  - "700K T-D cliff specific to fast-saturating REPA configs" (project_repa_td_cliff.md)

---

## Open narrative questions

1. **Framing choice**: lead with "REPA accelerates convergence" (paper-original framing — safe, expected) or with "REPA reshapes the model's preferred manifold attractor" (our novel observation — more provocative)?

2. **T-D framing**: present REPA's plateaued T-D as a feature ("converges to the *right* folds") or a limitation ("less variety in the long-trained regime")? Depends on whether we report at step-matched (REPA looks favorable) or absolute-best (baseline looks favorable).

3. **Headline narrative for the encoder-direction split**: do we lead with the GearNet-vs-MPNN-AFDB sign reversal as the headline finding (novel, interesting) or treat it as a falsifier control (supportive evidence)?

4. **How prominent should the falsifier controls be**? Random-GearNet (no shift) and MPNN-AFDB (opposite shift) are the cleanest mechanism evidence. Worth a section or worth a paragraph?

5. **The "sheets→fewer folds" hypothesis was falsified** (Exp 1) — worth mentioning as a "we considered this explanation and ruled it out" in the report, or just go straight to the right mechanism?

---

## Conversation context summary

The session built outward from "let me make a sampler ablation plot" to a deep mechanistic investigation of REPA. Key user pivots:

- Started with simple plotting, expanded to multi-sampler ablation
- Recognized data corruption risk (sweep_results.jsonl was getting overwritten); restored clean.jsonl from git
- Pushed for rigor on cross-dataset and cross-sampler robustness
- Asked specifically for β-stratified diversity to test the "sheets→fewer folds" hypothesis
- Wanted full encoder × dataset × layer decomposition to figure out generic-REPA vs specific
- Built supercolumn-grouped result tables for paper use
- Investigated the 700K cliff in detail (H1 loss balance, H2 compositional shift)

The user is building toward final narrative for a master's report. Be precise with claim attribution (what holds across what configurations) — the user is sensitive to overclaiming.
