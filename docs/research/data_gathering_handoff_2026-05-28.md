# Data-gathering handoff (2026-05-28) — Proteina chapter follow-ups

Three independent data-gathering tasks deferred from the chapter-writing
session. Pick up in a fresh session; chapter writing continues in parallel —
**do not touch report-draft.tex or proteina-chapter-flow.md from this thread**
unless the task explicitly asks for it.

## Context (read first, in this order)

1. `docs/masters-report/proteina-chapter-flow.md` — current chapter plan; the
   data-checklist (Part 3) lists each task, and the sampler-audit integration
   block (Part 4) explains why each matters.
2. `docs/research/proteina_sampler_regime_audit_2026-05-28.md` — full
   sampler-ablation audit; specifies what's missing and what's been recovered.
3. `docs/research/proteina_narratives.md` — overarching findings doc.

**Memories to load first**:
- `project_afdb_designability_proxy_alignment` — the proxy caveat Task 3 anchors
- `reference_proteina_eval_budget` — ~6h/ckpt for n=256 designability sweeps
- `reference_proteina_ckpt_ema_dupes` — non-EMA vs EMA checkpoint distinction
- `feedback_check_squeue_before_submit` — sanity-check before any sbatch
- `feedback_clean_jsonl_regen` — re-run `clean_variance_jsonl.py` on any
  mutated raw jsonl BEFORE replotting
- `feedback_smoketest_always_first` and `feedback_smoketest_qos_intr` — always
  smoke-test on intr QOS first
- `feedback_confirm_training_config` — confirm config before launching

---

## Task 1 — PDB L9 gen-eval extension  (highest priority)

**Why it matters**: §6.4 of the Proteina chapter currently frames the PDB
acceleration as *transient*: "REPA-L9 leads on FID-PDB at every step ≤1.2M
(−27% at 700K); the baseline matches at ~1.4M and edges below at 1.6M (288)."
The "baseline edges below" half is well-supported (we have baseline data to
1.6M), but the REPA-L9 side stops at 1.2M, so the "REPA plateaus around 330"
claim relies on the last few REPA-L9 ckpts (which are noisy). Filling 1.3M,
1.4M, 1.6M for L9-GN and L9-MPNN would tighten this — *if L9 keeps dropping
past 1.2M*, "transient acceleration" softens further; *if it plateaus*, the
current framing stands.

**What to run**: gen-eval (γ=0.45 SDE + designability + foldseek-novelty +
diversity + SS) for:
- `repa_l9_256_per_residue_bs24_2gpu` at training steps **{1.3M, 1.4M, 1.6M}**
  (1.2M already on disk)
- `repa_mpnn_l9_256_per_residue` at training steps **{1.3M, 1.4M, 1.6M}**
  (1.2M already on disk)

**First check — DO THE CHECKPOINTS EXIST?**
Training runs may have stopped at 1.2M. Look under proteina training output
dirs (typically `/rds/.../proteinfoundation/output/...` or
`/rds/.../wandb_runs/`). If 1.3M+ ckpts don't exist, training needs to extend
first — this is **a separate larger task** not in scope without explicit user
confirmation (extending training is ~weeks of GPU time, not an afternoon).
If checkpoints exist but only at sparse intervals (e.g. 1.4M and 1.6M), use
what's there.

**Cost**:
- **Cheap variant** (recommended first): 1 seed × 3 ckpts × 2 configs ≈ 6 jobs
  × ~6h ≈ 36 GPU-hours. Decide on full sweep based on what cheap shows.
- **Full**: 3 seeds × 3 ckpts × 2 configs = 18 jobs × ~6h ≈ 108 GPU-hours.

**Where output lands**:
`evaluation/proteina/generation/results/paper/n256_convergence_pdb/sweep_results.jsonl`
(append). Then run `evaluation/proteina/generation/scripts/clean_variance_jsonl.py`
to regenerate `.clean.jsonl`, then `evaluation/proteina/generation/scripts/jsonl_to_tsv.py all`
to refresh paper TSVs.

**Sweep config pattern**: see `evaluation/proteina/generation/configs/` for
existing convergence-sweep configs. Smoke-test on intr QOS first.

**On completion**: update Part 3 of `proteina-chapter-flow.md` (cross off the
"PDB L9 extension" line) and write a one-line note in Part 4 with the headline
finding (e.g. "L9-MPNN keeps dropping to ~310 by 1.6M; transient framing
softens" OR "L9-MPNN plateaus at 333; current framing stands"). Then signal
the chapter-writing session to re-read §6.4.

---

## Task 2 — PDB baseline backfill at γ ∈ {0.35, 0.5}  (lower priority)

**Why it matters**: Unlocks `build_sampler_regime_robustness.py` to auto-emit
PDB tables. The current audit (`proteina_sampler_regime_audit_2026-05-28.md`)
already worked around this by recovering baseline rows from git commit
51fddb6, and the chapter's §6.7.2 claims are defensible without backfill —
so this is **clean reproducibility, not load-bearing**.

**What to run**: 14 baseline cells:
`baseline_256_bs24_2gpu` at γ ∈ {0.35, 0.5} × steps
{100, 200, 400, 700, 1000, 1300, 1500}K = 14 (γ, step) pairs.

**Cost**: 14 jobs × ~1h ≈ 14 GPU-hours.

**Where output lands**:
`evaluation/proteina/generation/results/variance/n256_sampler_ablation/sweep_results.jsonl`
(append). Then `clean_variance_jsonl.py` + `jsonl_to_tsv.py all` +
`build_sampler_regime_robustness.py`.

**Config**: existing `n256_sampler_ablation` infrastructure. May need a new
`_baseline_backfill` variant of the sweep config. Smoke-test first.

**On completion**: confirm `build_sampler_regime_robustness.py` now emits
proper PDB tables (it currently skips them with "no step-matched ablation
steps"). Update Part 3 checklist.

---

## Task 3 — Citation hunt for the proxy/data-lineage caveat  (cheap)

**Why it matters**: We have empirical evidence (baseline AFDB Des is 2–5×
PDB at every γ) for the proxy–data alignment claim in
[project_afdb_designability_proxy_alignment](memory). The chapter cites it as
`[CITE]` in two places: **Ch 3 §3.4 NotA-as-prose** and **§6.4 claim 3b**.
We need 1–2 literature anchors.

**What to find**:
1. **Canonical citation** for "designability via ProteinMPNN → ESMFold is an
   in-silico proxy with model lineage; AFDB structures share AF2 ancestry
   with the proxy" — most likely the **Proteina paper** itself
   (`\cite{Proteina}` already in bib, Geffner et al. 2025) — search its
   designability/evaluation discussion for this caveat.
2. **Comparison citation** for "AFDB-trained generators score higher on
   designability than PDB-trained by construction" — candidates:
   - Genie2 (2024)
   - FrameFlow extensions
   - Recent NeurIPS/ICLR 2024–2026 backbone-generator papers
   - AF2 confidence-filtering / pLDDT discussion
3. (Optional, broader) AlphaFold-Database paper (the Tunyasuvunakool et al.
   Nature 2021 paper for the AF2/AFDB lineage itself).

**How**:
- WebFetch the Proteina paper from arxiv (2503.00710) and search for
  "designability", "proxy", "AF2", "MPNN", "ESMFold" — extract any caveats
  in their evaluation section.
- WebSearch for "AFDB-trained protein backbone generator designability" and
  similar; check recent papers' related-work / limitations sections.
- WebSearch for "designability ProteinMPNN ESMFold proxy AlphaFold lineage".

**Output**:
- 1–2 new `\bibitem{}` entries appended to `report-draft.tex` bibliography
  (or confirm existing `\cite{Proteina}` suffices for both).
- The precise sentence those citations would anchor — for the agent doing
  the writing to drop in at the `[CITE]` marker.
- Update Part 3 research-debt in `proteina-chapter-flow.md` to mark done.

**Cost**: <1h Claude time, no GPU.

---

## Coordination

- These tasks are **independent** — run in any order or in parallel.
- Tasks 1 and 2 are GPU spends — **confirm scope with the user before any sbatch**.
- **Do not modify** the chapter writing artefacts (`report-draft.tex`,
  `proteina-chapter-flow.md` Parts 1–2 prose flow) — those are owned by the
  parallel writing session. Updating Part 3 (checklist) and a one-line note
  in Part 4 (decisions log) is fine.
- After each task lands, optionally update the relevant memory entry if a new
  durable claim emerges (e.g. if Task 3 finds a strong citation, update
  `project_afdb_designability_proxy_alignment` with the citation key).
