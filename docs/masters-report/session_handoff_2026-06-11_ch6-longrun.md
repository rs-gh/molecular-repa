# Session handoff — 2026-06-11 — Ch6 §6.2.4 long-run restructure

Focus this session: rewrote and restructured **Chapter 6 §6.2.4** (the long-run results),
plus its supporting tables/figures. The report compiles clean (`pdflatex` ×2 → 64 pp, no
undefined refs). On-disk files are current; **reload report-draft.tex in your editor**
before typing so you don't save a stale buffer over the edits.

---

## TL;DR — current state of §6.2.4

Title: **"In the long run, REPA gains are data-regime dependent."** Single thread = the
**data regime**; the encoder is demoted to a downstream "instrument." Structure:

1. **Opener** — short-run acceleration is universal; the long run is data-regime dependent;
   the encoder that helps is dictated by the regime.
2. **Gaps beat** — "Each data regime leaves a different gap, and REPA fills the one it is
   given." AFDB: designability easy / distribution hard; PDB: reverse. (Fig 6.1)
3. **AFDB beat** — "REPA learns a genuinely better model": at 1.3M wins Quality + T-W + S-W
   at no cost to T-D/S-D → **pushes the designability–diversity envelope (§3.6)**; ODE
   confirms it's the learned field (Fig 6.6).
4. **PDB beat** — "REPA finds a different spot on the designability–diversity envelope":
   whole-model (T-W,S-W) gains + S-D β-strand shift, but **T-D drops** (collapse onto fewer
   β-rich folds) → **moves *along* the envelope** (trade, mirror of AFDB). Caveat added: the
   trade is **largely mid-training — L9-MPNN's designable diversity recovers to baseline by
   1.6M, even if it never reaches AFDB's durable gain on both axes.**
5. **Encoder beat** — "The right encoder follows from the regime, not the other way around"
   (routing is encoder-intrinsic; the regime selects which specialty is needed). Family-of-
   interventions deferred to §6.3.

§6.3 discussion ("Does alignment outlive its purpose?") + the "doubly conditional" line were
rewritten to the same **data-led / transient-cost** framing (no more "cost widens deep into
training").

---

## Inflight jobs

- **Backfill — job 30393014 (sl3, `computerlab-sl3-gpu --qos=gpu2`): RUNNING (task 0), 1–3 PENDING.**
  Fills the **4 placeholder rows of Table 6.6** at 1.3M with designability+diversity:
  PDB L4-GearNet, PDB L4-random, AFDB L9-GearNet, AFDB L4-MPNN. Checkpoints exist
  (eval-only). Output → `evaluation/proteina/generation/results/variance/n256_13m_backfill/`.
  - **Gotcha that bit us:** a prior backfill (30352629) silently produced FID-only because
    `--ckpts_file` mode drops the designability config. 30393014 passes the flags explicitly:
    `--metrics fid,designability,diversity,ss --designability_subset_per_length 50
    --designability_lengths 50,100,150,200,250`. The fid-only output is parked as
    `n256_13m_backfill/sweep_results.jsonl.fidonly_bak`.
  - **When it lands:** recompute the 4 rows at step 1300000 from the new jsonl, replace the
    step-tagged placeholder rows in `tables/table_proteina_13m.tex` with real Δ rows, drop
    the `\textit{\scriptsize(...)}` step tags. (AFDB L4-random stays "—" / tagged — no ckpt
    past 800K.) Then `pdflatex` ×2. Optionally rerun `clean_variance_jsonl.py` +
    `jsonl_to_tsv.py all` if any clean-jsonl consumer needs these rows.
  - Check: `squeue -j 30393014` / `ls -la evaluation/proteina/generation/results/variance/n256_13m_backfill/`.

- **Reseed (late PDB L9-MPNN) — jobs 30372005/365/366/367: COMPLETED, already propagated.**
  Result: L9-MPNN designable pwTM gap vs baseline = +0.12 (1.3M) → −0.004 (1.6M) → −0.001
  (1.8M). The diversity cost **closes by 1.6M** for L9-MPNN; the middle-layer L4-MPNN
  **holds** (+0.08). Data lives in `n256_late_reseed_pdb/`. Fig 6.5 regenerated; §6.2.4 +
  §6.3 + Fig 6.5 caption already reflect this.

> Two other gen-sweep jobs (30393135/136) were in the queue at handoff — not mine to my
> knowledge; check before assuming.

**Infra:** SL2 GPU-minutes are exhausted (`AssocGrpGRESMinutes`). Route all eval jobs to
**`computerlab-sl3-gpu --qos=gpu2`** (override the hardcoded `#SBATCH -A LIO-CHARM-SL2-GPU`
on the CLI). The sbatch gate is bypassed via `SKIP_SMOKE_GATE=1` already in
`.claude/settings.local.json` env.

---

## Key data / findings (so you don't re-derive)

- **Table 6.6 kept at 1.3M for BOTH blocks.** Why not AFDB@1.0M: the AFDB baseline FPSD
  spikes at 1.0M (469→**534**→386 over 700K/1.0M/1.3M), which *inflates* REPA's margin
  (L4-GearNet FPSD Δ −228 @1.0M vs settled −70 @1.3M), and AFDB-Des is saturated there
  (flat). Why not PDB@1.0M: that's the PDB designability **dip** (L4-MPNN troughs to 0.34).
  1.3M is post-dip + post-spike. Why not PDB@1.6M for the snapshot: GearNet/random variants
  have no late ckpts, and the β≥25 concentration (the thing the PDB block shows) has *closed*
  by 1.6M.
- **#Clust dropped from Table 6.6** (2026-06-11): n-confounded — L9-MPNN #Clust 116 vs base
  107 but designable-n 250 vs 217, while pwTM says *more* concentrated (+0.12). It pointed
  the wrong way. T-D is now pwTM only. §3.6 updated accordingly.
- **Table 6.7 (concentration)** trimmed to baseline + strongest variant per regime
  (PDB L9-MPNN, AFDB L4-GearNet). Shows the concentration is **β-rich-fold-specific** (β≥25
  bin: 0.29→0.69; low-β flat) and PDB-only. Caveat: "relaxes after 1.3M (Fig 6.5)."
- **Fig 6.6 (NEW)** — ODE trajectory, baseline + strongest variant per regime, FPSD+Des over
  training (single-seed, from 400K). AFDB wins both (genuinely better field); PDB wins Des,
  loses FPSD (redistributes). Replaces the old all-variant ODE table.
- **Table 6.7 (old ODE all-variant 700K)** moved to Appendix (`\section{Deterministic (ODE)
  sampling across variants}`, `app:ode`).

---

## Open decisions / follow-ups for you

1. **PDB relaxation framing (eyeball this).** The reseed means the PDB "move along the
   envelope" is now framed as a **transient trade that converges to baseline by 1.6M**, not a
   held trade-off. I wrote the honest version ("recovers to the baseline's by 1.6M, even if
   it never reaches AFDB's durable gain on both axes"). If you want a different framing, this
   is the line to revisit — but "trade-off holds late / still worse at 1.6M" is NOT supported
   by the data (1.6M pwTM 0.174 vs baseline 0.178).
2. **Table 6.6 backfill** — fill the 4 rows when 30393014 lands (above).
3. **AFDB beat** dropped the explicit durability number (best FPSD 282 vs 386, Table 6.3) and
   the random-control falsifier when you rewrote it. No flow problem (Fig 6.6 shows
   persistence), but the 282-vs-386 clause is cheap to restore if you want a crisp long-run
   number in that beat.
4. **Full chapter read-through for flow** once the backfill lands and the dust settles.

---

## Files touched this session
- `docs/masters-report/report-draft.tex` — §6.2.4 restructure, §6.3 rewrite, §3.6 metric
  def, chapter-intro compute caveat, "what we don't claim" bullet, headroom→gaps beat.
- `tables/table_proteina_13m.tex` — #Clust dropped; caption trimmed; placeholder rows remain.
- `tables/table_concentration.tex` — trimmed to strongest variant per regime.
- `figures/fig5_diversity_trend.{png,tex}` — regenerated (reseed data); caption updated.
- `figures/fig6_ode_trajectory.{py,tex,png}` — NEW (ODE trajectory).

## Memory updated
- `project_ch6_longrun_restructure.md`, `project_late_reseed_inflight.md` (now RESOLVED),
  `project_13m_centerpiece_inflight.md` (job 30393014 + the fid-only gotcha), MEMORY.md index.
