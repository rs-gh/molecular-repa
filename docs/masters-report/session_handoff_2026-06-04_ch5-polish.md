# Session handoff — Ch5 polish: protocols, accuracy, captions, terminology (2026-06-04)

Continues [session_handoff_2026-06-03_ch5-build.md](session_handoff_2026-06-03_ch5-build.md)
(which built/restructured Ch5). This session was **prose + table polishing**: documented
the sampling/probe **protocols**, made several **accuracy** fixes, reworked **table
captions**, removed value **bolding**, scrubbed the **"student"** term, and rewrote the
Ch5 close. Everything is committed and pushed.

## TL;DR — where we are

- **All work committed + pushed** to `origin/main`. Tip is **`dd99856`**.
  Tree clean except LaTeX build artifacts (`.aux/.bbl/.log/.out/.toc`,
  `tables/texput.log`) and the usual `.bak`/`.propagate*` scratch — all
  intentionally untracked. **Never `git add` those; stage explicit paths.**
- **Ch5 (Tabasco) prose + 3 tables are drafted and now polished.** Compiles in the
  full report at **51 pp** via `make pdf` (clean, exit 0).
- **Word count: 10,378 / 15,000** (texcount, core chapters). Declared number in the
  front matter is now **synced** (see below).
- **Ch3, Ch4, Ch7 still comment skeletons.** Ch5 forward-refs them; they resolve at
  chapter level today. ~4.6k words headroom across the three.
- **Deadline: 11:00 Tue 9 June 2026** (MPhil ACS anonymised report). ~5 days.

## Commits this session (newest first)

- `dd99856` docs(report): sync declared word count to 10,378
- `ba89db5` docs(report): sampler protocols, Ch5 accuracy/captions, trunk terminology

## What changed this session (don't redo)

**Protocols documented (Ch5 metrics §):**
- **Generation** (`sec:tabasco-metrics`, ~l.1140): 1,000 molecules sampled from the
  model, atom counts drawn from the GEOM-drugs size distribution, SDE interpolant
  100 steps, **FCD reference = GEOM-drugs training set**. Verified against
  `evaluation/tabasco/generation/scripts/evaluate.py` (`num_mols=1000`,
  `num_steps=100`) and `chem/convert.py` (DetermineConnectivity → SMILES).
- **Probe** (~l.1144): 1,000 GEOM-**val** molecules, **800 train / 200 held-out**,
  `Ridge(alpha=1.0)`, single split seed 42. From
  `evaluation/tabasco/representation/lib/utils.py` + `scripts/run_all.py`.

**SDE sampler rows added to BOTH setup tables:**
- Tabasco `table_tabasco_setup.tex`: prior noise scale 1.0, Langevin schedule
  `1/(t+0.01)` off for t≥0.9, white-noise scale 0.01. From `flow_model.yaml`.
- Proteina `table_setup_model.tex`: 400 integration steps (dt=0.0025), γ=0.45
  (`sc_scale_noise`, overrides base 0.4), self-conditioning on, gt=1/t schedule.
  From `src/proteina/configs/.../inference/{inference_base,full_eval/inference_fid_60m_baseline}.yaml`.
  **`sc_scale_score` deliberately omitted** — it's "not implemented yet" / dead in
  `r3n_fm.py` (analogous to omitting Tabasco's `time_factor`, which only reweights
  the loss, not sampling).

**Accuracy rewrites (Ch5 Discussion, `sec:tabasco-diagnosis`):**
- **CheMeleon/FCD** (~l.1169): old text claimed CheMeleon "cannot guide the 3D
  geometry that FCD rewards" — **wrong**. FCD scores the *inferred 2D graph* (SMILES
  → ChemNet), so it's essentially conformer-invariant itself; geometry only enters
  via bond perception. Rewrote to lean on **redundancy** (the trunk already holds
  the 2D graph) instead. Conformer-invariance of CheMeleon is real (L2=0 across
  conformers) but was the wrong mechanism to invoke for FCD.
- **MACE "narrowness"** (~l.1171): "narrow" now scoped to *local, energy-tuned*
  content, NOT low-dimensional (MACE is eff-rank 40/192, 0% sparse, **no** bottleneck;
  atom-type probe = 1.000). "Undecodable" scoped to *these global 2D-graph
  descriptors from a mean-pooled embedding*. LogP exception (0.31) restored — it's
  the one descriptor with local-atomic character.

**Table captions / formatting:**
- **Removed all value bolding** from `table_tabasco_gen` (5.2) and
  `table_tabasco_probe` (5.3) — bolding a "winner" fought the null-result message.
- `table_tabasco_gen` caption → "REPA brings no measurable **generation quality**
  gain over the baseline."
- `table_tabasco_probe` caption rewritten to be opinionated (lead: "Linear-probe R²
  reveals little representational difference between REPA and the baseline"), with
  the MACE clause made accurate ("MACE's embedding is too narrow to decode these
  descriptors", not "decodes little semantic information").

**Terminology — "student" purged (teacher–student framework was never introduced):**
- `table_tabasco_probe` header: "student internal representation" → "the model's".
- Proteina gen-rep captions: "student" → "**trunk**" in `table_genrep_corr.tex`,
  `table_genrep_corr_afdb.tex` **and** their generators
  `tables/scripts/make_genrep_corr{,_afdb}.py` (so a regen won't reintroduce it).
- **"teacher" KEPT** at l.1165/1177 ("usable teacher") — it's glossed inline and is a
  named condition, so it stands as plain metaphor. **Open decision:** if you want
  fully framework-free, swap "usable teacher" → "usable target/encoder".

**Ch5 close rewritten** (~l.1177): dropped the "multitudes"/Whitman wordplay (it was
tautological + grammatically inverted). Now: broad domain → **distinct regimes** →
verdict is local → "a **family of interventions**, not a single one" (seeds the Ch6
framing). Used "regimes" (matches report vocab), not "sub-regimes".

**Grammar fixes:** "this information **are**"→"is" (l.1171); "On the contrary"→
"**Indeed**" (l.1116, was a wrong connective — the sentence elaborates, not contrasts);
"the training-set"→"the training set" (l.1140).

**Word count synced:** front-matter "Main chapters word count" 10246 → **10378**
(l.151), PDF rebuilt. NB: only **one** hardcoded spot exists (l.151) — the prior
handoff's "two spots" is stale; the verbatim methodology block doesn't repeat the
number.

## Build / workflow gotchas (updated)

- **Pre-commit hook IS active and will bite.** The `trailing-whitespace` hook strips
  trailing spaces and **aborts the first commit** (modifies the file, exits 1). Fix:
  re-`git add` the affected file and commit again — that's all. (The old handoff said
  use `--no-verify`; letting the hook run + re-staging is cleaner and what I did.)
- **Rebuild PDF with `make pdf`** (pdflatex via makefile, runs rerun-loop; ~51 pp,
  exit 0). `make wordcount` needs no compile. `upquote.sty` still optional via
  `\IfFileExists` — draft compiles without it.
- **The .tex is usually open in the IDE and edited under you** — `Read` immediately
  before each `Edit` or you'll hit "file modified since read" (happened repeatedly).
- **Commit the built `report-draft.pdf`** alongside `.tex` (it's tracked; the user
  commits it each time). Rebuild it before committing if prose changed, so source and
  PDF stay consistent. Stage explicit paths — never `git add -A`.

## Outstanding / next steps

- **Fig 5.1 (still deferred):** epoch-matched validation curves (validity /
  connectivity / novelty) from `evaluation/tabasco/.../validation_curves.csv`. Needs
  a plot script + PNG; `% TODO Fig 5.1` marker in §5.3. FCD is NOT tracked over
  training, so it can't appear there. Still the one missing Ch5 asset.
- **"teacher" decision (above):** keep as metaphor, or purge to "usable target".
- **Commented-out "student"** still in `%` planning notes (l.1020, 1051, 1079, 1198,
  1254, 1256) — don't render; scrub only if you want tidiness.
- **Bootstrap-CI caveat (unchanged):** generation `metrics.json` has CIs only for the
  two MACE variants (validity/connectivity/qed), never FCD/baseline. Keep the
  single-run caveat; don't claim CI significance on FCD.
- **Ch3/Ch4/Ch7 skeletons:** writing Ch3 (evaluation) + Ch4 (profiling) will let
  Ch5's FCD/encoder forward-refs land on real sections. ~4.6k-word budget.
- **Before submission:** re-run `make wordcount`, resync l.151, rebuild PDF.

## Key paths

- Report: `docs/masters-report/report-draft.tex` (Ch5 `ch:tabasco-study` ≈ l.1083;
  Ch6 `ch:proteina-study` ≈ l.1181).
- Ch5 tables: `tables/table_tabasco_{setup,gen,probe}.tex`.
- Ch6 setup/genrep tables: `tables/table_setup_model.tex`,
  `tables/table_genrep_corr{,_afdb}.tex` (+ generators in `tables/scripts/`).
- Tabasco plan / numbers: `docs/masters-report/tabasco-chapter-flow.md`.
- Word-count: `make wordcount`; tool at `tools/texcount.pl`.
- Verification sources: `evaluation/tabasco/{generation,representation}/`,
  `src/tabasco/configs/model/flow_model.yaml`,
  `src/proteina/configs/experiment_config/inference/`.
