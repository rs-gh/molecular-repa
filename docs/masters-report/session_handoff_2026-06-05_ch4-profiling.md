# Session handoff — Ch4 (encoder profiling): §4.1 recast as a profiling protocol, "recoverability" framing (2026-06-05)

This session built up and then **conceptually reworked Chapter 4** ("Encoder targets and
profiling"). The headline outcome is a redrafted **§4.1** that reframes the second profiling
property from a "reachability gap (bigger = better)" into **recoverability** — a cheap-input
*ladder* read as a *lower bound* — which fixes a framing that was quietly backwards. That
reframing now needs to propagate into §4.2/§4.3 (the next job).

## TL;DR — where we are

- **Tip is `8ee9353`** ("update Ch4 profiling tables + molecule-encoder figure PDF").
  Recent Ch4 chain: `cf6785c` (chapter + Ch3 intro) → `d40a1f4` (CheMeleon figs) →
  `4e6a302` (mol figure) → `f003a46` (figure fix) → `a76a602` (Ch4 intro) →
  `f33faca` (§4.1 as protocol) → `8ee9353` (recoverability redraft + tables).
- **One small uncommitted edit** in `report-draft.tex` (an in-flight `\emph{...}` tweak near
  l.1087 — yours; commit/push is being handled separately).
- **Ch4 compiles clean at 57 pp.** Word count **12,665 / 15,000** (texcount).
- **Deadline: 16 June 2026** (the prior handoffs said 9 June — that was corrected this
  session; treat 16 June as the working date unless 9 June is a separate earlier milestone).

## The §4.1 framework as it now stands (and why)

§4.1 is **"A profiling protocol"** — an experimental-setup section (like the Ch5/Ch6 setup
sections), with §4.2–4.4 as the "results." It contains:

- **Figure 4.1** (`figures/fig_ch4_framework.tex`) — a TikZ funnel: *a candidate encoder →
  Information → Recoverability → Conditioning → verdict (guides the two studies)*.
  Plain-text verdict (de-emphasised on purpose). One-line caption.
- **Table 4.1** (`tables/table_profiling_diagnostics.tex`) — the diagnostics, grouped by the
  three properties, with upper-/lower-bound glosses. Information has four probe rows (3D
  geometry, token identity, object identity, secondary structure); Recoverability is the
  **ladder** row; Conditioning is rank/sparsity/norm.
- **Prose** — protocol paragraph (sample sizes), then one paragraph per property, then a
  synthesis paragraph.

**The three properties (named — we dropped Q1/Q2/Q3):**

1. **Information** = what the encoder encodes. *Upper bound* on what REPA can transfer.
   Probes: 3D geometry (coord-perturbation cosine), token identity (linear probe), object
   identity (within−between cosine), secondary structure (helix/sheet/loop, proteins).
2. **Recoverability** = how much a projector can recover, **and from what** — *the load-bearing,
   subtle one* (see next section).
3. **Conditioning** = is the embedding well-formed? Encoder is **frozen**, so it is NOT about
   gradients into it — it is about whether a collapsed (low effective rank) or sparse embedding
   is a **bottleneck** through which no signal flows. Probes: RankMe, sparsity, norm.

## The recoverability insight — READ THIS before touching §4.2/§4.3

This is the conceptual core and it is **easy to get backwards** (the old text did).

- We fit a small MLP projector (REPA's own form) up a **ladder of cheap per-token inputs**:
  constant baseline (≈ mean embedding) → token identity → identity + position. We record the
  cosine each reaches. Code: `encoder_profiling/proteina/_probes/lib.py:analyze_projector_saturation`
  (and tabasco `chemeleon/investigate.py` section 5).
- It is a **lower bound**: the real trunk feeds its projector all of this **plus** noisy
  coordinates, cross-token attention, and the timestep — so it can only do better.
- **The ladder withholds geometry.** So whatever the cheap rungs recover is structure available
  *without* geometry, and the geometric content a 3D generator needs is exactly what they
  **cannot** reach.
- **Therefore gap magnitude does NOT track quality** (the inversion):
  - **Large cheap-lift = wrong signal.** The embedding is mostly a lookup on identity —
    conformation-invariant, already available to the trunk. (ESM2: gap **+0.053**, the largest,
    yet wrong-kind.)
  - **Small cheap-lift ≠ failure.** A genuinely geometric encoder sits near the constant floor,
    because identity can't recover geometry without coordinates. (GearNet: gap **+0.009**,
    nearly the smallest, yet the **best** target.)
  - **Saturation shows in the constant floor, not the lift.** High mean-direction cosine =
    embeddings barely vary = saturated. (MACE: floor **0.86**.) Low floor = spread (GearNet 0.43).
  - **We judge recoverability against information** — the floor + the lift only mean something
    once Q1's probes say whether the spread is geometric.

Numbers to keep straight: GearNet floor 0.43 / gap +0.009 (best, geometric); ESM2 floor 0.67 /
gap +0.053 (wrong-kind, identity); MACE floor 0.86 / gap +0.005 (saturated).

## THE NEXT JOB — §4.2/§4.3 zoo tables + prose (flagged, not yet done)

The §4.1 reframing has **not** propagated downstream. Concretely:

- **The zoo tables (`table_profiling_mol.tex`, `table_profiling_protein.tex`)** have their
  group header renamed to **Recoverability**, but the **`Gap` column still presents the old
  "bigger = better" reading.** As it stands, **GearNet's +0.009 reads as a failure** in the
  table — exactly backwards. Proposed fix: surface the **mean-direction floor** as its own
  column next to the gap, so *saturation* (MACE, high floor) and *triviality* (ESM2, big lift)
  read as distinct failures, and *small-gap-but-geometric* (GearNet) reads as good.
- **The §4.2/§4.3 prose verdicts** still lean on the old framing in places (e.g. MACE's
  "saturated" is right, but the gap is still discussed as headroom). Align them with the
  recoverability story: lift vs floor vs information.
- Decide whether to add a **floor/saturation** number to the tables (it's the real
  saturation signal; currently only the gap is shown).

The numbers are in `encoder_profiling/{tabasco,proteina}/FINDINGS.md` (headline tables). No new
compute needed — this is a writing + table-restructuring pass.

## Other things done this session (context, already committed)

- **Ch4 intro** (`a76a602`): imaging→molecular bridge — content/geometry *coupling* in images
  vs *decoupling* in molecules → representation is multi-factorial → "the encoder we align to
  routes REPA toward whichever factor it encodes." Calls back to §2.2.1 (Singh/Yu) and §2.4.2
  (geometry vs identity). Safe framing: coupling/decoupling, never "images are single-factor".
- **RankMe fixed**: described as **effective dimensionality** (Roy–Vetterli effective rank),
  NOT "a stand-in for ImageNet linear-probe accuracy" — that was wrong and would collapse the
  Information and Conditioning axes together.
- **Prediction framing softened** across Ch4: profiling **guides** the studies rather than being
  a test "checked back against" them (matches the intro's NotA — "consistent with the lens, not
  a controlled experiment"). §4.4 retitled **"What the profiles imply."**
- **CheMeleon RankMe re-run** after an env fix (see gotchas): **1166** (GEOM) / 1195 (QM9),
  now same metric as MACE's 40.6. The high rank is a *diffuse-and-sparse* bottleneck vs the
  128-d projector, not collapse — opposite failure mode to MC-GearNet.
- **CheMeleon-vs-MACE figure** (`figures/fig_ch4_mol_encoders.{py,png}`): sparsity histogram +
  conformer-invariance. MACE cross-conformer cosine reported as **0.995** (40-mol measure).

## Gotchas (carried + new)

- **rdkit env drift (NEW, important).** A rogue `rdkit-pypi 2022.9.5` shadows the locked
  `rdkit 2025.9.3` in `.venv`, so the LMDBs won't depickle ("ENDMOL tag not found"). Fix is
  targeted (`uv pip uninstall rdkit-pypi; uv pip install --reinstall rdkit==2025.9.3`; then pin
  `numpy==1.26.4 pillow==12.0.0` back) — **do NOT `uv sync`** (churns the PyG/jax stack).
  Memory: `reference_rdkit_pypi_shadow`.
- **Never commit `figures/fig_ch4_mol_encoders.pdf`** — png-only convention; a broad `git add`
  catches it (it did this session). Stage explicit paths.
- **`figures/.fig_ch4_cache.npz`** is gitignored (regenerable embedding cache for the figure).
- **`\multicolumn` group glosses in Table 4.1 don't wrap** — keep them ≤ ~65 chars or they
  overflow the right margin (the Recoverability gloss did on the first render; trimmed).
- **Pre-commit hooks** (trailing-whitespace, ruff-format) modify files and abort the first
  commit → re-stage and re-commit.
- **`report-draft.tex` is usually open in the IDE** — `Read` immediately before each `Edit`.
- **`make pdf` runs from `docs/masters-report/`**; bash cwd resets across some calls — `cd` first.

## Key paths

- §4.1 ≈ `report-draft.tex` l.1091 (`sec:profiling-framework`); §4.2 `sec:profiling-mol`,
  §4.3 `sec:profiling-protein`, §4.4 `sec:profiling-predictions`.
- Ch4 assets: `figures/fig_ch4_framework.tex` (schematic), `figures/fig_ch4_mol_encoders.{py,png}`,
  `tables/table_profiling_{diagnostics,mol,protein}.tex`.
- Profiling data + numbers: `encoder_profiling/{tabasco,proteina}/FINDINGS.md`; probe code
  `encoder_profiling/proteina/_probes/lib.py`, `encoder_profiling/tabasco/chemeleon/investigate.py`.
- Word count: `make wordcount`.

## Open decisions

- **Name "Recoverability"** adopted (over "Reachability") — confirm or revisit.
- **Floor column** in the zoo tables — add it (recommended) or keep gap-only?
- One slightly loose sentence in §4.1: *"Saturation shows … in that floor — how spread the
  embeddings are at all."* The floor measures *concentration* (high = saturated), so "how
  spread" is imprecise; tighten if desired.
