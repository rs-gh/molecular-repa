# Report — appendix additions & cleanup tracker

Running list of things to add to the appendix and other deferred cleanup for
`report-draft.tex`. Add items as we defer them; tick/remove when done.

## Appendix content to write / expand

- [ ] **Encoder & injection-depth sweep** — full results for the encoder/depth
  combinations beyond the L4/L9 + GearNet/MPNN subset shown in Ch6. Referenced
  from the model-config table caption ("Other ablations are deferred to
  Appendix A") and the Models prose.
  - [ ] **MUST include the depth-matched control:** trained encoder at layer~4
    vs random-weights GearNet at layer~4. The Models prose now asserts "a trained
    encoder at the same layer~4 still outperforms the random one, so the
    learned-versus-random gap is not an artefact of injection depth
    (Appendix~\ref{ch:appendix})" — that claim is unbacked until this lands.
- [ ] **Optimisation ablations** — λ, batch size, learning rate sweeps.
  Referenced from the merged model/hparams table caption and the Models prose.
- [ ] **Projector-depth ablation** — referenced at the training-dynamics
  callback (`Appendix~\ref{ch:appendix}`, ~Ch6 scale section).
- [ ] **Full CKNNA model×encoder matrix** — the §alignment prose now asserts
  "every REPA variant we tested raises alignment to all three encoders … (full
  model×encoder matrix in Appendix)". Back it with a table: baseline + 4 learned
  variants (GearNet/MPNN × L4/L9) × 3 encoders (GearNet, MPNN, ESM2), peak
  per-residue CKNNA. Source: `evaluation/proteina/alignment/results/cknna_matrix_per_residue.jsonl`
  (verified 2026-05-31: every variant > baseline on all three; no random control
  in this matrix).
- [x] **Dataset composition & split overlap** — done (`app:datasets`, with
  `table_setup_datasets`).
- [x] **Choice of CATH fold-probe level** — done (`app:cath`, with
  `table_setup_cath`).
- [x] **ssJSD-2D metric derivation** — done (`app:ssjsd`); first draft written.
  - [ ] Add a citation for P-SEA / Biotite secondary-structure assignment
    (currently named without a `\cite`).
  - [ ] Optionally expand with the 3-bin centroid sanity-check variant.
- [x] **Leakage-controlled probe evaluation** — done (`app:leakage`); model vs
  probe leakage, blinded vs probe-clean splits, per-probe regime choice.
  Summarised from `docs/research/pdb_split_leakage_audit.md` (the full audit,
  decomposition tables, and per-probe lead-regime choices live there).
- [ ] Remove the `\section{Placeholder section}` stub once the appendix is real.

## Citations / references to finish

- [ ] **ProteinWorkshop** (`\bibitem{ProteinWorkshop}`, Jamasb et al. ICLR 2024)
  and **Ingraham2019** (`\bibitem{Ingraham2019}`, NeurIPS 2019) — author/title/
  venue only; add `\href{}` arXiv/DOI/OpenReview links to match house style.
- [ ] Confirm **CKNNA** is the alignment measure actually reported in the
  original REPA and BoltzREPA papers (vs CKA/another) before final.

## Representation-quality table & figure (Ch6 §rep)

- [ ] **Regenerate `table_rep_quality` from current data.** Committed numbers are
  a stale 2026-05-30 11:39 snapshot (the source CSVs grew at 17:05) and no longer
  reproduce. Caption claims "mean over steps $\ge$700K" but the numbers match a
  last-checkpoint snapshot, not that window. Intended methodology going forward:
  best-layer probe, mean over the **700K–1.2M** window common to all four families.
  - [ ] **Extend L4-random representation evals to 1.3M and 1.4M** — checkpoints
    exist (`proteina_60m_repa_l4_256_per_residue_random_bs24_2gpu`, steps to 1400k)
    but evals stop at 1200k; this is the only family capping the common window.
    Once in, widen the window to 700K–1.4M and regenerate.
  - [ ] **Add the CATH-C column** (currently IF, dihedral, CATH-A, CATH-T) so the
    table shows C/A/T for completeness; headline stays on CATH-A.
  - [ ] **Reconcile the Metrics-section prose** (§proteina-metrics): once C/A/T are
    all in the main table, drop "defer the full hierarchy to Appendix" and soften
    "Topology too finely-bucketed to probe reliably" → "noisiest level, but the
    direction is consistent across C→A→T".
  - [ ] Write the table from a small reproducible generator script (don't hand-maintain).
- [ ] **Run the representation sweep over multiple seeds** — the rep-quality and
  alignment results (table_rep_quality, fig02, fig03) are currently single-seed
  point estimates (`sweep_config.yaml`: `seeds: "42"`, `probe_seed 42`,
  `manifest_seed 42`). Re-run across several seeds so we can report variance /
  error bars (probe-fit seed at minimum; ideally also the manifest-sample seed),
  and state the spread rather than bare point values. Especially important given
  the small absolute CKNNA magnitudes — error bars would show whether the
  alignment pattern is robust to seed.
  - [ ] **AFDB random run (`repa_l4_afdb_256_random`) — prioritise multi-seed.**
    This is the random-encoder REPA falsifier control trained on AFDB; it is a
    headline control yet currently single-seed (seed 42). Re-run its rep/align
    evals over several probe-fit seeds so the control carries an error bar.
    Checkpoints on /rds: steps 100k/200k/400k/500k
    (`store/proteina_60m_repa_l4_256_afdb_per_residue_random_bs24_2gpu`). Appears
    in profiles `paper_n256_cath_if_dih_convergence_afdb_ext{3,4}` (→
    `results/paper/n256_convergence_cath_if_dih_afdb`, n_eval=4521) and
    `paper_n256_cath_if_dih_xclean_pdb_afdb_ext{3,4}` (→
    `results/paper/n256_xclean_pdb_afdb`, n_eval=62). **Caveat:** passing
    `--seeds` ≠ `42` flips `run_sweep.py` into the *rich* per-seed schema, which
    does not co-merge with the existing flat-schema rows in those dirs — write
    multi-seed output to a separate dir (or re-run the whole comparison set
    multi-seed), don't append into the flat dirs.
- [ ] **Figure/plot conventions to propagate** (introduced on fig02 this session):
  single shared legend below the panels, (a)/(b)/(c) panel-title labels, and
  "(thousands, log scale)" x-axis label. Apply to fig1c, fig3, fig4, fig5.

## Prose / structure cleanup

- [ ] **n=128 generation/representation protocol** — parked out of the Ch6
  Evaluation-protocol section (was: 500 backbones over 4 lengths {50,75,100,125}).
  Document it in the n=128 scale section (`sec:proteina-scale`) when that is
  written up.

- [ ] Cover-page word count is stale (says 8895; main chapters now ~8232).
  Regenerate via `make wordcount` on the real build env.
- [ ] Ch3 (Evaluation) is scaffold-only; several `\ref{ch:evaluation}`
  callbacks sharpen to section refs once it is written.
- [ ] Re-read "What we do not claim" against final results.

## Build / layout nits

- [ ] Appendix **datasets** and **CATH** tables have ~17pt / ~16pt right-margin
  overfull hboxes — tighten column widths.
- [ ] Orphaned `tables/table_setup_hparams.tex` — merged into
  `table_setup_model.tex`; safe to `git rm` (kept pending explicit OK).
- Build note: this machine lacks `upquote.sty`; drop a stub then `rm` it to
  compile locally (do NOT commit the stub). On the real env just `make pdf`.
