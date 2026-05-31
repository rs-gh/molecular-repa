# Report — appendix additions & cleanup tracker

Running list of things to add to the appendix and other deferred cleanup for
`report-draft.tex`. Add items as we defer them; tick/remove when done.

## Appendix content to write / expand

- [ ] **Encoder & injection-depth sweep** — full results for the encoder/depth
  combinations beyond the L4/L9 + GearNet/MPNN subset shown in Ch6. Referenced
  from the model-config table caption ("Other ablations are deferred to
  Appendix A") and the Models prose.
- [ ] **Optimisation ablations** — λ, batch size, learning rate sweeps.
  Referenced from the merged model/hparams table caption and the Models prose.
- [ ] **Projector-depth ablation** — referenced at the training-dynamics
  callback (`Appendix~\ref{ch:appendix}`, ~Ch6 scale section).
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
