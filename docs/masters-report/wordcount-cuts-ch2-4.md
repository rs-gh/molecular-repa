# Word-count cuts: exposition chapters (Ch 2–4)

Target: ~150 words. This is a **menu**, not a plan — the candidates below total
~220 words, ranked by confidence/low-risk-first. Picking the top 4 (A–D) gets
us to ~157 words with essentially zero loss of content. Nothing is edited
inline yet.

Word counts are prose-word counts (math/macros stripped), so they're
approximate but consistent across before/after.

---

## Tier 1 — high confidence (figure captions that re-state the prose)

The user flagged captions specifically. Both of these captions sit directly
above/below prose that already says the same thing in full, so trimming them
loses nothing for a reader who reads the section, and a skimmer still gets a
self-contained gloss.

### A. REPA-construction caption — `report-draft.tex:779` · **save ~38 words** (95 → 57)

The §2.2 prose (lines 783–805) walks through every element of this figure:
the interpolant, the generator path, the frozen encoder, the projector, the
loss, and the combined objective. The caption currently re-derives all of it,
including the `x_t = (1-t)x_0 + t x_1` construction that the prose and the
`L_FM` equation already give.

**Current:**
> The REPA construction. A noised sample $x_t = (1-t)x_0 + t\,x_1$ is
> constructed at each training step from a noise sample $x_0$, a clean data
> sample $x_1$, and a sampled timestep $t \in [0,1]$. The generator $v_\theta$
> processes $x_t$ through its sequence of layers $\ell_1, \ldots, \ell_L$ to
> produce the velocity field that feeds the flow-matching loss
> $\mathcal{L}_{\text{FM}}$. Separately, the clean sample $x_1$ is also passed
> through a frozen pretrained encoder $f_{\text{enc}}$ (shaded). The
> generator's hidden state $z_\ell$ at the chosen alignment layer is mapped
> into the encoder's feature space by a trainable projector $h_\phi$; the REPA
> loss $\mathcal{L}_{\text{REPA}}$ is the negated similarity between
> $h_\phi(z_\ell)$ and $f_{\text{enc}}(x_1)$, averaged across tokens. The total
> training loss $\mathcal{L}$ combines them with weight $\lambda$.

**Proposed:**
> The REPA construction. The generator $v_\theta$ processes the noised sample
> $x_t$ to produce the velocity field for the flow-matching loss
> $\mathcal{L}_{\text{FM}}$. In parallel, a frozen encoder $f_{\text{enc}}$
> (shaded) embeds the clean sample $x_1$. A trainable projector $h_\phi$ maps
> the generator's hidden state $z_\ell$ into the encoder's space, and the REPA
> loss $\mathcal{L}_{\text{REPA}}$ is their token-averaged negated similarity.
> The total loss combines the two with weight $\lambda$.

### B. Design-cycle caption — `report-draft.tex:861` · **save ~19 words** (60 → 41)

The §2.3 prose (lines 865–867) states the upstream→generate→filter→assay→
feedback cycle in full. The "thousands to tens of thousands" pool size is
already given inside the figure box (the `~10^3+` node) and in the §2.3 prose.

**Current:**
> The de-novo molecular design cycle. An upstream specification drives a
> generative model to produce a candidate pool, which is funnelled through
> in-silico filters and then synthesised (or expressed) and assayed in the wet
> lab. Experimental results feed back into the next round of generation.
> Typical generation-step pool sizes are in the thousands to tens of thousands
> of candidates per round~\cite{RFdiffusion}.

**Proposed:**
> The de-novo molecular design cycle. An upstream specification drives a
> generative model to produce a candidate pool, which is funnelled through
> in-silico filters and then synthesised (or expressed) and assayed in the wet
> lab. Results feed back into the next round.

> Note: this drops the `\cite{RFdiffusion}` pool-size citation from the
> caption. The same claim + cite can stay in the §2.3 prose if we want to keep
> the reference; check it isn't the only occurrence first.

---

## Tier 2 — high confidence (trivia / asides, no load-bearing content)

### C. Titin / peptide aside — `report-draft.tex:902` · **save ~15 words**

Pure colour. The residue-range and the `\cite{BritannicaProtein}` carry the
point; titin and the peptide parenthetical are never referenced again.

**Current:**
> A \emph{protein} is a chain of amino-acid residues, ranging from around 50
> to several thousands in length~\cite{BritannicaProtein}; the largest known,
> titin, has approximately 34,000 residues. (Shorter chains are typically
> classified as \emph{peptides} instead.)

**Proposed:**
> A \emph{protein} is a chain of amino-acid residues, ranging from around 50
> to several thousands in length~\cite{BritannicaProtein}.

> (If you like the peptide distinction, keep just `(Shorter chains are
> typically classified as peptides.)` and drop only titin — saves ~9 instead.)

---

## Tier 3 — medium confidence (real content, but trimmable)

These touch prose, not just captions — lower priority. Use only if A–C don't
get us far enough, or to build in slack.

### D. Tabasco/Proteina baseline lists — `report-draft.tex:878` · **save ~25 words**

The closing sentence is a generic "minimalism still wins" claim. The two
baseline triplets here (EQGAT-diff/MiDi/SemlaFlow and RFdiffusion/FrameFlow/
Genie) **also appear** in §2.3.1 (line 891) and §2.3.2 (line 908), so the
citations are not lost if this paragraph leans out. Tighten the wrap-up:

**Current (last sentence):**
> Despite their architectural minimalism, both achieve performance comparable
> or superior to equivariant baselines on quality evaluations.

**Proposed:** fold into the prior sentence or cut — the "minimalism scales"
point is already made in the §2.1/§2.3 framing and restated in the intro.
(~15 words if only the trailing sentence goes; ~25 if the lineage clauses are
also compressed.)

> ⚠️ Slightly higher risk: this paragraph is the only place the architectural
> *lineage* (AF3 pair-bias → Proteina) is stated. Trim the editorialising
> sentence, not the lineage.

### E. Ch3 opener trade-off preview — `report-draft.tex:1088` · **save ~20 words**

The closing clause previews the "push the envelope" idea that §3.7 (line 1167)
develops in full ("pushes the envelope itself … makes a model canonically
better"). The preview can be shortened to a pointer.

**Current (last sentence):**
> A gain on one axis may imply a loss on another (Chapter~\ref{ch:proteina-study}),
> so finding an acceptable balance, or better yet, pushing this envelope, is
> very useful.

**Proposed:**
> A gain on one axis may imply a loss on another
> (Chapter~\ref{ch:proteina-study}), a tension we return to in
> \S\ref{sec:protein-tradeoffs}.

### F. Ch4 encoder-lens examples — `report-draft.tex:1216` · **save ~20 words**

The three-example list (interatomic-potential / fold-classifier / sequence-
model) is re-instantiated concretely in §4.2–§4.3 with the actual encoders
(MACE, GearNet, ESM2). The abstract preview can shrink.

**Current:**
> An encoder is typically optimised for one task in isolation, and sees a
> molecule through that lens. For example, an interatomic-potential model reads
> local geometry; a fold classifier reads global shape; a sequence model reads
> chemical identity. The encoder we align to therefore selects which factor
> REPA is most likely to transfer.

**Proposed:**
> An encoder is typically optimised for one task in isolation, and sees a
> molecule through that lens — local geometry, global shape, or chemical
> identity. The encoder we align to therefore selects which factor REPA is
> most likely to transfer.

---

## Recommendation

| | Cut | Save | Risk |
|---|---|---|---|
| A | REPA caption trim | ~38 | none |
| B | design-cycle caption trim | ~19 | none (verify cite) |
| C | titin/peptide aside | ~15 | none |
| D | Tabasco/Proteina wrap-up sentence | ~15–25 | low |
| E | Ch3 opener preview | ~20 | low |
| F | Ch4 encoder-lens examples | ~20 | low |

**A + B + C + D ≈ 90–100 words.** Add **E + F** to land at ~130–140, or take
a touch more from D's lineage clauses to clear 150 comfortably.

My pick for the cleanest 150: **A + B + C + E + F** (≈ 112) plus **D** at the
fuller ~25-word trim → **~137**, then one of the smaller asides if we want a
buffer. All of these are either caption redundancy or previews of content that
pays off later — none remove a fact or a citation that doesn't survive
elsewhere.
