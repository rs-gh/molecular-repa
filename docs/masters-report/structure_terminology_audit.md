# "structure" terminology audit — `report-draft.tex`

Audit of every case-insensitive `structur` occurrence (structure / structural / structures / structured / structurally) in `docs/masters-report/report-draft.tex`. Pure-comment lines (first non-whitespace char `%`) are noted but not classified; all non-comment occurrences are classified below.

The author wants the word disambiguated. Senses tracked:

- **GEO** — 3D geometry (spatial arrangement of atoms); author prefers "geometry" here.
- **CON** — content / identity (atom or residue types, bonding graph, chemical identity).
- **GENQ** — generation-quality metric group ("structural quality" = validity/connectivity/uniqueness).
- **HIER** — protein structural hierarchy (primary / secondary / tertiary as a named tier).
- **INST** — a physical / determined structure instance ("PDB structures", "214M structures").
- **PRIOR** — structural priors (architectural inductive biases, SE(3)-equivariance etc.).
- **ENC** — "structure encoder" / "structure module" / "structure prediction" — a named model component or task (citation/title or proper-noun usage).
- **VAGUE** — loose filler or genuinely ambiguous referent. Flagged below.

## Summary table

| Sense | Count (non-comment) |
|---|---|
| GEO (3D geometry) | 6 |
| CON (content / identity) | 2 |
| GENQ (generation-quality metric group) | 8 |
| HIER (protein structural hierarchy) | 18 |
| INST (physical / determined structure instance) | 9 |
| PRIOR (structural priors / inductive bias) | 6 |
| ENC (named model component / task / title) | 18 |
| VAGUE (loose filler / ambiguous) | 12 |
| **Total non-comment classified** | **79** |
| Comment-only (noted, not classified) | ~31 |

(ENC includes 13 bibliography-title occurrences L1903–L2231, which are proper-noun and not actionable.)

---

## GEO — 3D geometry

- L877 (Ch2): "having to recover bond structure post hoc from interatomic distances" — *borderline CON; see flagged.*
- L1227 (Ch4): "predict energies from 3D structure" / "after a ... perturbation ... only a single structure" — geometry of a conformer.
- L1259 (Ch4): "pretrained as an interatomic potential to predict energies from 3D structure" — 3D geometry.
- L1272 (Ch4): "predicting sequence from structure" — 3D geometry (inverse-folding input).
- L1572 (Ch6): "the geometry of these structures is correspondingly more complex" — *INST + GEO mix; the "structures" = protein instances, but the sentence is about geometry.*
- L1766 (Ch6): "improve two things ... structural diversity" — *see GENQ / flagged; diversity of folds = geometry.*

## CON — content / identity

- L875 (Ch2): "its structure is typically represented using two coupled features: a discrete molecular graph ... and a continuous conformer" — here "structure" = the whole molecular representation (graph + geometry); leans CON-superset. *See flagged.*
- L877 (Ch2): "recover bond structure post hoc" — the bonding graph (content). Clear CON.

## GENQ — generation-quality metric group ("structural quality")

- L857 (Ch2): "checks for structural validity and physical plausibility" — the validity metric.
- L859 (Ch2): "metrics we adopt, such as structural validity and designability" — metric group.
- L866 (Ch2): "performance comparable or superior ... on structural-quality evaluations" — metric group.
- L1125 (Ch3): "per-molecule structural quality (validity, connectivity, uniqueness)" — definitional; the canonical use.
- L1125 (Ch3): "the structural-quality metrics are particularly amenable to saturation" — metric group.
- L1316 (Ch5): "On structural-quality metrics, the baseline is already at ceiling" — metric group.
- L1340 (Ch5): "validity, connectivity, and uniqueness (structural quality)" — metric group.
- L1613 (Ch6): tertiary/secondary structure metrics — *HIER, listed there; the metric grouping itself is taxonomy.*

## HIER — protein structural hierarchy (named tier)

- L892 (Ch2): "A protein's secondary structure is its local pattern of $\alpha$-helices..." — secondary tier.
- L892 (Ch2): "its tertiary structure is the global ... fold" — tertiary tier.
- L892 (Ch2): "Class captures broad secondary-structure content" — secondary tier.
- L894 (Ch2): "For the structure-generation problem we study" — *borderline; the backbone-geometry generation task. See flagged.*
- L1103 (Ch3): "secondary-structure elements to the tertiary fold" — hierarchy tiers.
- L1132 (Ch3): "from secondary-structure elements to the global fold" — tiers.
- L1140 (Ch3): "tertiary versus secondary structure" (supercolumn axis) — tiers.
- L1140 (Ch3): "Secondary-whole ... compare secondary-structure composition" — secondary tier.
- L1144 (Ch3): "global fold ... spanning local to global structure" — hierarchy span.
- L1613 (Ch6): "tertiary structure ... secondary structure over the same two (S-W, S-D)" — tiers.
- L1615 (Ch6): "the full (helix-fraction, strand-fraction) secondary-structure distribution" — secondary tier.
- L1689 (Ch6): "tertiary- and secondary-structure decomposition" — tiers.
- L1714 (Ch6): "secondary-structure fidelity (ssJSD-2D)" — secondary tier.
- L1722 (Ch6): "secondary- and tertiary-structure distributions" — tiers.
- L1750 (Ch6): "The secondary-structure shift is encoder×dataset conditional" — secondary tier.
- L2321 (App): "compares the mean helix and strand content" / "secondary-structure metric" — secondary tier.
- L2323 (App): "reproducing the shape of the secondary-structure distribution" — secondary tier.
- L2325/L2328 (App): "Secondary-structure composition of the designable subset" — secondary tier.

## INST — physical / determined structure instance

- L420 (Ch1): "synthetic structures from folding-model predictions" — structure instances.
- L894 (Ch2): "high-confidence synthetic structures from the AlphaFold Database" — instances.
- L1136 (Ch3): "scRMSD between generated and re-folded structures" — instances.
- L1138 (Ch3): "whose structures are themselves AF2 predictions" — instances.
- L1151 (Ch3): "hundreds of thousands of structures" — instances (PDB count).
- L1151 (Ch3): "hundreds of millions of AF2-predicted structures" — instances (AFDB count).
- L1221 (Ch4): "run frozen on a sample of structures from our training sets" — instances.
- L414 (Ch1): "trained on at most half a million structures" — instances.
- L1572 (Ch6): "the geometry of these structures" — instances (the proteins). *Also under GEO.*

## PRIOR — structural priors / architectural inductive bias

- L410 (Ch1): "data scale, data quality, and structural priors" — priors.
- L416 (Ch1): "encoding structural priors directly into model architectures" — priors.
- L416 (Ch1): "hard structural priors come with a computational cost" — priors.
- L422 (Ch1): "the field has traded hard structural priors for architectural simplicity" — priors.
- L864 (Ch2): "rather than relying on structural priors imposed by model architecture" — priors.
- L866 (Ch2): "setting it apart from equivariant ... generators" (context: structural priors) — *the sentence is about priors; the word "structure" here is in "protein-structure generation". See ENC.*

## ENC — named model component / task / title (proper-noun, mostly non-actionable)

- L418 (Ch1): "replacing the structure module with a diffusion model" — AlphaFold2 "structure module" (proper component name).
- L866 (Ch2): "motivated by the success ... in protein-structure generation" — task name.
- L894 (Ch2): "the structure-generation problem we study" — task name. *See flagged (could be "backbone-geometry generation").*
- L1596 (Ch6): "GearNet ... a structure encoder pretrained for CATH fold classification" — named encoder type.
- L1611 (Ch6): "the trunk versus a structural encoder" — named encoder type.
- L805/L806/L813/L1193/L1212 (Ch2/Ch4): "spatial structure of patch-token representations" — refers to the Singh et al. "spatial structure" hypothesis (a defined term). Treated as proper-noun term; see flagged for one borderline.
- Bibliography titles (proper nouns, not actionable): L1903, L1945, L1969, L2002, L2014, L2019, L2051, L2057, L2075, L2128, L2134, L2150, L2222, L2230, L2231.

---

## ⚠️ FLAGGED: vague / ambiguous uses

Grouped by chapter for act-on-it-by-chapter editing.

### Chapter 1 — Introduction

- **L422**: "this trade-off dilutes the **structural signal** available to a model."
  - *Why ambiguous:* "structural signal" has no precise referent. Does the synthetic-scale trade-off dilute geometry signal, content/identity signal, or the inductive-bias prior? The previous sentence is about priors, but "signal available to a model" reads like learnable signal in the data.
  - *Suggested:* "dilutes the **geometric signal** in the data" or "dilutes the **inductive-bias signal** the model can rely on" — pick whichever you mean.

- **L428**: "the training objective must now shoulder the additional responsibility of learning **structural symmetries**."
  - *Why ambiguous:* "structural symmetries" most likely means the SE(3) rotational/translational symmetries, but the word "structural" is doing nothing here; it could be misread as protein-hierarchy.
  - *Suggested:* "learning the **rotational and translational (SE(3)) symmetries**" or simply "the **geometric symmetries**".

- **L430**: "One response ... is to supply **representational structure** from elsewhere."
  - *Why ambiguous:* Pure filler. "Representational structure" = the auxiliary representation signal REPA injects; "structure" adds no information and collides with every other sense.
  - *Suggested:* "to supply a **representation target** from elsewhere" or "to supply **representational guidance** from elsewhere".

### Chapter 2 — Background and Related Work

- **L875**: "its **structure** is typically represented using two coupled features: a discrete molecular graph ... and a continuous conformer."
  - *Why ambiguous:* Here "structure" is the whole molecule (graph + geometry), but the report elsewhere reserves "geometry" for the conformer and "content/graph" for the bonding. Using "structure" as the umbrella term blurs the very GEO/CON split the report builds on.
  - *Suggested:* "a molecule is typically **represented** using two coupled features" (drop "its structure"), or "its **representation** is typically ...".

- **L894**: "For the **structure-generation problem** we study, it is customary to only model the backbone."
  - *Why ambiguous:* "structure-generation" reads as the generic task name, but the report's actual object is backbone *geometry* generation (CA-trace). Could be read as HIER or ENC.
  - *Suggested:* "For the **backbone-geometry generation problem** we study" (consistent with the GEO convention).

- **L866**: "Despite their architectural minimalism, both achieve performance comparable or superior to equivariant baselines on **structural-quality evaluations**."
  - *Why ambiguous:* This is in the protein+small-molecule joint sentence. "Structural-quality" is the small-molecule metric group (GENQ), but applied to both models it's unclear whether it means the GENQ group or generic "quality of generated structures".
  - *Suggested:* If GENQ is meant, say "on **generation-quality evaluations**"; "structural-quality" is a small-molecule-specific term and may mislead in a protein context.

### Chapter 3 — Evaluating molecular generation

- **L1103**: "Geometry is further complicated by **structural elements** at both local and global scales."
  - *Why ambiguous:* "structural elements" overlaps with HIER (secondary-structure *elements*), but here it's a general claim about geometry at multiple scales for molecules *and* proteins. Reads as filler glued onto "Geometry is further complicated by".
  - *Suggested:* "Geometry is further complicated by **organisation at both local and global scales**" or "**multi-scale geometric organisation**".

- **L1105**: "No single encoder captures geometry, content, and **every scale of structure** at once."
  - *Why ambiguous:* The triad "geometry, content, and ... structure" lists "structure" alongside geometry and content as if it were a *third* independent factor, yet structure (hierarchy) is itself geometric. This is the central conceptual term and it's used loosely against the report's own GEO/CON axis.
  - *Suggested:* "captures geometry, content, and **every scale of organisation** at once" or "geometry, content, and the **multi-scale hierarchy** at once".

- **L1158**: "averaging them into a composite discards exactly the **structure** we care about."
  - *Why ambiguous:* Pure filler — here "structure" means the per-axis detail / the Pareto frontier shape, nothing to do with molecules.
  - *Suggested:* "discards exactly the **per-axis detail** we care about" or "the **trade-off structure**" → "the **trade-offs** we care about".

- **L1165**: "measurable properties of the representation itself: how much **structure** it carries, on how many factors, and how well-conditioned it is."
  - *Why ambiguous:* Author-acknowledged vague filler. "How much structure it carries" means information content / spectral richness of the embedding, not molecular structure.
  - *Suggested:* "how much **information** it carries" or "how **rich** it is, on how many factors".

### Chapter 4 — Encoder targets and profiling

- **L1214**: "geometry, content, and **structure across scales** are pulled apart."
  - *Why ambiguous:* Same triad problem as L1105 — "structure across scales" listed as a third factor beside geometry and content.
  - *Suggested:* "geometry, content, and **multi-scale organisation** are pulled apart" (and make consistent with L1105).

- **L1246**: "Too little **structure** passes through for the alignment to transmit anything."
  - *Why ambiguous:* Filler. Here "structure" means variance / usable signal in the embedding (the sentence is about effective rank and sparsity).
  - *Suggested:* "Too little **signal** passes through" or "Too little **variance**".

### Chapter 6 — REPA on protein backbones

- **L1683**: "REPA may be nudging the trunk toward a common **structural representation** that several encoders approximate."
  - *Why ambiguous:* "structural representation" — does this mean a representation *of* protein structure, or just a shared representation (Platonic)? "structural" adds nothing precise.
  - *Suggested:* "toward a **common representation** that several encoders approximate" (drop "structural"), or "a **shared geometric representation**" if geometry is meant.

- **L1764**: "models learn to represent **structural characteristics** that the flow-matching loss never learns in isolation."
  - *Why ambiguous:* "structural characteristics" is vague — fold-level features? secondary-structure composition? local geometry? The whole headline rests on this and the referent is unspecified.
  - *Suggested:* "models learn to represent **fold- and secondary-structure features**" (or whichever the probe suite actually measures — be specific to the probe tasks).

---

## Recommended convention

1. Use **"geometry"** (never "structure") for the 3D spatial arrangement of atoms or C$_\alpha$ coordinates.
2. Use **"content"**, **"identity"**, or **"the bonding graph"** for discrete atom/residue types and connectivity.
3. Keep **"structural quality"** strictly as the named small-molecule metric group (validity/connectivity/uniqueness); prefer **"generation quality"** when speaking across molecules and proteins.
4. Reserve **"structure"** for the protein hierarchy only when a specific tier is named: **"secondary structure"**, **"tertiary structure / fold"** — and for concrete instances (**"PDB structures"**, **"AF2-predicted structures"**).
5. Use **"structural priors"** only for architectural inductive biases (SE(3)-equivariance); say **"priors"** or **"inductive biases"** if "structural" is doing no work.
6. Never use bare **"structure / structural"** as filler for *information / variance / signal / detail / organisation* — name the actual quantity (e.g. "the geometric signal", "the embedding's variance", "multi-scale organisation").
