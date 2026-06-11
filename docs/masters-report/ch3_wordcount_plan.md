# Ch3 word-budget plan (2026-06-06)

**Goal:** land under 15,000 words with a strong ~800–1,200 w Ch7 conclusion still
to write. Current state: **15,220 counted / 13,584 body**, Ch7 = 0 words.
We need to free **~1,000–1,400 words**.

This pass: (1) move Ch6→Ch3 where it belongs, (2) trim Ch3 hard, prefer tables.
**Do NOT touch Ch6 results prose yet** (only the Metrics/protocol subsections are
in scope, and only for *moving* shared concepts — not cutting results).

---

## Current per-chapter body counts

| Ch | Title | Body words |
|----|-------|-----------:|
| 1 | Introduction | 1,170 |
| 2 | Background | 2,313 |
| **3** | **Evaluation (new)** | **2,842** |
| 4 | Profiling | 1,517 |
| 5 | Tabasco | 1,544 |
| 6 | Proteina | 4,198 |
| 7 | Conclusions | 0 |

## Ch3 section breakdown (the trim surface)

| § | Section | Words | Trim target |
|---|---------|------:|------------:|
| 3.1 | What "good" means | 286 | → ~200 (−86) |
| 3.2 | Multi-factorial (imaging vs molecular) | 379 | KEEP (RT-Ch3-1, load-bearing) — light touch only, −30 |
| 3.3 | Small-molecule metrics | 372 | → table + ~180 prose (−190) |
| 3.4 | Protein-backbone metrics | 596 | → table + ~300 prose (−250) |
| 3.5 | PDB vs AFDB | 217 | → ~150 (−67) |
| 3.6 | Probe principle | 407 | → ~280 (−127) |
| 3.7 | Reading metrics together | 256 | → ~180 (−76) |
| 3.8 | Closing question | 206 | KEEP (Ch7 spine setup) — light, −30 |
| | **Total** | **2,719** | **target ~1,850 (−870)** |

---

## Part A — What can move from Ch6 → Ch3 (net word effect)

Ch6 has two relevant subsections:
- **§6.x Metrics** (lines 1501–1523, **505 w**)
- **§6.x Evaluation protocol** (lines 1524–1567, **486 w**)

Classify each Ch6 Metrics paragraph:

| Ch6 ¶ | Content | Verdict |
|-------|---------|---------|
| ¶1 "three families" framing + Table S4 ref | **DUPLICATED** by new Ch3 §3.6 intro. | Ch6 can drop to 1 sentence; concept lives in Ch3. |
| ¶2–3 probe battery (IF / dihedral / CATH) | Protein-**student-specific instrument**. Per the flow doc's "open placement question", recommendation was *inline in Ch6 §6.2*. | **STAYS in Ch6** — it is the instrument, not the principle. Ch3 owns the *principle* (§3.6). |
| ¶4 CKNNA definition | Ch3 §3.6 now introduces CKNNA conceptually. | Ch6 keeps the *protocol* (k=10, bootstrap) but can drop the conceptual gloss → saves ~30 w in Ch6, already covered in Ch3. |
| ¶5 "five groups of Chapter 3" | Already defers to Ch3. | Good as-is. |
| ¶6 fidelity-vs-diversity axis | **CONCEPTUAL** — belongs in Ch3 §3.4/§3.7. | **MOVE to Ch3** (it's a cross-cutting reading axis, not Ch6-specific). |
| ¶7 ssJSD-2D introduction | Ch6's **own new metric**. | STAYS in Ch6 (with the appendix ref). |

**Key realisation:** Table S4 (`table_setup_metrics`, 46 lines, the full 5-group
spec) is currently `\input` in Ch6. It is *exactly* the taxonomy Ch3 needs as a
"reader primitive." **Moving Table S4's home to Ch3** lets Ch3 §3.4 replace ~250 w
of prose-defining-each-metric with a table + ~300 w of *narrative* (what the groups
mean, the T-W/T-D tension, the designability caveat). Ch6 then `\input`s nothing
there and just back-references `Table~\ref{tab:setup-metrics}` (the `\label` is
global, so the ref still resolves from Ch6).

### Net effect of Part A
- **Ch3 gains a table** (Table S4) → lets §3.3 and §3.4 shed ~440 w of prose that
  was hand-defining metrics the table already lists.
- **Ch6 loses ~80–120 w** (the duplicated "three families" framing, the CKNNA
  conceptual gloss, the fidelity/diversity axis paragraph that moves to Ch3).
- Table S4 itself is **caption/tabular = NOT counted** by texcount (it's in the
  `159` tabular-words bucket already excluded). So moving it is **word-free** but
  buys prose cuts on both sides.

---

## Part B — Ch3 prose trims (independent of the move)

General lever (user pref): **short single-idea sentences**; cut throat-clearing
and hedges; let tables carry definitions.

- **§3.1 (−86):** drop the second-paragraph restatement of "axes in tension"
  (§3.2 makes the point structurally). Keep the filter-proxy reframe (load-bearing).
- **§3.3 (−190):** replace the three labelled prose blocks (validity/connectivity/
  uniqueness; diversity/novelty; FCD) with a compact **small-molecule metric table**
  (new, ~12 lines, word-free) + a short narrative on saturation. Saturation point
  is the only thing Ch5 needs from here.
- **§3.4 (−250):** lean on **Table S4** (moved in from Ch6). Keep the *narrative*:
  why five groups, the T-W vs T-D tension, the designability-proxy/AFDB-lineage
  caveat. Cut the per-metric definitional sentences the table now carries.
- **§3.5 (−67):** tighten PDB/AFDB to the two-regime contrast + the two caveats;
  drop the restated scarcity motivation (already in Ch1).
- **§3.6 (−127):** keep the probe principle + the RT-Ch3-2 "principled not arbitrary"
  flag (both load-bearing). Trim the CKNNA paragraph (now also defined in Ch6 protocol)
  and the "two places we use it" elaboration.
- **§3.7 (−76):** tighten frontier + statistical-hygiene to one crisp para each.
- **§3.2, §3.8:** protect; only sentence-level tightening (−30 each).

**Projected Ch3 after A+B: ~1,850 w** (from 2,842) → **frees ~990 w**, plus Ch6
sheds ~80–120 w. Combined headroom ≈ **1,070–1,110 w** → comfortably fits an
~900–1,000 w Ch7 under the 15k cap.

---

## New tables to add (all word-free under texcount rules)

1. **`table_eval_smallmol_metrics.tex`** (NEW) — small-molecule suite: metric |
   definition | what it guards | direction. ~6 rows. Lets §3.3 shrink to narrative.
2. **`table_setup_metrics.tex`** (EXISTING) — **re-home to Ch3 §3.4**; Ch6 keeps
   the `\ref`. Already the 5-group spec.

## Order of operations (proposed)
1. Move Table S4 `\input` from Ch6 → Ch3 §3.4; verify `\ref` still resolves in Ch6.
2. Strip the 3 movable bits from Ch6 Metrics subsection (framing, CKNNA gloss,
   fidelity/diversity ¶ → relocate the last into Ch3 §3.4/§3.7).
3. Build the new small-molecule metric table; rewrite §3.3 around it.
4. Trim §3.1/3.4/3.5/3.6/3.7 per Part B.
5. `make wordcount` + `make pdf`; confirm <15k with ~1k spare for Ch7.
6. THEN draft Ch7.

## Guardrails
- Do not cut Ch6 *results* prose. Only the Metrics/protocol subsections, only to
  move shared concepts.
- Protect §3.2 (RT-Ch3-1) and §3.8 (Ch7 spine). Light touch only.
- Keep metric names + the 5-group taxonomy stable with Ch6 (Table S4 is the anchor).
- Tabular/caption text is NOT counted — push definitions into tables freely.
