# Redundancy savings scope (post-Ch3) — 2026-06-06

Now that Ch3 owns the conceptual ground (multi-factorial framing, metric
definitions in tables, probe principle, PDB/AFDB regimes, designability proxy),
several downstream passages restate it. This is the scope of what we *could*
trim. **No report changes made** — estimates only.

Current word count: **14,457 / 15,000**.

Convention below: "compress to a callback" = replace the full restatement with
1 short sentence that cross-refs Ch3 (`Chapter~\ref{ch:evaluation}`).

---

## TIER 1 — clear duplications of Ch3 (safe, recommended)

| # | Location | What it is | Words | After | Save |
|---|----------|-----------|------:|------:|-----:|
| 1 | **Ch4 intro ¶2** (l.1207) | Imaging-couples / molecules-decouple / "no single encoder" — **near-verbatim Ch3 §3.2** | 152 | ~45 | **~105** |
| 2 | **Ch6 datasets ¶3** (l.1583) "Different datasets carry different biases" | designability proxy + ProteinMPNN/ESMFold/AF2 lineage — **Ch3 §3.4 now states this** | 97 | ~30 | **~65** |
| 3 | **Ch6 datasets ¶1** (l.1579) | "two regimes spanning experimental–synthetic" — **Ch3 §3.5 owns the regime framing** | 65 | ~45 | **~20** |
| | | | | **Tier 1 total** | **~190** |

Notes:
- **#1 is the one you flagged.** Ch4's job is to *apply* the framing to encoders,
  not re-derive it. Keep the encoder-specific punchline ("the encoder we align to
  selects which factor REPA transfers" + the interatomic-potential/fold-classifier/
  sequence-model examples, since those are Ch4-specific), drop the pixel-grid /
  coupling/decoupling exposition → callback to §3.2.
- **#2** keep the *empirical* hook ("training on AFDB raises the designability
  floor at every setting, see §6.x") but drop the re-explanation of *why* (the
  lineage mechanism) since §3.4 gives it. One sentence + `\ref`.
- **#3** the dataset *facts* (Swiss-Prot subset, chain counts, n≤256) MUST stay —
  those are Ch6-specific setup. Only the "experimental vs synthetic spectrum"
  motivational clause duplicates §3.5; trim that clause, keep the facts. Small win.

---

## TIER 2 — partial overlap, defensible to keep (your call)

| # | Location | What it is | Words | Possible save | Risk |
|---|----------|-----------|------:|--------------:|------|
| 4 | **Ch6 Metrics ¶ rep-quality + list** (l.1602–1609) | Defines IF / dihedral / CATH probes (the *student battery*) | 192 | ~60 if pushed to a table | Med — this is the *instrument*, point-of-use is defensible (flow-doc recommended inline) |
| 5 | **Ch5 Metrics rep-quality ¶** (l.1335) | Defines linear-probe protocol + RDKit descriptors | 134 | ~40 if descriptors → table | Med — protocol detail, point-of-use defensible |
| 6 | **Ch2 §2.2.1 encoder-characterisation ¶** (l.813) | Yu vs Singh global/local debate | ~140 | 0 recommended | This is *background*, legitimately Ch2's; Ch3/Ch4 cite it. Leave. |
| 7 | **Ch4 intro ¶1** (l.1205) Yu/Singh recap | 53 | ~20 | Low | Mild echo of l.813; could compress but tiny |

Notes:
- **#4 and #5 are the "reduce metric discussion in Ch5/Ch6" you asked about.**
  BUT: last pass we already established the agreed split — *definitions* live in
  Ch3 tables, *protocol* (sample counts, probe-fit details, descriptor lists)
  stays at point-of-use. These two paragraphs are mostly protocol, not
  re-definition. The genuine duplication ("following the probe principle of
  Ch3", "linearly decodable") is already a 1-line callback. **Real remaining
  savings here are modest (~100 combined) and cost some readability** — the
  reader currently learns the probe setup where the numbers appear.
  - Option to recover more: move the RDKit-descriptor list (#5) and the
    IF/dihedral/CATH probe-task list (#4) into a small shared "representation
    probes" table. That's word-free (tabular) and would let both paragraphs
    shrink to narrative. ~100 w saved, but adds a table.

---

## TIER 3 — NOT redundant (leave alone)

These *reference* the multi-factorial theme but are doing new work (they're the
*payoff*, not the setup). Cutting them would weaken the spine.

- l.1666 "Every encoder routes representation gain differently" — the Ch6 finding
  that *pays off* §3.2/§3.4. Keep.
- l.1770 "These results point past Proteina" — Ch6→Ch7 bridge. Keep.
- l.1768 "REPA helps most where the baseline struggles" — data-efficiency payoff. Keep.
- l.1695 representation-bottleneck coupling — Ch6 empirical result. Keep.
- l.2308 AFDB genrep appendix — appendix, not counted heavily. Keep.

---

## Bottom line

| Bucket | Realistic save |
|--------|---------------:|
| Tier 1 (clear dups) | **~190 w** |
| Tier 2 (#4+#5 via shared probe table) | ~100 w (adds a table) |
| **Total achievable** | **~190–290 w** |

Combined with the current **543 w of headroom**, doing **Tier 1 alone** gives
**~730 w of room** for Ch7 — comfortable for a strong ~900–1000 w conclusion if
we also keep the ~540 we already have... actually 543 + 190 = **~733 w free**,
which is a healthy conclusion budget without touching Tier 2.

**Recommendation:** do **Tier 1** (low-risk, ~190 w, pure dedup of things Ch3 now
owns). Hold Tier 2 unless Ch7 comes in long — the protocol paragraphs earn their
place at point-of-use, and the shared-probe-table refactor is more disruptive than
its ~100 w is worth right now.
