"""Generate tables/table_speedup.tex --- two coloured %-delta columns separating
the two questions, with absolute scores alongside each delta:

  Acceleration: REPA vs the baseline at matched compute (400K steps, the anchor of
      Fig 6.4). Robust to the non-monotone AFDB baseline (compares values at a
      fixed step, not "best so far").
  Long run: REPA's own best vs the baseline's best, with the checkpoint at which
      REPA's best occurs. The baseline is capped at the furthest trained REPA
      variant in each regime (PDB 2.0M, AFDB 1.5M), so the comparison never reads a
      baseline checkpoint with no variant alongside.

Each cell shows a signed % delta (coloured green if REPA is better, red if worse,
via the preamble macros \\gd / \\rd) followed by the absolute score. Training extent
is shown in parentheses after each variant name (it differs by dataset --- AFDB and
PDB are separate models), so the undertrained random control is visible inline and
no separate column is needed. Capping the baseline at the furthest trained variant
per regime drops its out-of-bounds tail (PDB 2.0M--2.4M, AFDB 1.5M--1.8M) without
losing any headline number.

PDB FPSD (dagger): REPA leads at 400K but its ceiling sits below the baseline's late
best --- accelerates, does not win. AFDB designability (ddagger): saturated proxy,
random scores highest, so its green long-run deltas do not reflect REPA.

Run from repo root:  python docs/masters-report/tables/scripts/make_speedup_table.py
"""

import json
import re
import collections
from statistics import mean

ROOT = "/home/sr2173/git/molecular-repa"
GENDIR = f"{ROOT}/evaluation/proteina/generation/results/paper"
OUT = f"{ROOT}/docs/masters-report/tables/table_speedup.tex"
CAP = 2_000_000
EARLY = 400_000


def fam_step(run):
    m = re.match(r"(.*)_step(\d+)k$", run)
    return (m.group(1), int(m.group(2)) * 1000) if m else (run, None)


def label(fam):
    if "random" in fam:
        return "Random control"
    if fam.startswith("baseline"):
        return "baseline"
    if "mpnn_l9" in fam:
        return "L9-MPNN"
    if "mpnn_l4" in fam:
        return "L4-MPNN"
    if fam in ("repa_l9", "repa_l9_afdb"):
        return "L9-GearNet"
    if fam in ("repa_l4", "repa_l4_afdb"):
        return "L4-GearNet"
    return None


def norm(fam):
    return (
        fam.replace("_bs24_2gpu", "")
        .replace("_per_residue", "")
        .replace("_256", "")
        .replace("_afdb", "")
    )


def traj(d, metric):
    rows = [
        json.loads(line)
        for line in open(f"{GENDIR}/{d}/sweep_results.jsonl")
        if line.strip()
    ]
    agg = collections.defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        f, s = fam_step(r["run"])
        if s is not None:
            agg[(label(norm(f)), s)].append(v)
    out = collections.defaultdict(dict)
    for (lab, s), vs in agg.items():
        if lab:
            out[lab][s] = mean(vs)
    return out


def fmt_step(s):
    return f"{s / 1e6:.1f}M" if s >= 1_000_000 else f"{s // 1000}K"


def colour(txt, good):
    if good is None:
        return txt
    return f"\\gd{{{txt}}}" if good else f"\\rd{{{txt}}}"


# Random control listed first (right after the baseline anchor row), matching the
# convention in the other result tables (6.2, 6.6, ODE, appendix rep/ranges).
VARIANT_ORDER = ["Random control", "L4-GearNet", "L9-GearNet", "L4-MPNN", "L9-MPNN"]


def cell(d, metric, lower, vfmt):
    t = traj(d, metric)
    last_true = {lab: max(t[lab]) for lab in t}
    b400 = t["baseline"].get(EARLY)
    # Per-regime cap: truncate the baseline (and random control) to the furthest
    # step of any genuine trained REPA variant in this regime, so the long-run
    # column never reads a baseline checkpoint with no variant alongside.
    # PDB -> 2.0M (L9-MPNN); AFDB -> 1.5M (L9-MPNN).
    genuine = [lab for lab in t if lab not in ("baseline", "Random control")]
    cap = min(CAP, max(max(t[lab]) for lab in genuine))
    tc = {lab: {s: v for s, v in d2.items() if s <= cap} for lab, d2 in t.items()}
    bb = (min if lower else max)(tc["baseline"].values())
    bb_step = (min if lower else max)(tc["baseline"], key=lambda s: tc["baseline"][s])

    def pct(target, v):
        return (target - v) / target * 100 if lower else (v - target) / target * 100

    rows = []
    for lab in VARIANT_ORDER:
        if lab not in tc:
            continue
        name = f"{lab} ({fmt_step(last_true[lab])})"
        v400 = t[lab].get(EARLY)
        if v400 is None:
            early = ("n/a", None)
        else:
            pe = pct(b400, v400)
            early = (f"${pe:+.0f}\\%$ ({vfmt(v400)})", pe > 0)
        d2 = tc[lab]
        own = (min if lower else max)(d2.values())
        ostep = (min if lower else max)(d2, key=lambda s: d2[s])
        # Long-run column is absolute-only: the signed %-vs-baseline-best was
        # read against a *different* denominator than the acceleration column
        # (baseline-best vs baseline@400K), so a single row could show +X% next
        # to -Y% and look like an arithmetic error. We keep the win/lose colour
        # (better/worse than the bolded baseline-best row) but drop the number.
        better = (own < bb) if lower else (own > bb)
        late = (f"{vfmt(own)} (@{fmt_step(ostep)})", better)
        rows.append((name, early, late))
    return b400, bb, bb_step, min(last_true["baseline"], cap), rows


blocks = [
    (
        "PDB FPSD $\\downarrow$",
        "n256_convergence_pdb",
        "_res_PDB_FID",
        True,
        lambda v: f"{v:.0f}",
        "",
    ),
    (
        "PDB designability $\\uparrow$",
        "n256_convergence_pdb",
        "_res_designability_rate",
        False,
        lambda v: f"{v:.2f}",
        "",
    ),
    (
        "AFDB FPSD $\\downarrow$",
        "n256_convergence_afdb",
        "_res_AFDB_FID",
        True,
        lambda v: f"{v:.0f}",
        "",
    ),
    (
        "AFDB designability $\\uparrow$",
        "n256_convergence_afdb",
        "_res_designability_rate",
        False,
        lambda v: f"{v:.2f}",
        "",
    ),
]

lines = [
    "% Auto-generated by tables/scripts/make_speedup_table.py --- do not hand-edit.",
    "\\begin{table}[tbp]",
    "\\centering",
    "\\small",
    "\\begin{tabular}{l c c}",
    "\\toprule",
    "\\textbf{Variant} (last ckpt) & \\textbf{Accel.} @400K (vs base) "
    "& \\textbf{Long run} (abs.\\ best) \\\\",
]
for title, d, metric, lower, vfmt, mark in blocks:
    b400, bb, bb_step, base_last, rows = cell(d, metric, lower, vfmt)
    bb_s = f"{bb:.0f}" if lower else f"{bb:.2f}"
    b4_s = f"{b400:.0f}" if lower else f"{b400:.2f}"
    lines.append("\\midrule")
    lines.append("\\multicolumn{3}{l}{" + f"\\textit{{{title}}}{mark}" + "} \\\\")
    # Baseline anchor row (plain, not bold): gives both columns a visible
    # reference without adding weight across the four blocks. The acceleration
    # column's deltas divide by this @400K value; the long-run column's
    # absolutes are compared (by colour) against this best value.
    lines.append(
        f"baseline (to {fmt_step(base_last)}) & {b4_s} (@400K) "
        f"& {bb_s} (@{fmt_step(bb_step)}) \\\\"
    )
    for name, (etxt, eg), (ltxt, lg) in rows:
        lines.append(f"{name} & {colour(etxt, eg)} & {colour(ltxt, lg)} \\\\")

lines += [
    "\\bottomrule",
    "\\end{tabular}",
    "\\caption{\\textbf{Most REPA variants accelerate generation quality over the baseline.} In some "
    "regimes they also win the long run. Acceleration is the \\%-delta vs the baseline at "
    "400K (the anchor of Figure~\\ref{fig:proteina-genrep}), with the absolute score in "
    "parentheses; the long run is each variant's absolute best, with the baseline capped "
    "at the furthest trained variant per regime (2.0M PDB, 1.5M AFDB), coloured against the "
    "baseline-best row. ($n{=}1$--$3$ seeds, seed-mean; "
    "$1{,}125$ backbones/seed, $250$ for designability.)}",
    "\\label{tab:proteina-speedup}",
    "\\end{table}",
]

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved {OUT}")
print("\n".join(lines[7:-8]))
