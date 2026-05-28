"""Build the supercolumn-grouped sampler-ablation table for PDB & AFDB.

Emits two artefacts alongside the plots:
  - sampler_ablation_table.md  (HTML tables with colspan/rowspan grouping)
  - sampler_ablation_table.csv (flat columns; super-group encoded in prefixes)

Pulls from the .clean.jsonl files (post-dedup) so γ=0.45 reflects multi-rep means.
Three rows per table: step-matched baseline, REPA, and latest-baseline ckpt.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures/paper/n256_sampler_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSONLS = [
    ROOT / "results/variance/n256_sampler_ablation/sweep_results.clean.jsonl",
    ROOT / "results/paper/n256_convergence_pdb/sweep_results.clean.jsonl",
    ROOT / "results/variance/n256_afdb_sampler_ablation/sweep_results.clean.jsonl",
    ROOT / "results/paper/n256_convergence_afdb/sweep_results.clean.jsonl",
]


def load_rows():
    rows = []
    for f in JSONLS:
        if not f.exists():
            continue
        for ln in f.open():
            rows.append(json.loads(ln))
    return rows


def sampler_tag(r):
    if r.get("sampling_mode") == "vf":
        return "ODE"
    g = r.get("sc_scale_noise")
    return f"γ={g:g}" if g is not None else None


def grab(rows, run, step, sampler):
    acc = defaultdict(list)
    for r in rows:
        if r.get("run") != run or r.get("step") != step:
            continue
        if sampler_tag(r) != sampler:
            continue
        for k, v in r.items():
            if isinstance(v, (int, float)) and k.startswith("_res_"):
                acc[k].append(v)
    n_reps = len(next(iter(acc.values()), [])) if acc else 0
    return {k: sum(v) / len(v) for k, v in acc.items()}, n_reps


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, int):
        return str(v)
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def pct(v):
    return f"{v * 100:.1f}" if v is not None else "—"


SAMPLERS = ["ODE", "γ=0", "γ=0.35", "γ=0.45", "γ=0.5", "γ=1"]


# (super_group, sub_group, header, key_extractor, formatter)
COLS = [
    ("Sample quality", "", "Des% ↑", "_res_designability_rate", pct),
    ("Sample quality", "", "scRMSD ↓", "_res_scRMSD_mean", fmt),
    ("Sample quality", "", "pLDDT ↑", "_res_plddt_mean", fmt),
    ("Tertiary structure", "Whole distribution", "FID-PDB ↓", "_res_PDB_FID", fmt),
    ("Tertiary structure", "Whole distribution", "FID-AFDB ↓", "_res_AFDB_FID", fmt),
    ("Tertiary structure", "Whole distribution", "fJSD-A ↓", "_res_PDB_fJSD_A", fmt),
    ("Tertiary structure", "Whole distribution", "fS-A ↑", "_res_fS_A", fmt),
    (
        "Tertiary structure",
        "Designable subset",
        "#Clust ↑",
        "_res_diversity_clusters_total",
        lambda v: f"{int(v)}" if v is not None else "—",
    ),
    (
        "Tertiary structure",
        "Designable subset",
        "pwTM ↓",
        "_res_diversity_pairwise_tm_mean",
        fmt,
    ),
    (
        "Tertiary structure",
        "Designable subset",
        "Nov-PDB% ↑",
        "_res_novelty_foldseek_pdb_rate",
        pct,
    ),
    (
        "Tertiary structure",
        "Designable subset",
        "Nov-AFDB% ↑",
        "_res_novelty_foldseek_afdb_swissprot_rate",
        pct,
    ),
    ("Secondary structure", "Whole", "fJSD-C ↓", "_res_PDB_fJSD_C", fmt),
    ("Secondary structure", "Whole", "fS-C ↑", "_res_fS_C", fmt),
    (
        "Secondary structure",
        "Designable",
        "ss-JSD-2D ↓",
        "_res_ss_jsd_pdb_designable_2d",
        fmt,
    ),
    ("Secondary structure", "Designable", "α/β %", None, None),  # computed below
]


def alpha_beta(m):
    a = m.get("_res_ss_frac_H_designable")
    b = m.get("_res_ss_frac_E_designable")
    return f"{a*100:.0f} / {b*100:.0f}" if (a is not None and b is not None) else "—"


TABLES = {
    "PDB-trained — sampler ablation": [
        (
            "baseline_256_bs24_2gpu_step1000k",
            1000000,
            "Baseline @ 1.0M",
            "step-matched",
        ),
        ("repa_l9_256_per_residue_bs24_2gpu_step900k", 900000, "REPA L9 @ 900K", ""),
        (
            "baseline_256_bs24_2gpu_step1500k",
            1500000,
            "Baseline @ 1.5M",
            "latest, +500K",
        ),
    ],
    "AFDB-trained — sampler ablation": [
        ("baseline_afdb_256_step700k", 700000, "Baseline @ 700K", "step-matched"),
        ("repa_l4_afdb_256_step700k", 700000, "REPA L4 @ 700K", ""),
        ("baseline_afdb_256_step1600k", 1600000, "Baseline @ 1.6M", "latest, +900K"),
    ],
}


def supergroup_spans():
    """Return (groups, sub_groups) — each a list of (label, colspan)."""
    groups, subs = [], []
    cur_g, cur_s, g_n, s_n = None, None, 0, 0
    for sg, sub, *_ in COLS:
        if sg != cur_g:
            if cur_g is not None:
                groups.append((cur_g, g_n))
            cur_g, g_n = sg, 1
        else:
            g_n += 1
        if sub != cur_s:
            if cur_s is not None:
                subs.append((cur_s, s_n))
            cur_s, s_n = sub, 1
        else:
            s_n += 1
    groups.append((cur_g, g_n))
    subs.append((cur_s, s_n))
    return groups, subs


def _group_prefix(sg, sub):
    """Abbreviated supergroup tag prepended to each metric header."""
    if sg == "Sample quality":
        return "Q"
    if sg == "Tertiary structure":
        return "T-W" if "Whole" in sub else "T-D"
    if sg == "Secondary structure":
        return "S-W" if sub == "Whole" else "S-D"
    return "?"


def render_md_plain(title, ckpts, rows):
    """Plaintext markdown table. Group encoded as a prefix on each header."""
    headers = ["Model", "Sampler"] + [
        f"`{_group_prefix(c[0], c[1])}` {c[2]}" for c in COLS
    ]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "---|" * len(headers))
    for run, step, model_label, note in ckpts:
        for i, s in enumerate(SAMPLERS):
            m, n_reps = grab(rows, run, step, s)
            row = []
            if i == 0:
                tag = f" *({note})*" if note else ""
                row.append(f"**{model_label}**{tag}")
            else:
                row.append("")
            rep_note = f" (×{n_reps})" if n_reps > 1 else ""
            row.append(f"{s}{rep_note}")
            if not m:
                row.extend(["_no data_"] * len(COLS))
            else:
                for col in COLS:
                    sg, sub, hdr, key, formatter = col
                    if hdr == "α/β %":
                        row.append(alpha_beta(m))
                    else:
                        v = m.get(key)
                        row.append(formatter(v))
            out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def render_html_table(title, ckpts, rows):
    groups, subs = supergroup_spans()
    h = ["<table>", "<thead>"]
    g_cells = "".join(f'<th colspan="{n}">{lbl}</th>' for lbl, n in groups)
    h.append(
        f'<tr><th rowspan="3">Model</th><th rowspan="3">Sampler</th>{g_cells}</tr>'
    )
    s_cells = ""
    for lbl, n in subs:
        if lbl:
            s_cells += f'<th colspan="{n}">— {lbl} —</th>'
        else:
            s_cells += f'<th colspan="{n}"></th>'
    h.append(f"<tr>{s_cells}</tr>")
    metric_cells = "".join(f"<th>{c[2]}</th>" for c in COLS)
    h.append(f"<tr>{metric_cells}</tr>")
    h.append("</thead>")
    h.append("<tbody>")
    for run, step, model_label, note in ckpts:
        for i, s in enumerate(SAMPLERS):
            m, n_reps = grab(rows, run, step, s)
            row_cells = []
            if i == 0:
                tag = f"<br><i>({note})</i>" if note else ""
                row_cells.append(
                    f'<td rowspan="{len(SAMPLERS)}"><b>{model_label}</b>{tag}</td>'
                )
            rep_note = f" (×{n_reps})" if n_reps > 1 else ""
            row_cells.append(f"<td>{s}{rep_note}</td>")
            if not m:
                row_cells.append(f'<td colspan="{len(COLS)}"><i>no data</i></td>')
            else:
                for col in COLS:
                    sg, sub, hdr, key, formatter = col
                    if hdr == "α/β %":
                        row_cells.append(f"<td>{alpha_beta(m)}</td>")
                    else:
                        v = m.get(key)
                        row_cells.append(f"<td>{formatter(v)}</td>")
            h.append("<tr>" + "".join(row_cells) + "</tr>")
    h.append("</tbody></table>")
    return "\n".join(h)


LEGEND = """**Column group prefixes** (encoded with backtick tag at start of each header):

- `Q` — Sample quality (per-sample, no subset distinction)
- `T-W` — Tertiary structure, whole-distribution
- `T-D` — Tertiary structure, designable subset
- `S-W` — Secondary structure, whole-distribution
- `S-D` — Secondary structure, designable subset

γ=0.45 (paper default) is multi-rep mean across the convergence sweep's seeds; other γ values are single seed 42 (×N annotation flags multi-rep cells).
"""


def render_markdown(rows):
    out = ["# Sampler ablation tables (n=256)\n"]
    out.append("Generated from `.clean.jsonl` ablation + convergence sweeps.\n")
    out.append(LEGEND)
    for title, ckpts in TABLES.items():
        out.append(f"\n## {title}\n")
        out.append(render_md_plain(title, ckpts, rows))
        out.append("")
    return "\n".join(out)


def render_csv(rows, path):
    """Flat CSV. Group encoded in column prefix: `tertiary_whole_FID-PDB ↓` etc."""
    header = ["dataset", "model", "training_step", "sampler", "n_reps"]
    for sg, sub, hdr, *_ in COLS:
        prefix = sg.split()[0].lower()
        if sub:
            prefix += "_" + sub.split()[0].lower()
        header.append(f"{prefix}__{hdr}")
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for title, ckpts in TABLES.items():
            dataset = "PDB" if "PDB" in title else "AFDB"
            for run, step, model_label, _note in ckpts:
                for s in SAMPLERS:
                    m, n_reps = grab(rows, run, step, s)
                    row = [dataset, model_label, step, s, n_reps]
                    if not m:
                        row.extend(["" for _ in COLS])
                    else:
                        for col in COLS:
                            sg, sub, hdr, key, formatter = col
                            if hdr == "α/β %":
                                row.append(alpha_beta(m))
                            else:
                                v = m.get(key)
                                row.append(formatter(v) if v is not None else "")
                    w.writerow(row)


def main():
    rows = load_rows()
    md = render_markdown(rows)
    md_path = OUT_DIR / "sampler_ablation_table.md"
    csv_path = OUT_DIR / "sampler_ablation_table.csv"
    md_path.write_text(md)
    render_csv(rows, csv_path)
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    print()
    print("--- markdown preview (first 60 lines) ---")
    for line in md.splitlines()[:60]:
        print(line)


if __name__ == "__main__":
    main()
