"""SS fraction vs sampler γ at the latest checkpoint of each model.

Direct evidence panel for the claim that γ controls helix-vs-sheet balance.
Reads the same two jsonls as plot_sampler_ablation:
  - results/variance/n256_sampler_ablation/sweep_results.jsonl   (γ ∈ {ODE, 0.0, 0.35, 0.5, 1.0}, seed 42)
  - results/paper/n256_convergence_pdb/sweep_results.clean.jsonl (γ=0.45, 3 reps)

Pins to the latest available ckpt for each model:
  - baseline_256_bs24_2gpu_step1500k
  - repa_l9_256_per_residue_bs24_2gpu_step900k

x-axis: γ on a linear scale, with ODE plotted at the rightmost slot as a
separate category (deterministic, "γ=0 with no Langevin" so visually distinct).
y-axes (3 panels):
  1. Helix fraction (designable)            ↑ when γ is small  (helix-dominant)
  2. Sheet fraction (designable)            ↑ when γ grows     (sheet-emerging)
  3. H/E ratio (log y, designable)          1.0 = balanced
Plus an annotation row showing designable-N at each γ so noise from small
subsets is visible (γ=1.0 is the danger zone for baseline).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
ABLATION = ROOT / "results/variance/n256_sampler_ablation/sweep_results.clean.jsonl"
CONVERGENCE = ROOT / "results/paper/n256_convergence_pdb/sweep_results.clean.jsonl"
FIG_OUT = ROOT / "figures/paper/n256_sampler_ablation/pdb"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# (run-id, ckpt step, model label, color, marker) — report-wide convention.
MODELS = [
    ("baseline_256_bs24_2gpu_step1500k", 1500000, "Baseline @ 1.5M", "tab:blue", "s"),
    (
        "repa_l9_256_per_residue_bs24_2gpu_step900k",
        900000,
        "REPA L9 @ 900K",
        "tab:green",
        "o",
    ),
]

# Sampler grid on the x-axis. ODE plotted at -0.1 (left of γ=0) so it's
# visually adjacent to the γ=0 SDE config (both have "no injected noise"
# but differ in score-correction). Tick label highlights this.
GAMMAS = [
    ("ODE", -0.1, "ODE"),
    ("0.0", 0.00, "0.0"),
    ("0.35", 0.35, "0.35"),
    ("0.45", 0.45, "0.45\n(paper)"),
    ("0.5", 0.50, "0.5"),
    ("1.0", 1.00, "1.0"),
]


def _humanize(v, _pos=None):
    av = abs(v)
    if av >= 1e6:
        return f"{v / 1e6:g}M"
    if av >= 1e3:
        return f"{v / 1e3:g}K"
    if av >= 1:
        return f"{v:g}"
    return f"{v:.3g}"


def _g_key(r):
    if r.get("sampling_mode") == "vf":
        return "ODE"
    if r.get("sampling_mode") == "sc":
        v = r.get("sc_scale_noise")
        if v is None:
            return None
        return f"{float(v)}"
    return None


def load_all_rows():
    rows = []
    for p in (ABLATION, CONVERGENCE):
        if not p.exists():
            continue
        for line in p.open():
            rows.append(json.loads(line))
    return rows


def aggregate(rows, run_id):
    """Return {γ_key: [(H, E, n_des), ...]} for one (run, ckpt)."""
    out = defaultdict(list)
    for r in rows:
        if r.get("run") != run_id:
            continue
        g = _g_key(r)
        if g is None:
            continue
        H = r.get("_res_ss_frac_H_designable")
        E = r.get("_res_ss_frac_E_designable")
        n = r.get("_res_ss_n_designable") or r.get("_res_designability_n")
        if H is None or E is None:
            continue
        out[g].append((float(H), float(E), int(n or 0)))
    return out


def main():
    rows = load_all_rows()

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(7.2, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 3, 1.2]},
    )

    panels = [
        ("Helix fraction (designable)", lambda H, E: H, False),
        ("Sheet fraction (designable)", lambda H, E: E, False),
        ("H/E ratio (designable)", lambda H, E: (H / E) if E > 0 else None, True),
    ]

    xticks = [x for _, x, _ in GAMMAS]
    xtick_labels = [lbl for _, _, lbl in GAMMAS]

    # Top 3 panels: H, E, H/E
    for i, (title, fn, log_y) in enumerate(panels):
        ax = axes[i]
        for run_id, _step, model_label, color, marker in MODELS:
            agg = aggregate(rows, run_id)
            xs, means, mins, maxs = [], [], [], []
            for k, x, _ in GAMMAS:
                vals = [fn(H, E) for H, E, _n in agg.get(k, []) if fn(H, E) is not None]
                if not vals:
                    continue
                xs.append(x)
                means.append(sum(vals) / len(vals))
                mins.append(min(vals))
                maxs.append(max(vals))
            if not xs:
                continue
            # Connect with faint line (γ is continuous; ODE is annotated apart).
            ax.plot(xs, means, color=color, linewidth=1.3, alpha=0.5)
            # Min/max envelope where >1 rep (only γ=0.45 has multi-rep).
            has_var = any(mn != mx for mn, mx in zip(mins, maxs))
            if has_var:
                ax.fill_between(xs, mins, maxs, color=color, alpha=0.2, linewidth=0)
            ax.plot(
                xs,
                means,
                marker=marker,
                markersize=8,
                markeredgewidth=0.8,
                markeredgecolor="white",
                linestyle="none",
                color=color,
                label=model_label,
            )
        ax.set_ylabel(title, fontsize=10)
        if log_y:
            ax.set_yscale("log")
            ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.5)
            ax.text(
                1.01,
                1.0,
                " H=E (balanced)",
                va="center",
                fontsize=8,
                color="black",
                alpha=0.6,
                transform=ax.get_yaxis_transform(),
            )
        ax.grid(True, axis="y", alpha=0.3, which="both" if log_y else "major")
        ax.yaxis.set_major_formatter(FuncFormatter(_humanize))
        # Vertical guide at γ=0.45 to flag the paper operating point.
        ax.axvline(0.45, color="grey", linestyle=":", linewidth=1, alpha=0.4)

    # Bottom panel: designable-N annotation (caveat for SS noise at low des).
    ax = axes[3]
    for run_id, _step, model_label, color, marker in MODELS:
        agg = aggregate(rows, run_id)
        for k, x, _ in GAMMAS:
            ns = [n for _H, _E, n in agg.get(k, [])]
            if not ns:
                continue
            n_mean = sum(ns) / len(ns)
            ax.plot(
                x,
                n_mean,
                marker=marker,
                markersize=8,
                markeredgewidth=0.8,
                markeredgecolor="white",
                linestyle="none",
                color=color,
            )
            # Annotate if small (<30) — those H/E points are noisy.
            if n_mean < 30:
                ax.annotate(
                    f"n={int(n_mean)}",
                    (x, n_mean),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=color,
                )
    ax.set_ylabel("# designable / 250", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 300)
    ax.axhline(30, color="black", linestyle=":", linewidth=1, alpha=0.4)
    ax.text(
        1.01,
        30,
        " noisy below",
        va="center",
        fontsize=7,
        color="black",
        alpha=0.5,
        transform=ax.get_yaxis_transform(),
    )
    ax.axvline(0.45, color="grey", linestyle=":", linewidth=1, alpha=0.4)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.yaxis.set_major_formatter(FuncFormatter(_humanize))

    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels(xtick_labels, fontsize=9)
    axes[-1].set_xlabel(
        "Sampler  (ODE = deterministic;  γ = SDE Langevin scale)", fontsize=10
    )

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(MODELS),
        fontsize=10,
        frameon=False,
    )
    fig.suptitle(
        "Secondary-structure balance vs sampler γ (latest ckpt per model, n=256)\n"
        "Designable subset. γ=0.45 shows 3-rep mean+min/max; others are seed 42.",
        fontsize=11,
        y=1.02,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_OUT / "ss_vs_gamma.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
