"""Cross-encoder collation.

Reads each encoder's latest results.json (and gearnet_random's mean_std.json),
emits comparison.csv and a small set of cross-encoder bar charts under
encoder_profiling/proteina/figures/.

Usage:
  python encoder_profiling/proteina/collate.py
  python encoder_profiling/proteina/collate.py --encoders gearnet,esm,...

Layout assumed:
  encoder_profiling/proteina/<enc>/results/<timestamp>/results.json
  encoder_profiling/proteina/gearnet_random/results/<timestamp>/mean_std.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_ENCODERS = [
    "gearnet",
    "gearnet_random",
    "esm",
    "mc_gearnet",
    "pw_gearnet",
]


def _latest_run(encoder_dir: Path):
    results_root = encoder_dir / "results"
    if not results_root.is_dir():
        return None
    runs = sorted([p for p in results_root.iterdir() if p.is_dir()])
    return runs[-1] if runs else None


def _flatten(d, prefix=""):
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[key] = float(v)
    return flat


def load_encoder_metrics(name: str, root: Path):
    """Return (display_name, flat_metrics_dict) or None."""
    enc_dir = root / name
    run = _latest_run(enc_dir)
    if run is None:
        print(f"  [skip] {name}: no results dir under {enc_dir}/results")
        return None

    if name == "gearnet_random":
        path = run / "mean_std.json"
        if not path.exists():
            print(f"  [skip] {name}: missing {path}")
            return None
        data = json.loads(path.read_text())
        flat = {k: v["mean"] for k, v in data["aggregate"].items()}
        flat_std = {f"{k}.std": v["std"] for k, v in data["aggregate"].items()}
        flat.update(flat_std)
        return data.get("name", name), flat

    # Standard single-run encoder
    path = run / "results.json"
    if not path.exists():
        # PW has multiple variants under the same dir; pick all
        return None
    data = json.loads(path.read_text())
    return data.get("name", name), _flatten(data)


HEADLINE_KEYS = [
    ("embed_dim", "Embed dim"),
    ("dimensionality.rankme", "RankMe"),
    ("dimensionality.participation_ratio", "PR"),
    ("dimensionality.dims_for_95pct_var", "Dims@95% var"),
    ("distribution.exact_zero_frac", "Exact-zero frac"),
    ("perturbation.sigma_1.0A.mean", "Pert@1A cos"),
    ("rotation.rotation_cos_mean", "Rotation cos"),
    ("residue_probe.linear_probe_acc", "AA probe acc"),
    ("projector.mean_direction_cos", "Mean-dir baseline"),
    ("projector.conditions.random.test_cos", "Projector cap (random)"),
    ("projector.conditions.onehot.test_cos", "Projector (onehot)"),
    ("norms.norm_mean", "Mean L2 norm"),
    ("norms.dead_dims", "Dead dims"),
    ("protein_similarity.delta", "Within-vs-between Δ"),
]


def write_csv(rows: dict, out_path: Path):
    keys = [k for k, _ in HEADLINE_KEYS]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encoder", *[label for _, label in HEADLINE_KEYS]])
        for name, flat in rows.items():
            w.writerow([name, *[flat.get(k, "") for k in keys]])
    print(f"Wrote {out_path}")


def bar_chart(rows: dict, key: str, label: str, out_path: Path):
    names = list(rows.keys())
    vals = [rows[n].get(key) for n in names]
    pairs = [(n, v) for n, v in zip(names, vals) if v is not None]
    if not pairs:
        return
    names, vals = zip(*pairs)
    fig, ax = plt.subplots(figsize=(max(5, len(names) * 0.9), 4))
    bars = ax.bar(names, vals)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title(label)
    ax.set_ylabel(label)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--encoders",
        type=str,
        default=",".join(DEFAULT_ENCODERS),
        help="Comma-separated subdir names under encoder_profiling/proteina/.",
    )
    args = ap.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    rows = {}
    for name in encoders:
        loaded = load_encoder_metrics(name, HERE)
        if loaded is None:
            continue
        display, flat = loaded
        rows[display] = flat

    if not rows:
        print("No encoder results found. Run the explore_*.py scripts first.")
        return

    write_csv(rows, HERE / "comparison.csv")

    fig_dir = HERE / "figures"
    fig_dir.mkdir(exist_ok=True)
    for key, label in HEADLINE_KEYS:
        bar_chart(rows, key, label, fig_dir / f"compare_{key.replace('.', '_')}.png")

    # Projector-saturation gap = REPA-condition (best of onehot/onehot+pos) - mean-dir
    print("\nProjector saturation gap (best condition - mean-dir baseline):")
    for n, flat in rows.items():
        mean_cos = flat.get("projector.mean_direction_cos")
        cands = [
            flat.get("projector.conditions.onehot.test_cos"),
            flat.get("projector.conditions.onehot+pos.test_cos"),
        ]
        cands = [c for c in cands if c is not None]
        best = max(cands) if cands else None
        if mean_cos is not None and best is not None:
            print(
                f"  {n:30s}  best={best:.4f}  mean-dir={mean_cos:.4f}  gap={best - mean_cos:+.4f}"
            )


if __name__ == "__main__":
    main()
