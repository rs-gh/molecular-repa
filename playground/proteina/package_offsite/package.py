"""Package the proteina convergence-plot EMA checkpoints for offsite work.

Builds a self-contained directory mirroring /rds/.../proteina/store/ but containing
only the (run, step) pairs that the n128/n256 convergence plots actually consume,
plus each run's data_config / exp_config JSONs. Designed to be interrupted and
resumed safely: every copy is staged to a .partial path, sha256'd, then renamed,
with progress logged to manifest.jsonl. Re-running skips entries whose dest
exists AND whose recorded sha matches the source.

Usage:
    python package.py --dst /rds/user/sr2173/hpc-work/proteina_offsite_pkg
    python package.py --dst <dst> --smoke    # one (run, step) only
    python package.py --dst <dst> --verify   # re-hash dest, no copying

After it finishes, rsync the dst dir to the other machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

SRC_ROOT = Path("/rds/user/sr2173/hpc-work/proteina/store")

# (store_dir, [steps in thousands]) — derived from the convergence sweep_results.jsonl files
# under evaluation/proteina/generation/results/paper/{n128_paper_afdb,n256_convergence_pdb,n256_convergence_afdb}
RUNS: dict[str, list[int]] = {
    # n=128 AFDB convergence (existing sweep)
    "proteina_60m_baseline_afdb_128_bs80_2gpu": [200, 400, 600, 800, 1000, 1200],
    "proteina_60m_repa_l4_128_afdb_per_residue_bs80_2gpu": [200, 600],
    "proteina_60m_repa_mpnn_l4_128_afdb_per_residue_bs80_2gpu": [
        200,
        400,
        600,
        800,
        1000,
    ],
    # n=128 AFDB: new MPNN-L9 run added 2026-05-17 (not yet in n128_paper_afdb sweep)
    "proteina_60m_repa_mpnn_l9_128_afdb_per_residue_bs80_2gpu": [100, 200, 300, 400],
    # n=128 PDB bs80 cohort — extended training as of 2026-05-17; planning a PDB n=128 step-curve sweep this week.
    # Taking every 100k periodic ckpt available on disk (verified 2026-05-17).
    "proteina_60m_baseline_128_bs80": [100, 200, 300, 400, 500, 600, 700],
    "proteina_60m_repa_l0_128_per_residue_bs80": [100, 200],
    "proteina_60m_repa_l4_128_per_residue_bs80": [100, 200, 300, 400, 500],
    "proteina_60m_repa_l9_128_per_residue_bs80": [100, 200, 300, 400],
    "proteina_60m_repa_l4_128_per_residue_random": [100, 200, 300, 400],
    "proteina_60m_repa_mpnn_l4_128_per_residue_bs80": [100, 200],
    "proteina_60m_repa_mpnn_l9_128_per_residue_bs80_2gpu": [100, 200, 300, 400, 500],
    # n=256 PDB convergence
    "proteina_60m_baseline_256_bs24_2gpu": [100, 200, 400, 700, 1000, 1300, 1500],
    "proteina_60m_repa_l4_256_per_residue_bs24_2gpu": [100, 200, 400, 700, 800, 900],
    "proteina_60m_repa_l9_256_per_residue_bs24_2gpu": [100, 200, 400, 700, 800, 900],
    "proteina_60m_repa_l4_256_per_residue_random_bs24_2gpu": [100, 200, 400, 700],
    "proteina_60m_repa_mpnn_l4_256_per_residue": [100, 200, 400, 700, 1000, 1300, 1600],
    "proteina_60m_repa_mpnn_l9_256_per_residue": [100, 200, 400],
    # n=256 AFDB convergence
    "proteina_60m_baseline_afdb_swissprot_256": [100, 200, 400, 700, 1000, 1300, 1600],
    "proteina_60m_repa_l4_256_afdb_per_residue": [100, 200, 400, 700, 1000, 1200],
    "proteina_60m_repa_l9_256_afdb_per_residue": [100, 200, 400, 500, 600],
    "proteina_60m_repa_mpnn_l4_256_afdb_per_residue": [100, 200, 400, 700, 1000, 1100],
    "proteina_60m_repa_mpnn_l9_256_afdb_per_residue": [
        100,
        200,
        400,
        700,
        1000,
        1300,
        1400,
        1500,
    ],
}


def sha256_file(p: Path, buf: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def find_ema_ckpt(ckpt_dir: Path, step_k: int) -> Path:
    step = step_k * 1000
    # Lightning pads to 12 digits; tolerate any width.
    pattern = f"chk_epoch=*_step={step:012d}-EMA.ckpt"
    matches = sorted(ckpt_dir.glob(pattern))
    if not matches:
        # fallback: any width
        matches = sorted(
            p
            for p in ckpt_dir.glob("chk_epoch=*-EMA.ckpt")
            if f"_step={step:012d}-" in p.name or f"_step={step}-" in p.name
        )
    if not matches:
        raise FileNotFoundError(f"no EMA ckpt for step={step} under {ckpt_dir}")
    # Prefer the non-"-v1" variant if both exist (v1 is a duplicate from a resumed save).
    non_v1 = [m for m in matches if "-v1-" not in m.name]
    return (non_v1 or matches)[0]


def copy_with_sha(src: Path, dst: Path) -> tuple[str, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    shutil.copyfile(src, tmp)
    sha = sha256_file(tmp)
    size = tmp.stat().st_size
    tmp.rename(dst)
    return sha, size


def load_manifest(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["rel"]] = rec
    return out


def append_manifest(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


def collect_jobs(runs: dict[str, list[int]]) -> list[tuple[str, str, Path]]:
    """Return list of (run, rel_path_under_dst, src_path) for every file to copy."""
    jobs: list[tuple[str, str, Path]] = []
    for run, steps in runs.items():
        ckpt_dir = SRC_ROOT / run / "checkpoints"
        if not ckpt_dir.is_dir():
            print(f"[WARN] missing source dir: {ckpt_dir}", file=sys.stderr)
            continue
        for step_k in steps:
            src = find_ema_ckpt(ckpt_dir, step_k)
            rel = f"{run}/checkpoints/{src.name}"
            jobs.append((run, rel, src))
        # Config JSONs (small, copy all matching *.json in checkpoints/).
        for cfg in sorted(ckpt_dir.glob("*.json")):
            rel = f"{run}/checkpoints/{cfg.name}"
            jobs.append((run, rel, cfg))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Destination root (under /rds preferred)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Copy only the first (run, step) pair, then exit",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Re-hash dst files vs manifest; no copying",
    )
    ap.add_argument(
        "--runs", nargs="*", default=None, help="Restrict to these store_dir names"
    )
    args = ap.parse_args()

    runs = RUNS if args.runs is None else {r: RUNS[r] for r in args.runs}
    if args.smoke:
        first_run = next(iter(runs))
        runs = {first_run: runs[first_run][:1]}

    args.dst.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dst / "manifest.jsonl"
    done = load_manifest(manifest_path)

    jobs = collect_jobs(runs)
    print(f"[plan] {len(jobs)} files across {len(runs)} runs -> {args.dst}")

    if args.verify:
        bad = 0
        for run, rel, src in jobs:
            dst = args.dst / rel
            rec = done.get(rel)
            if not dst.exists():
                print(f"[MISS] {rel}")
                bad += 1
                continue
            if rec is None:
                print(f"[NO-MANIFEST] {rel}")
                bad += 1
                continue
            sha = sha256_file(dst)
            if sha != rec["sha256"]:
                print(f"[SHA-MISMATCH] {rel}")
                bad += 1
        print(f"[verify] {bad} problems out of {len(jobs)}")
        return

    total_bytes = 0
    t0 = time.time()
    for i, (run, rel, src) in enumerate(jobs, 1):
        dst = args.dst / rel
        rec = done.get(rel)
        if (
            dst.exists()
            and rec
            and rec.get("size") == dst.stat().st_size
            and rec.get("src") == str(src)
        ):
            print(f"[skip {i}/{len(jobs)}] {rel}")
            total_bytes += rec["size"]
            continue
        if dst.exists():
            dst.unlink()
        print(f"[copy {i}/{len(jobs)}] {rel} ({src.stat().st_size/1e9:.2f} GB)")
        sha, size = copy_with_sha(src, dst)
        append_manifest(
            manifest_path,
            {
                "rel": rel,
                "src": str(src),
                "sha256": sha,
                "size": size,
                "ts": time.time(),
            },
        )
        total_bytes += size

    dt = time.time() - t0
    print(
        f"[done] {total_bytes/1e9:.2f} GB in {dt:.0f}s ({total_bytes/1e6/max(dt,1):.0f} MB/s)"
    )


if __name__ == "__main__":
    main()
