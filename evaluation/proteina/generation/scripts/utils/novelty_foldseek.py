"""Foldseek-based novelty for proteina evaluation.

Replaces (or runs alongside) the centroid-file novelty path in evaluate.py.
For each generated PDB, runs `foldseek easy-search` against a prebuilt target
database, takes the max alntmscore across all hits, and reports rate / mean
max-TM. See docs/research/proteina_eval_protocol.md (TODO) for the full
protocol comparison vs the centroid pipeline.

Default knobs:
  --alignment-type 2        3Di+AA Gotoh + TM-align refinement on top hits.
                            Reported alntmscore is on the canonical TM-align
                            scale, but Foldseek skips TM-align on the long
                            tail of clearly-low survivors (~3-5 ms/pair vs
                            ~20 ms/pair for --alignment-type 1).
  --max-seqs 1000           Top-1000 prefilter survivors per query. For our
                            single-best-match question this is comfortably
                            above the saturation point on high-TM hits.
  -s 9.5                    Default max sensitivity.
  --tmscore-threshold 0.0   Report all hits; we filter by max post-hoc.

Internal-consistency note: lock these flag values across all sweep runs that
will be plotted together. Different alignment-type / max-seqs values produce
slightly different absolute novelty numbers and break per-checkpoint deltas.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from loguru import logger


def _ensure_foldseek_on_path(foldseek_bin: Optional[str]) -> str:
    """Return an absolute path to the foldseek binary or raise."""
    if foldseek_bin:
        if not os.path.isfile(foldseek_bin) or not os.access(foldseek_bin, os.X_OK):
            raise FileNotFoundError(f"foldseek binary not executable at {foldseek_bin}")
        return foldseek_bin
    found = shutil.which("foldseek")
    if found is None:
        raise FileNotFoundError(
            "foldseek not found on PATH. Either pass foldseek_bin=... or export "
            "PATH=/rds/user/sr2173/hpc-work/tools/foldseek/bin:$PATH before calling."
        )
    return found


def _foldseek_version(foldseek_bin: str) -> str:
    """Return the version commit hash so we can pin it in sweep result rows."""
    try:
        out = subprocess.check_output([foldseek_bin, "version"], text=True).strip()
        return out.splitlines()[0]
    except Exception:
        return "unknown"


def _stage_queries(query_pdbs: Iterable[str], staging_dir: Path) -> List[str]:
    """Copy / symlink query PDBs into a single flat directory for Foldseek.

    Foldseek's `easy-search` accepts either a directory of structures, an
    explicit list of files, or a Foldseek DB. A directory is the simplest
    path because the m8 output uses the input filename as the query
    identifier — so we use the original PDB basename as the join key.

    Returns the list of staged paths (so callers can verify counts).
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for src in query_pdbs:
        src_path = Path(src)
        if not src_path.is_file():
            logger.warning(f"Query PDB not found, skipping: {src}")
            continue
        dst = staging_dir / src_path.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        # Symlink keeps disk usage flat; resolve so the link is robust to cwd.
        dst.symlink_to(src_path.resolve())
        staged.append(str(dst))
    return staged


def _query_id_from_pdb(pdb_path: str) -> str:
    """Foldseek emits the query basename minus .pdb as the query column."""
    return Path(pdb_path).stem


def _run_easy_search(
    foldseek_bin: str,
    query_dir: Path,
    target_db: str,
    out_m8: Path,
    tmp_dir: Path,
    alignment_type: int,
    max_seqs: int,
    sensitivity: float,
    threads: int,
) -> None:
    """Run `foldseek easy-search` writing the m8 file in place."""
    cmd = [
        foldseek_bin,
        "easy-search",
        str(query_dir),
        str(target_db),
        str(out_m8),
        str(tmp_dir),
        "--alignment-type",
        str(alignment_type),
        "--tmscore-threshold",
        "0.0",
        "--max-seqs",
        str(max_seqs),
        "-s",
        str(sensitivity),
        "--threads",
        str(threads),
        "--format-output",
        "query,target,alntmscore,lddt",
        "-v",
        "2",  # warnings + errors only
    ]
    logger.info(f"foldseek cmd: {' '.join(cmd)}")
    res = subprocess.run(cmd, check=False, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"foldseek easy-search exited {res.returncode}; check stderr above."
        )


def _parse_per_query_max_tm(m8_path: Path) -> dict:
    """Stream the m8 file and return query_id -> (max_tm, best_target).

    Output format (set via --format-output): one row per (query, target) hit
    with columns query, target, alntmscore, lddt. We don't load the whole
    file into pandas — it can be hundreds of MB for big DBs.
    """
    best: dict = {}
    if not m8_path.exists():
        return best
    with open(m8_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            q, t = parts[0], parts[1]
            try:
                tm = float(parts[2])
            except ValueError:
                continue
            cur = best.get(q)
            if cur is None or tm > cur[0]:
                best[q] = (tm, t)
    return best


def compute_novelty_foldseek(
    query_pdbs: List[str],
    target_db: str,
    *,
    db_label: str,
    foldseek_bin: Optional[str] = None,
    alignment_type: int = 2,
    max_seqs: int = 1000,
    sensitivity: float = 9.5,
    threads: int = 0,
    tm_threshold: float = 0.5,
    keep_artifacts: bool = False,
    work_root: Optional[str] = None,
) -> dict:
    """Compute novelty by running Foldseek easy-search vs `target_db`.

    Args:
        query_pdbs: PDB paths to score. Typically the designable subset of
            generated samples — feed in `designable_pdbs` from
            compute_designability_metrics for paper-style protocol.
        target_db: Path to the foldseek-formatted target DB (the file written
            as `<DEST>/db` by `foldseek databases ...`). The siblings (.idx,
            .dbtype, .lookup, etc.) must be present in the same directory.
        db_label: Short tag (e.g. "pdb", "afdb_swissprot") that suffixes the
            emitted column names: `_res_novelty_foldseek_<label>_*`.
        alignment_type: Foldseek --alignment-type. 2 = 3Di+TMalign refine
            (default, paper-comparable score scale, fast). 1 = pure TM-align
            on every survivor (paper-exact). 0 = 3Di Gotoh only (fastest,
            approximate alntmscore).
        max_seqs: Foldseek prefilter cap. 1000 = default; 10000 for safety
            on borderline-novel queries.
        sensitivity: Foldseek -s.
        threads: Foldseek --threads. 0 (default) lets Foldseek pick all cores.
        tm_threshold: Boundary used to compute the rate column. 0.5 by default.
        keep_artifacts: If True, leave the m8 + tmp dir on disk under
            `work_root` for debugging. Default cleans up.
        work_root: Where to stage queries + write m8 files. Defaults to
            $TMPDIR or /tmp/foldseek_novelty.

    Returns:
        dict of `_res_novelty_foldseek_<label>_*` columns. Keys:
            _rate                  fraction of evaluated queries with max_tm < threshold
            _max_tm_mean           mean max-TM across queries
            _max_tm_median         median max-TM
            _n                     queries actually evaluated (= len(query_pdbs)
                                   minus any that didn't appear in the m8)
            _n_input               len(query_pdbs)
            _alignment_type        flag value used (for traceability)
            _max_seqs              flag value used
            _sensitivity           flag value used
            _foldseek_version      version commit hash
            _target_db             absolute path of the target DB
        Empty dict if `query_pdbs` is empty.
    """
    if not query_pdbs:
        return {}

    fbin = _ensure_foldseek_on_path(foldseek_bin)

    # Validate target DB up front so we fail loud rather than wedging in subprocess.
    if not Path(target_db).exists():
        raise FileNotFoundError(
            f"foldseek target DB not found: {target_db}. Build it with "
            f"hpc-scripts/proteina/data_prep/download_foldseek_dbs.sh."
        )

    work_root_path = Path(
        work_root or os.environ.get("TMPDIR") or "/tmp/foldseek_novelty"
    )
    work_root_path.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp(prefix=f"fs_nov_{db_label}_", dir=work_root_path))
    query_dir = work_dir / "queries"
    fs_tmp = work_dir / "tmp"
    out_m8 = work_dir / "novelty.m8"
    fs_tmp.mkdir(parents=True, exist_ok=True)

    try:
        staged = _stage_queries(query_pdbs, query_dir)
        if not staged:
            logger.warning(
                f"No queries staged for foldseek (label={db_label}); skipping."
            )
            return {}

        logger.info(
            f"Foldseek novelty [{db_label}]: {len(staged)} queries vs {target_db}"
        )
        _run_easy_search(
            fbin,
            query_dir=query_dir,
            target_db=target_db,
            out_m8=out_m8,
            tmp_dir=fs_tmp,
            alignment_type=alignment_type,
            max_seqs=max_seqs,
            sensitivity=sensitivity,
            threads=threads,
        )

        per_query = _parse_per_query_max_tm(out_m8)
        # Map staged path -> query id -> max-TM.
        max_tms = []
        for q in staged:
            qid = _query_id_from_pdb(q)
            entry = per_query.get(qid)
            if entry is None:
                # Foldseek may emit zero hits for a query whose 3Di prefilter
                # had nothing surviving --tmscore-threshold; treat as max=0
                # (i.e. fully novel) rather than dropping silently.
                max_tms.append(0.0)
            else:
                max_tms.append(entry[0])

        max_tm_arr = np.asarray(max_tms, dtype=np.float64)
        rate = float((max_tm_arr < tm_threshold).mean())
        prefix = f"_res_novelty_foldseek_{db_label}_"
        return {
            f"{prefix}rate": rate,
            f"{prefix}max_tm_mean": float(max_tm_arr.mean()),
            f"{prefix}max_tm_median": float(np.median(max_tm_arr)),
            f"{prefix}n": int(len(max_tm_arr)),
            f"{prefix}n_input": int(len(query_pdbs)),
            f"{prefix}alignment_type": int(alignment_type),
            f"{prefix}max_seqs": int(max_seqs),
            f"{prefix}sensitivity": float(sensitivity),
            f"{prefix}foldseek_version": _foldseek_version(fbin),
            f"{prefix}target_db": str(target_db),
        }
    finally:
        if not keep_artifacts:
            shutil.rmtree(work_dir, ignore_errors=True)


def parse_target_dbs_arg(spec: Optional[str]) -> List[tuple]:
    """Parse a `--foldseek_target_dbs` CLI value into [(label, db_path), ...].

    Format: comma-separated `label=path` pairs. Examples:
        "pdb=/rds/.../foldseek_dbs/pdb/db"
        "pdb=/.../pdb/db,afdb_swissprot=/.../afdb_swissprot/db"

    None or empty string -> []. Allows the existing centroid path to keep
    running unchanged when Foldseek isn't configured.
    """
    if not spec:
        return []
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Bad foldseek target spec '{item}': expected 'label=path'"
            )
        label, path = item.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Bad foldseek target spec '{item}'")
        out.append((label, path))
    return out
