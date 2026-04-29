# Training-set centroids for novelty evaluation

Precomputed centroid panels of training-set protein structures used by the
novelty metric in [`evaluate.py`](../../scripts/evaluate.py)
(`compute_novelty_metrics`). A generation is "novel" if its max TM-score
against any centroid is below the threshold (default 0.5 — same-fold cutoff).

## Artifacts

Outputs live on `/rds` (not in the repo — too large) and are produced by
[`precompute_centroids.sh`](../../../../../hpc-scripts/proteina/data_prep/precompute_centroids.sh).

| Dataset | Source LMDB | Output `.pt` | When to use |
|---|---|---|---|
| PDB | `/rds/.../pdb_train/lmdb/train.lmdb` (51 GB, 425K chains, len 1–511) | `/rds/.../data/centroids_pdb.pt` | All PDB-trained models (n=128, 256, 512) |
| AFDB SwissProt | `/rds/.../afdb_swissprot/lmdb/train.lmdb` (38 GB, 230K chains, len 16–256) | `/rds/.../data/centroids_afdb_swissprot.pt` | AFDB-trained models (n=128, 256 — AFDB is capped at 256 residues so it cannot service n=512) |

Output schema: `torch.save`'d Python list of `np.ndarray[L_i, 37, 3]` (atom37
coordinates), one entry per cluster representative. Loaded back via
`torch.load(..., weights_only=False)` in
[`evaluate.py:283`](../../scripts/evaluate.py#L283).

## Algorithm

Length-stratified greedy TM-score clustering (script:
[`precompute_centroids.py`](../../../../../hpc-scripts/proteina/data_prep/precompute_centroids.py)):

1. **Bin by length.** Default `--length_bin_width 16` → ~32 bins covering
   16–511 residues. Bin edges align with our generation lengths
   (128, 256, 512 are all multiples of 16).
2. **Subsample within bin.** Up to `--max_per_length_group 200` candidates per
   bin (random, seed=42). This caps the *input* to clustering; the number of
   centroids that emerge depends on how diverse the bin is (typically 30–50
   per bin at cap=200).
3. **Greedy TM-cluster.** Walk the bin in order, keep the first as a centroid;
   subsequent structures join an existing cluster if TM ≥ 0.5 to any centroid,
   otherwise become a new centroid.
4. **Output.** Concatenate centroids from all bins into one list, save as `.pt`.

The script reads `train_lengths.npy` + `train_keys.pkl` sidecar files (built
by [`build_lmdb_length_index.py`](../../../../../hpc-scripts/proteina/data_prep/build_lmdb_length_index.py))
to bin **before** opening the LMDB, then fetches only the chosen keys. This
avoids streaming the full 38–51 GB file. Falls back to a streaming scan if
sidecars are absent.

## How to (re)build

```bash
# Smoke test on intr (~2 min, ~5K entries, cap=50):
sbatch --qos=intr --time=00:30:00 \
    hpc-scripts/proteina/data_prep/precompute_centroids.sh \
    --dataset pdb --max_entries 5000 --max_per_length_group 50 \
    --output_path /tmp/centroids_pdb_smoke.pt

# Real PDB run:
sbatch hpc-scripts/proteina/data_prep/precompute_centroids.sh \
    --dataset pdb \
    --output_path /rds/user/sr2173/hpc-work/proteina/data/centroids_pdb.pt

# Real AFDB SwissProt run:
sbatch hpc-scripts/proteina/data_prep/precompute_centroids.sh \
    --dataset afdb \
    --output_path /rds/user/sr2173/hpc-work/proteina/data/centroids_afdb_swissprot.pt
```

CPU-only job (TM-score via biotite, no GPU needed). Account
`computerlab-sl3-cpu`, partition `icelake`. Wall time ~10–20 min for the real
runs (most of it loading entries from LMDB; clustering itself is minutes).

## How it's wired into the sweep

[`sweep_config.yaml`](../../sweep_config.yaml) sets `centroid_path` in
`_defaults`, pointing at `centroids_pdb.pt`. All current profiles (n128,
n256, n512_sm, n512_convergence) inherit this default since their runs are
all PDB-trained. AFDB-trained profiles (when added) should override
`centroid_path: /rds/.../centroids_afdb_swissprot.pt`.

[`run_sweep.py`](../../scripts/run_sweep.py) reads `centroid_path` from the
profile (or `--centroid_path` CLI override) and threads it into
[`evaluate.py`](../../scripts/evaluate.py)'s argv as
`--centroid_path <path>`. `compute_novelty_metrics` skips silently with a
warning if the file is missing — so older eval runs without a built
centroid file still complete normally, just without novelty columns.

Output columns added to `sweep_results.jsonl` / `.csv`:
- `_res_novelty_n` — number of generated PDBs evaluated (capped at `max_eval=500`)
- `_res_novelty_rate` — fraction with max-TM < threshold (i.e., "novel")
- `_res_novelty_max_tm_mean` / `_res_novelty_max_tm_median` — distribution of max-TM scores

## Caveats and known biases

**Subsample cap inflates novelty.** With `max_per_length_group=200`, we keep
≤200 input candidates per length bin, which (pre-clustering) is a small
fraction of bins like 100–200 residues that contain thousands of training
structures. A generation gets called "novel" iff it's TM<0.5 against every
centroid we kept; if we discarded a structure that would have matched, we
under-count non-novelty. So **reported novelty is an upper bound** under
this panel.

This is fine for **comparing models at fixed panel** (baseline vs REPA at
matched training budget) — the bias hits both equally and the relative
ranking is meaningful. To tighten the absolute number for headline
comparisons against the literature, rebuild with
`--max_per_length_group 1000` or 2000.

**AFDB is length-capped at 256.** Cannot be used for novelty on n=512
generations. Currently moot since no AFDB-trained models generate at 512;
flag this if/when n=512 AFDB runs are added.

**Per-bin clustering, not global.** Two structurally similar generations of
very different lengths (e.g. a 100-residue and 200-residue version of the
same fold) won't be matched. This is by design (TM-score is most meaningful
within similar lengths) but means the panel is a length-stratified rather
than fully global representation of training-set fold diversity.

## Provenance

| Built | Job ID | `max_per_length_group` | `length_bin_width` | `tm_threshold` | Centroids saved | Wall time |
|---|---|---|---|---|---|---|
| 2026-04-27 | 28519486 (AFDB) | 200 | 16 | 0.5 | 2194 (16 bins, lengths 16–256) | ~9 min |
| 2026-04-28 | 28559111 (PDB) | 200 | 16 | 0.5 | 4441 (32 bins, lengths 3–510) | ~10 min |
| ~~2026-04-27~~ | ~~28519485 (PDB)~~ | — | — | — | ~~OOM at `torch.save` with default 3.3 GB mem — fix: `--mem=16G` + per-bin RSS release~~ | — |

SLURM logs at `/rds/user/sr2173/hpc-work/proteina/logs/centroids-<jobid>.out`.

Smoke test (job 28518993): PDB, `max_entries=5000`, cap=50 → 1229 centroids
across 32 bins, ~2 min wall.
