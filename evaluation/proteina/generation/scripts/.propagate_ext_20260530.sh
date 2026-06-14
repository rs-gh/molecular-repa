#!/bin/bash
#!
#! One-shot propagation of the 2026-05-30 ext7/ext4 frontier-checkpoint sweep
#! (5 runs × new periodic steps; gen + rep, both datasets) into every dataset,
#! plot, and masters-report figure that consumes it.
#!
#! Jobs feeding this:
#!   gen:  29879663 (afdb_ext7), 29879664 (pdb_ext7)
#!   rep:  29879670 (conv afdb), 29879671 (conv pdb),
#!         29879849 (cleantrain pdb), 29879850 (xclean afdb_pdb),
#!         29879851 (xclean pdb_afdb)
#!
#! Idempotent: reads only completed shards/rows. Safe to re-run as more jobs
#! finish. Data steps fail hard (prerequisites); plots/figures fail soft.
set -uo pipefail
REPO=/home/sr2173/git/molecular-repa
GEN="$REPO/evaluation/proteina/generation"
REP="$REPO/evaluation/proteina/representation"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
export PROJECT_ROOT="$REPO/src/proteina"

LOG="$GEN/scripts/.propagate_ext_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== ext7/ext4 propagation start $(date) ==="
echo "log: $LOG"

# ───────────────────────── GEN data (fail hard) ─────────────────────────────
set -e
for f in \
  "$GEN/results/paper/n256_convergence_pdb/sweep_results.jsonl" \
  "$GEN/results/paper/n256_convergence_afdb/sweep_results.jsonl" ; do
  echo; echo "--- clean_variance_jsonl.py $f"
  python "$GEN/scripts/clean_variance_jsonl.py" "$f"
done
echo; echo "--- jsonl_to_tsv.py all"
python "$GEN/scripts/paper/jsonl_to_tsv.py" all

# ───────────────────────── REP data (fail hard) ─────────────────────────────
# Consolidate per-task shards -> canonical CSV/JSON for each of the 5 regimes.
for prof in \
  paper_n256_cath_if_dih_convergence_pdb_ext4 \
  paper_n256_cath_if_dih_convergence_afdb_ext4 \
  paper_n256_cath_if_dih_cleantrain_pdb_ext4 \
  paper_n256_cath_if_dih_xclean_afdb_pdb_ext4 \
  paper_n256_cath_if_dih_xclean_pdb_afdb_ext4 ; do
  echo; echo "--- consolidate rep shards: $prof"
  python "$REP/scripts/paper/pretrain_probe_sweep.py" --config "$prof" --consolidate_only
done
echo; echo "--- build_clean_manifests_v2.py"
python "$REP/scripts/paper/build_clean_manifests_v2.py"
set +e

# ───────────────────────── PLOTS + FIGURES (fail soft) ──────────────────────
PLOTS=(
  # gen convergence (consumes clean jsonl / TSV)
  "$GEN/scripts/paper/plot_convergence_fid.py"
  "$GEN/scripts/paper/plot_convergence_fid_multi_seed.py"
  "$GEN/scripts/paper/plot_convergence_des.py"
  "$GEN/scripts/paper/plot_convergence_des_ci.py"
  "$GEN/scripts/paper/plot_convergence_des_multi_seed.py"
  "$GEN/scripts/paper/plot_convergence_quality_multi_seed.py"
  # gen head-to-head
  "$GEN/scripts/paper/plot_h2h_n256.py"
  # rep convergence / leakage / xclean
  "$REP/scripts/paper/plot_repr_quality_over_training.py"
  "$REP/scripts/paper/plot_leakage_decomp_n256.py"
  "$REP/scripts/paper/plot_xclean_pdb_afdb_trained.py"
  # masters-report figures (fig3 alignment unaffected; skipped)
  "$REPO/docs/masters-report/figures/scripts/fig1c_headline_fid_des.py"
  "$REPO/docs/masters-report/figures/scripts/fig2_representation.py"
  "$REPO/docs/masters-report/figures/scripts/fig4c_gen_vs_rep_combined.py"
)
FAILED=()
for p in "${PLOTS[@]}"; do
  echo; echo "--- python $p"
  if ! python "$p" ; then
    echo "!!! FAILED: $p"
    FAILED+=("$p")
  fi
done

echo; echo "=== ext7/ext4 propagation done $(date) ==="
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "PLOTS/FIGS THAT FAILED (${#FAILED[@]}):"; printf '  %s\n' "${FAILED[@]}"
else
  echo "All plot/figure steps succeeded."
fi
