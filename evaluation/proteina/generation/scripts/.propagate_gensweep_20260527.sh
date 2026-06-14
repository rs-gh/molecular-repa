#!/bin/bash
# One-shot propagation of 2026-05-27 gen-sweep data:
#   raw sweep_results.jsonl  ->  .clean.jsonl  ->  paper TSVs  ->  plots
# Data steps fail hard (prerequisites); plots fail soft (one bad plot must not
# block the rest). Everything logged. Lives in-repo (untracked, dotfile) so it
# survives /tmp reaping.
set -uo pipefail
REPO=/home/sr2173/git/molecular-repa
GEN="$REPO/evaluation/proteina/generation"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate

LOG=/home/sr2173/git/molecular-repa/evaluation/proteina/generation/scripts/.propagate_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "=== Propagation start $(date) ==="
echo "log: $LOG"

set -e
for f in \
  "$GEN/results/variance/n256_afdb_sampler_ablation/sweep_results.jsonl" \
  "$GEN/results/paper/n256_convergence_pdb/sweep_results.jsonl" \
  "$GEN/results/variance/n256_sampler_ablation/sweep_results.jsonl" ; do
  echo; echo "--- clean_variance_jsonl.py $f"
  python "$GEN/scripts/clean_variance_jsonl.py" "$f"
done

echo; echo "--- jsonl_to_tsv.py all"
python "$GEN/scripts/paper/jsonl_to_tsv.py" all
set +e

PLOTS=(
  "scripts/paper/plot_sampler_ablation.py --dataset pdb"
  "scripts/paper/plot_sampler_ablation.py --dataset afdb"
  "scripts/paper/plot_ss_vs_gamma.py"
  "scripts/paper/build_sampler_ablation_table.py"
  "scripts/paper/build_sampler_regime_robustness.py"
  "scripts/paper/plot_convergence_des.py"
  "scripts/paper/plot_convergence_des_ci.py"
  "scripts/paper/plot_convergence_des_multi_seed.py"
  "scripts/paper/plot_convergence_fid.py"
  "scripts/paper/plot_convergence_fid_multi_seed.py"
  "scripts/paper/plot_convergence_quality_multi_seed.py"
)
FAILED=()
for p in "${PLOTS[@]}"; do
  echo; echo "--- python $GEN/$p"
  # shellcheck disable=SC2086
  if ! python "$GEN/"$p ; then
    echo "!!! FAILED: $p"
    FAILED+=("$p")
  fi
done

echo; echo "=== Propagation done $(date) ==="
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "PLOTS THAT FAILED (${#FAILED[@]}):"; printf '  %s\n' "${FAILED[@]}"
else
  echo "All plot steps succeeded."
fi
