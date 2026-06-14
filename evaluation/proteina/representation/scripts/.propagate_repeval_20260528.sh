#!/bin/bash
# Propagate today's rep-eval CSV updates to figures.
# Order: build_clean_manifests_v2 (prereq) -> 3 plot scripts.
set -uo pipefail
REPO=/home/sr2173/git/molecular-repa
REP="$REPO/evaluation/proteina/representation"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate

LOG="$REP/scripts/.propagate_repeval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== Rep-eval propagation start $(date) ==="
echo "log: $LOG"

# build_clean_manifests is a prerequisite (xclean plots may depend on it) — fail hard
set -e
echo; echo "--- python $REP/scripts/paper/build_clean_manifests_v2.py"
python "$REP/scripts/paper/build_clean_manifests_v2.py"
set +e

PLOTS=(
  "scripts/paper/plot_repr_quality_over_training.py"
  "scripts/paper/plot_leakage_decomp_n256.py"
  "scripts/paper/plot_xclean_pdb_afdb_trained.py"
)
FAILED=()
for p in "${PLOTS[@]}"; do
  echo; echo "--- python $REP/$p"
  if ! python "$REP/$p" ; then
    echo "!!! FAILED: $p"
    FAILED+=("$p")
  fi
done

echo; echo "=== Rep-eval propagation done $(date) ==="
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED (${#FAILED[@]}):"; printf '  %s\n' "${FAILED[@]}"
else
  echo "All steps succeeded."
fi
