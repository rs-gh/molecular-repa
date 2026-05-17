#!/usr/bin/env bash
# n128 convergence sweep monitor.
# Polls SLURM queue and the four sweep_results.jsonl files; auto-commits +
# pushes when new rows land.
#
# Usage (called from a /loop wake-up):
#   bash playground/n128_convergence_monitor.sh

set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

JOBS=(29486180 29486181 29486189 29486190)
LABELS=("gen-pdb" "gen-afdb" "rep-pdb" "rep-afdb")

GEN_PDB_JSONL="evaluation/proteina/generation/results/paper/n128_convergence_pdb/sweep_results.jsonl"
GEN_AFDB_JSONL="evaluation/proteina/generation/results/paper/n128_convergence_afdb/sweep_results.jsonl"
REP_PDB_DIR="evaluation/proteina/representation/results/paper/n128_convergence_cath_if_dih_pdb"
REP_AFDB_DIR="evaluation/proteina/representation/results/paper/n128_convergence_cath_if_dih_afdb"

echo "=== n128 convergence monitor: $(date) ==="

# Job status
echo "--- SLURM queue (n128 sweep arrays) ---"
for i in "${!JOBS[@]}"; do
    jid="${JOBS[$i]}"
    label="${LABELS[$i]}"
    state=$(squeue -h -j "$jid" --format='%T,%D,%R' 2>/dev/null | head -1)
    if [ -z "$state" ]; then
        # Maybe completed; check sacct for last status across the array
        sa=$(sacct -j "$jid" -X -P --format=JobID,State,Elapsed 2>/dev/null | tail -n +2 | head -3)
        printf "%-10s %s [FINISHED/missing from squeue] sacct:\n%s\n" "$label" "$jid" "$sa"
    else
        # Use squeue array summary
        n_run=$(squeue -h -j "$jid" -r --format='%T' 2>/dev/null | grep -c RUNNING || true)
        n_pend=$(squeue -h -j "$jid" -r --format='%T' 2>/dev/null | grep -c PENDING || true)
        n_done=$(sacct -j "$jid" -X -P --format=JobID,State 2>/dev/null | tail -n +2 | grep -cE 'COMPLETED|FAILED|CANCELLED|TIMEOUT' || true)
        printf "%-10s %s  pending=%s running=%s done=%s\n" "$label" "$jid" "$n_pend" "$n_run" "$n_done"
    fi
done

# Result file sizes
echo "--- Result files ---"
for f in "$GEN_PDB_JSONL" "$GEN_AFDB_JSONL"; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        size=$(du -h "$f" | cut -f1)
        printf "  %-70s lines=%s size=%s\n" "$f" "$lines" "$size"
    else
        printf "  %-70s (not yet created)\n" "$f"
    fi
done
for d in "$REP_PDB_DIR" "$REP_AFDB_DIR"; do
    if [ -d "$d" ]; then
        shards=$(ls "$d" 2>/dev/null | grep -c "\.jsonl$" || true)
        printf "  %-70s shards=%s\n" "$d" "$shards"
    else
        printf "  %-70s (not yet created)\n" "$d"
    fi
done

# Auto-commit any new result rows
echo "--- Git: scan for new results ---"
TO_ADD=()
for path in "$GEN_PDB_JSONL" "$GEN_AFDB_JSONL" "$REP_PDB_DIR" "$REP_AFDB_DIR"; do
    if [ -e "$path" ]; then
        # is there any change vs HEAD?
        if ! git diff --quiet HEAD -- "$path" 2>/dev/null || \
           [ -n "$(git ls-files --others --exclude-standard -- "$path" 2>/dev/null)" ]; then
            TO_ADD+=("$path")
        fi
    fi
done

if [ "${#TO_ADD[@]}" -eq 0 ]; then
    echo "  no new result data to commit."
else
    echo "  staging: ${TO_ADD[*]}"
    git add "${TO_ADD[@]}"
    if git diff --cached --quiet -- "${TO_ADD[@]}"; then
        echo "  (after staging, no diff in TO_ADD paths — skipping commit)"
    else
        ts=$(date +%Y%m%d-%H%M)
        msg="n128 convergence: incremental results $ts"
        # Commit ONLY the n128 result paths (-- <pathspec> arg form). Pre-existing
        # unstaged changes in other dirs stay untouched.  Pre-commit's end-of-file-fixer
        # may modify our staged files on the first attempt; if so, re-stage and retry.
        for attempt in 1 2; do
            if git commit -m "$msg" -- "${TO_ADD[@]}" 2>&1 | tail -5; then
                break
            fi
            echo "  commit attempt $attempt failed; re-staging hook auto-fixes and retrying"
            git add "${TO_ADD[@]}"
        done
        git push 2>&1 | tail -3
    fi
fi

echo "=== monitor pass complete: $(date) ==="
