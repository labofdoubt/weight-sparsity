#!/usr/bin/env bash
# Disk-pressure guard for a vast.ai box: when free space dips below a soft
# threshold, reclaim space with clean_checkpoints.sh --keep-latest-only (which
# deletes ONLY files it first verifies on the Drive remote, so it can never
# destroy a sole copy); if free space is still below a hard threshold after
# that, print a loud warning with the biggest offenders.  Reclaim-and-warn
# only: it never kills a run and never touches /workspace/analysis.
#
#   usage: disk_guard_watch.sh <runs_dir> <drive_remote> [interval_s] [soft_G] [hard_G]
#   e.g.:  disk_guard_watch.sh /workspace/runs gdrive:weight-sparsity/runs_taiwan_2 300 40 15
set -u
RUNS=${1:?runs dir}
REMOTE=${2:?drive remote prefix}
INTERVAL=${3:-300}
SOFT=${4:-40}
HARD=${5:-15}
cd "$(dirname "$0")/.."

free_g() { df -BG "$RUNS" | tail -1 | awk '{print $4}' | tr -d G; }

while true; do
  f=$(free_g)
  if [ "$f" -lt "$SOFT" ]; then
    echo "[disk_guard $(date '+%F %T')] ${f}G free < soft ${SOFT}G -- pruning Drive-verified checkpoints"
    bash scripts/clean_checkpoints.sh "$RUNS" "$REMOTE" --keep-latest-only 2>&1 | tail -2
    f=$(free_g)
    if [ "$f" -lt "$HARD" ]; then
      echo "[disk_guard $(date '+%F %T')] STILL ${f}G free < hard ${HARD}G -- MANUAL ACTION NEEDED. Top offenders:"
      du -sh /workspace/runs/* /workspace/analysis/* 2>/dev/null | sort -rh | head -8
    fi
  fi
  sleep "$INTERVAL"
done
