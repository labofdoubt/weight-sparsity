#!/usr/bin/env bash
# Periodic backup of the analysis datasets (scores / probe / gain / wnorm) to
# an rclone remote, meant to run in its own tmux session:
#
#   tmux new -d -s backup_analysis "bash scripts/backup_analysis_watch.sh \
#       /workspace/analysis gdrive:weight-sparsity/analysis_<machine> 600"
#
# A separate script because backup_runs.sh must not be reused here: its filter
# list admits logs/configs/events and nothing else, so it would silently skip
# every .npy -- and the arrays ARE the result (the probe pipelines take hours
# of GPU time to recompute; the viewer is useless without them).
#
# rclone `copy` never deletes at the destination, and --min-age keeps a file
# that is being written right now out of this cycle; it is picked up on the
# next one, so a truncated mid-write copy can only ever be transient.
set -uo pipefail

SRC=${1:?usage: backup_analysis_watch.sh <analysis_dir> <remote:path> [interval_s]}
DST=${2:?usage: backup_analysis_watch.sh <analysis_dir> <remote:path> [interval_s]}
INTERVAL=${3:-600}

echo "[backup-analysis] every ${INTERVAL}s: $SRC -> $DST"
while true; do
  rclone copy "$SRC" "$DST" \
    --filter "- **/*.tmp" --filter "- **/.~*" \
    --min-age 2m --drive-chunk-size 128M --transfers 4 --stats-one-line -v \
    || echo "[backup-analysis] rclone failed, retrying next cycle"
  echo "[backup-analysis] cycle done  $(date -Is)"
  sleep "$INTERVAL"
done
