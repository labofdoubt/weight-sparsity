#!/usr/bin/env bash
# Periodic light-tier backup, meant to run in its own tmux session:
#
#   backup_watch.sh <runs_dir> <remote:path> [interval_s] [extra backup_runs.sh flags]
#
#   tmux new -d -s backup "bash scripts/backup_watch.sh \\
#       /workspace/runs_bottleneck gdrive:weight-sparsity/runs 600 --all-checkpoints"
#
# With --all-checkpoints the interval must stay comfortably under
# keep_last_checkpoints x checkpoint_every_steps, or training will prune a
# checkpoint locally before the backup has copied it.
set -uo pipefail

SRC=${1:?usage: backup_watch.sh <runs_dir> <remote:path> [interval_s]}
DST=${2:?usage: backup_watch.sh <runs_dir> <remote:path> [interval_s]}
INTERVAL=${3:-600}
shift $(( $# < 3 ? $# : 3 ))
EXTRA=( "$@" )
HERE=$(cd "$(dirname "$0")" && pwd)

echo "[backup-watch] every ${INTERVAL}s: $SRC -> $DST ${EXTRA[*]:-}"
while true; do
  bash "$HERE/backup_runs.sh" "$SRC" "$DST" "${EXTRA[@]:-}" \
    || echo "[backup-watch] failed, retrying next cycle"
  sleep "$INTERVAL"
done
