#!/usr/bin/env bash
# Periodic light-tier backup, meant to run in its own tmux session:
#
#   tmux new -d -s backup "bash scripts/backup_watch.sh /workspace/runs_bottleneck gdrive:weight-sparsity/runs 600"
#
# Checkpoints are deliberately not copied on the timer -- they are ~1.3 GB each
# and only change at checkpoint boundaries.  Back those up once per finished run
# with `backup_runs.sh ... --with-checkpoints`.
set -uo pipefail

SRC=${1:?usage: backup_watch.sh <runs_dir> <remote:path> [interval_s]}
DST=${2:?usage: backup_watch.sh <runs_dir> <remote:path> [interval_s]}
INTERVAL=${3:-600}
HERE=$(cd "$(dirname "$0")" && pwd)

echo "[backup-watch] every ${INTERVAL}s: $SRC -> $DST"
while true; do
  bash "$HERE/backup_runs.sh" "$SRC" "$DST" || echo "[backup-watch] failed, retrying next cycle"
  sleep "$INTERVAL"
done
