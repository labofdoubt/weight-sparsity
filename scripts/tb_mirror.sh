#!/usr/bin/env bash
# Pull another machine's TensorBoard event files in, so one TensorBoard shows
# every run regardless of which box produced it.
#
# Goes via the rclone remote rather than machine-to-machine: the two servers
# have no SSH trust between them, but both already back up to the same remote.
#
#   scripts/tb_mirror.sh gdrive:bucket/runs_taiwan /workspace/runs/taiwan 300
#
# The destination is a SUBDIRECTORY of the TensorBoard log dir on purpose.
# TensorBoard names a run by its path relative to --logdir, so mirroring into
# <logdir>/taiwan/ makes the runs appear as "taiwan/<run>/tb" and they cannot
# collide with a local run of the same name -- which would otherwise merge into
# one interleaved series, silently and without looking like an error.
#
# Only event files are pulled; checkpoints stay where they are.
set -uo pipefail

SRC="${1:?usage: tb_mirror.sh <remote:path> <dest_dir> [interval_s]}"
DST="${2:?missing destination}"
INTERVAL="${3:-300}"

mkdir -p "$DST"
echo "[tb-mirror] every ${INTERVAL}s: $SRC -> $DST  (event files only)"
while :; do
  rclone copy "$SRC" "$DST" \
      --filter '+ */tb/**' \
      --filter '+ */config.json' \
      --filter '+ */metrics.jsonl' \
      --filter '+ */summary.json' \
      --filter '- *' \
      --transfers 8 --checkers 16 --stats-one-line 2>&1 | tail -2
  echo "[tb-mirror] synced $(find "$DST" -name 'events.out.tfevents*' 2>/dev/null | wc -l) event files  $(date -Is)"
  sleep "$INTERVAL"
done
