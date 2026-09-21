#!/usr/bin/env bash
# Push TensorBoard event files (and the small per-run JSONs) to a remote while
# training is still writing them.
#
# A direct `rclone copy` of a live event file races the writer: rclone uploads,
# Drive compares md5, the file has grown in the meantime, and rclone reports
# "corrupted on transfer: md5 hash differ" / "source file is being updated" and
# then fails the whole cycle after three attempts.  The files are only a few MB,
# so snapshot them first and upload the snapshot -- static input, matching hash.
# TensorBoard reads a truncated record stream happily, so a snapshot taken
# mid-write is a valid prefix.
#
#   scripts/tb_push.sh /workspace/runs gdrive:bucket/runs_korea 60
#
# Use this alongside backup_watch.sh (which handles checkpoints, whose writes
# have completed by the time they are seen).
set -uo pipefail

SRC="${1:?usage: tb_push.sh <runs_dir> <remote:path> [interval_s] [stage_dir]}"
DST="${2:?missing remote}"
INTERVAL="${3:-60}"
STAGE="${4:-/tmp/tb_stage}"

mkdir -p "$STAGE"
echo "[tb-push] every ${INTERVAL}s: $SRC -> $DST (snapshot via $STAGE)"
while :; do
  # snapshot: event files plus the cheap metadata a reader needs
  while IFS= read -r -d '' f; do
    rel="${f#$SRC/}"
    mkdir -p "$STAGE/$(dirname "$rel")"
    cp -f "$f" "$STAGE/$rel" 2>/dev/null
  done < <(find "$SRC" -type f \( -name 'events.out.tfevents*' -o -name 'config.json' \
             -o -name 'metrics.jsonl' -o -name 'summary.json' -o -name 'diverged.json' \) -print0)

  rclone copy "$STAGE" "$DST" --transfers 8 --checkers 16 \
      --stats-one-line --stats 0 2>&1 | tail -2
  n=$(find "$STAGE" -name 'events.out.tfevents*' | wc -l)
  echo "[tb-push] pushed $n event file(s)  $(date -Is)"
  sleep "$INTERVAL"
done
