#!/usr/bin/env bash
# Back up training results to an rclone remote (e.g. Google Drive).
#
#   scripts/backup_runs.sh <runs_dir> <remote:path> [--with-checkpoints]
#
# Light tier (default): logs, metrics, configs, samples and TensorBoard events.
# A few MB per run, so it is cheap enough to run on a timer while training.
# Downloading the result and pointing `tensorboard --logdir` at it reproduces
# the full comparison offline -- the event files are all TensorBoard needs.
#
# --with-checkpoints additionally copies each run's final `latest.pt`
# (~1.3 GB at 108M parameters).  Intermediate `ckpt_step*.pt` are never copied.
set -uo pipefail

SRC=${1:?usage: backup_runs.sh <runs_dir> <remote:path> [--with-checkpoints]}
DST=${2:?usage: backup_runs.sh <runs_dir> <remote:path> [--with-checkpoints]}
shift 2

WITH_CKPT=0
for arg in "$@"; do
  [ "$arg" = "--with-checkpoints" ] && WITH_CKPT=1
done

# Filters are first-match-wins, so the explicit exclude leads.  It is redundant
# while the rules below are all includes (rclone excludes anything unmatched),
# but it keeps intermediate checkpoints out even if someone widens the list.
FILTERS=(
  --exclude "**/ckpt_step*.pt"
  --exclude "**/*.tmp"
  --include "*.log"
  --include "**/metrics.jsonl"
  --include "**/samples.txt"
  --include "**/config.yaml"
  --include "**/config.json"
  --include "**/summary.json"
  --include "**/feature_usage.*"
  --include "**/tb/**"
)
[ "$WITH_CKPT" = 1 ] && FILTERS+=( --include "**/latest.pt" )

echo "[backup] $SRC -> $DST  (checkpoints: $([ $WITH_CKPT = 1 ] && echo yes || echo no))"
rclone copy "$SRC" "$DST" "${FILTERS[@]}" \
    --transfers 8 --checkers 16 --retries 3 --low-level-retries 10 \
    --stats 30s --stats-one-line -v
rc=$?
echo "[backup] rclone exit=$rc  $(date -Is)"
exit $rc
