#!/usr/bin/env bash
# Back up training results to an rclone remote (e.g. Google Drive).
#
#   scripts/backup_runs.sh <runs_dir> <remote:path> [--with-checkpoints|--all-checkpoints] [--checkpoints-only]
#
# Light tier (default): logs, metrics, configs, samples and TensorBoard events.
# A few MB per run, so it is cheap enough to run on a timer while training.
# Downloading the result and pointing `tensorboard --logdir` at it reproduces
# the full comparison offline -- the event files are all TensorBoard needs.
#
# --with-checkpoints  also copies each run's final `latest.pt` (~1.3 GB at 108M
#                     parameters); intermediate `ckpt_step*.pt` are skipped.
# --all-checkpoints   also copies every `ckpt_step*.pt`.  Because rclone `copy`
#                     never deletes at the destination, this accumulates the
#                     full checkpoint history even though training prunes all
#                     but the last `keep_last_checkpoints` locally -- so the
#                     backup interval must stay well under
#                     keep_last_checkpoints x checkpoint_every_steps.
# --checkpoints-only  drops the light tier, leaving `latest.pt` (and every
#                     `ckpt_step*.pt` with --all-checkpoints).  For the
#                     checkpoint watcher, so that each file has one writer: two
#                     rclone processes uploading the same new file at once
#                     leave two same-named objects on Drive.
set -uo pipefail

USAGE="usage: backup_runs.sh <runs_dir> <remote:path> [--with-checkpoints|--all-checkpoints] [--checkpoints-only]"
SRC=${1:?$USAGE}
DST=${2:?$USAGE}
shift 2

WITH_CKPT=0
ALL_CKPT=0
CKPT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --with-checkpoints) WITH_CKPT=1 ;;
    --all-checkpoints)  WITH_CKPT=1; ALL_CKPT=1 ;;
    --checkpoints-only) WITH_CKPT=1; CKPT_ONLY=1 ;;
    "") ;;  # backup_watch.sh passes one empty word when it has no extra flags
    *) echo "[backup] unknown flag '$arg' -- $USAGE" >&2; exit 2 ;;
  esac
done

# rclone warns that mixing --include and --exclude has *indeterminate* parse
# order, so everything goes through --filter, where rules are first-match-wins
# in the order given.  The trailing "- **" is what makes the include list
# exhaustive; the ckpt_step rule leads so intermediate checkpoints can never be
# picked up by a later, broader rule.
FILTERS=( --filter "- **/*.tmp" )
[ "$ALL_CKPT" = 1 ] && FILTERS+=( --filter "+ **/ckpt_step*.pt" ) \
                    || FILTERS+=( --filter "- **/ckpt_step*.pt" )
if [ "$CKPT_ONLY" = 0 ]; then
  FILTERS+=(
    --filter "+ *.log"
    --filter "+ **/metrics.jsonl"
    --filter "+ **/samples.txt"
    --filter "+ **/config.yaml"
    --filter "+ **/config.json"
    --filter "+ **/summary.json"
    --filter "+ **/stopped.json"
    --filter "+ **/diverged.json"
    --filter "+ **/feature_usage.*"
    --filter "+ **/tb/**"
  )
fi
[ "$WITH_CKPT" = 1 ] && FILTERS+=( --filter "+ **/latest.pt" )
FILTERS+=( --filter "- **" )

tier=light; [ "$WITH_CKPT" = 1 ] && tier=final-checkpoint; [ "$ALL_CKPT" = 1 ] && tier=all-checkpoints
[ "$CKPT_ONLY" = 1 ] && tier="$tier, checkpoints only"
echo "[backup] $SRC -> $DST  (tier: $tier)"
# --drive-chunk-size is the one that matters for multi-GB checkpoints: at the
# 8M default this host managed 1.4 MB/s to Drive, at 128M it manages 10 MB/s,
# against a measured 103 MB/s raw upstream.  Slower than the link, but well
# inside the checkpoint cadence.
#
# --local-no-check-updated is for the files training is still appending to
# (event files, metrics.jsonl, the run log).  Without it rclone aborts such a
# file ("source file is being updated") or, when it grew between upload and
# hash check, reports "corrupted on transfer: md5 hash differ" and DELETES the
# remote copy: on a live run 7 of 8 cycles failed and the event file ended up
# missing from Drive (guide section 5).  With it, rclone sends the size it saw
# at first stat and checksums only that, a consistent prefix of an append-only
# file.  A file rewritten in place still fails the check.  Checkpoints are not
# at risk: save_checkpoint writes a .tmp and renames, so they never change.
rclone copy "$SRC" "$DST" "${FILTERS[@]}" \
    --drive-chunk-size 128M --drive-pacer-min-sleep 10ms \
    --transfers 4 --checkers 16 --retries 3 --low-level-retries 10 \
    --local-no-check-updated \
    --stats 30s --stats-one-line -v
rc=$?
echo "[backup] rclone exit=$rc  $(date -Is)"
exit $rc
