#!/usr/bin/env bash
# Back up training results to an rclone remote (e.g. Google Drive).
#
#   scripts/backup_runs.sh <runs_dir> <remote:path> [--with-checkpoints|--all-checkpoints]
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
set -uo pipefail

SRC=${1:?usage: backup_runs.sh <runs_dir> <remote:path> [--with-checkpoints]}
DST=${2:?usage: backup_runs.sh <runs_dir> <remote:path> [--with-checkpoints]}
shift 2

WITH_CKPT=0
ALL_CKPT=0
for arg in "$@"; do
  case "$arg" in
    --with-checkpoints) WITH_CKPT=1 ;;
    --all-checkpoints)  WITH_CKPT=1; ALL_CKPT=1 ;;
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
FILTERS+=(
  --filter "+ *.log"
  --filter "+ **/metrics.jsonl"
  --filter "+ **/samples.txt"
  --filter "+ **/config.yaml"
  --filter "+ **/config.json"
  --filter "+ **/summary.json"
  --filter "+ **/feature_usage.*"
  --filter "+ **/tb/**"
)
[ "$WITH_CKPT" = 1 ] && FILTERS+=( --filter "+ **/latest.pt" )
FILTERS+=( --filter "- **" )

tier=light; [ "$WITH_CKPT" = 1 ] && tier=final-checkpoint; [ "$ALL_CKPT" = 1 ] && tier=all-checkpoints
echo "[backup] $SRC -> $DST  (tier: $tier)"
rclone copy "$SRC" "$DST" "${FILTERS[@]}" \
    --transfers 8 --checkers 16 --retries 3 --low-level-retries 10 \
    --stats 30s --stats-one-line -v
rc=$?
echo "[backup] rclone exit=$rc  $(date -Is)"
exit $rc
