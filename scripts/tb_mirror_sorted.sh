#!/usr/bin/env bash
# Mirror another machine's TensorBoard events in, under sortable names AND in a
# controlled position in TensorBoard's run list.
#
# Two things govern what you see in TensorBoard 2.x:
#   * the run NAME is its path relative to --logdir;
#   * the run ORDER is ascending by event-file modification time -- NOT by name
#     (measured on TB 2.21: 60 of 85 adjacent pairs ascending, oldest first).
# So zero-padding the names is necessary but not sufficient: while the local box
# is still training, a live mirror interleaves with the local runs because both
# sets are being written continuously.
#
# Hence stage, then publish:
#   remote --rclone--> STAGE/<real name>      (true mtimes; rclone's sync state)
#          --copy---->  DST/<k032_j032_...>   (synthetic mtimes; what TB sees)
# The published copies get mtimes far in the future (base = now + 30 days),
# increasing in label order, so the mirrored runs appear *after* every local run
# and in (K, J) order among themselves.  rclone never sees the touched copies,
# so its size/mtime comparison against the remote stays correct.
#
#   scripts/tb_mirror_sorted.sh gdrive:bucket/runs_korea /workspace/runs/korea 300
set -uo pipefail

SRC="${1:?usage: tb_mirror_sorted.sh <remote:path> <dest_dir> [interval_s] [stage_dir]}"
DST="${2:?missing destination}"
INTERVAL="${3:-300}"
STAGE="${4:-/workspace/mirror_$(basename "$DST")}"
# far enough ahead that a still-training local run can never overtake it
FUTURE_OFFSET=$((30 * 86400))

label_for () {
  # k / j with leading zeros, plus a short family + variant tag.  Bash regex
  # rather than sed: "\+" is a GNU extension and silently matches nothing on
  # BSD sed, which made an earlier version fall through to the raw name.
  local name="$1" k="" j="" fam tag=""
  [[ "$name" =~ _k([0-9]+) ]] && k="${BASH_REMATCH[1]}"
  [[ "$name" =~ _j([0-9]+) ]] && j="${BASH_REMATCH[1]}"
  case "$name" in
    *rblapsum_kappa*) fam=kappa ;;
    *rblapsum_through_rank*) fam=throughrank ;;
    *rout_hard*) fam=hard ;;
    *rout_soft*) fam=soft ;;
    *) fam=other ;;
  esac
  if [[ "$name" =~ _k[0-9]+(_j[0-9]+)?_(.+)$ ]]; then
    tag="${BASH_REMATCH[2]}"
    tag="${tag%_md_abs}"; tag="${tag%_md}"; tag="${tag%_abs}"
  fi
  [ -z "$k" ] && { echo "$name"; return; }
  if [ -n "$j" ]; then
    printf "k%03d_j%03d_%s_%s\n" "$k" "$j" "$fam" "${tag:-x}"
  else
    printf "k%03d_%s_%s\n" "$k" "$fam" "${tag:-x}"
  fi
}

mkdir -p "$DST" "$STAGE"
echo "[tb-mirror-sorted] every ${INTERVAL}s: $SRC -> $STAGE -> $DST"
while :; do
  for r in $(rclone lsf --dirs-only "$SRC" 2>/dev/null | sed 's:/$::'); do
    rclone copy "$SRC/$r" "$STAGE/$r" \
        --filter '+ tb/**' --filter '+ config.json' --filter '+ metrics.jsonl' \
        --filter '+ summary.json' --filter '+ diverged.json' --filter '- *' \
        --transfers 8 --checkers 16 --stats 0 2>&1 | tail -1
  done

  # publish in label order, stamping increasing future mtimes
  base=$(( $(date +%s) + FUTURE_OFFSET ))
  i=0
  while IFS=$'\t' read -r lab real; do
    [ -d "$STAGE/$real" ] || continue
    mkdir -p "$DST/$lab"
    cp -rf "$STAGE/$real/." "$DST/$lab/" 2>/dev/null
    ts=$(( base + i * 60 ))
    find "$DST/$lab" -type f -exec touch -d "@$ts" {} + 2>/dev/null
    find "$DST/$lab" -type d -exec touch -d "@$ts" {} + 2>/dev/null
    i=$((i + 1))
  done < <(for r in $(rclone lsf --dirs-only "$SRC" 2>/dev/null | sed 's:/$::'); do
             printf "%s\t%s\n" "$(label_for "$r")" "$r"
           done | sort)

  echo "[tb-mirror-sorted] published $i run(s), $(find "$DST" -name 'events.out.tfevents*' | wc -l) event file(s)  $(date -Is)"
  sleep "$INTERVAL"
done
