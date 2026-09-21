#!/usr/bin/env bash
# Mirror another machine's TensorBoard events in, under sortable names AND in a
# controlled position in TensorBoard's run list.
#
# TensorBoard names a run by its path relative to --logdir, so this relabels
# each mirrored run as k%03d_j%03d_<family>_<tag>: zero-padded, hence readable
# and correctly ordered anywhere the names are sorted (TB's own run list is NOT
# sorted -- measured on 2.21, the order is its internal parallel load order and
# is not affected by names, mtimes or directory order, and there is no sort
# flag; so keep a mirror in its OWN logdir / TB instance rather than inside a
# still-training box's logdir, where the two sets interleave).
#
# The remote keeps the real run names, so provenance is unaffected.
#
#   scripts/tb_mirror_sorted.sh gdrive:bucket/runs_korea /workspace/runs/korea 300
set -uo pipefail

SRC="${1:?usage: tb_mirror_sorted.sh <remote:path> <dest_dir> [interval_s] [stage_dir]}"
DST="${2:?missing destination}"
INTERVAL="${3:-300}"
STAGE="${4:-/workspace/mirror_$(basename "$DST")}"

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

  # publish under the sortable label (plain copy: no mtime games -- TB does
  # not order by mtime, and touching files here would confuse anything that
  # syncs this tree)
  i=0
  while IFS=$'\t' read -r lab real; do
    [ -d "$STAGE/$real" ] || continue
    mkdir -p "$DST/$lab"
    cp -rf "$STAGE/$real/." "$DST/$lab/" 2>/dev/null
    i=$((i + 1))
  done < <(for r in $(rclone lsf --dirs-only "$SRC" 2>/dev/null | sed 's:/$::'); do
             printf "%s\t%s\n" "$(label_for "$r")" "$r"
           done | sort)

  echo "[tb-mirror-sorted] published $i run(s), $(find "$DST" -name 'events.out.tfevents*' | wc -l) event file(s)  $(date -Is)"
  sleep "$INTERVAL"
done
