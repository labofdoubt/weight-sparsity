#!/usr/bin/env bash
# Mirror another machine's TensorBoard events in under *sortable* run names.
#
# TensorBoard names a run by its path relative to --logdir and sorts those
# paths as strings, so "k128" lands before "k32" and "j448" before "j64".
# Zero-padding the numbers makes lexicographic order match numeric order:
#
#   ko_rout_rblapsum_kappa_k32_j480_supp03_md_abs  ->  k032_j480_kappa_supp03
#   ko_rout_hard_k512_pnorm_md                     ->  k512_hard_pnorm
#
# The remote (and therefore provenance) keeps the real run names; only this
# read-only view is relabelled.  One rclone call per run, event files and the
# small JSONs only.
#
#   scripts/tb_mirror_sorted.sh gdrive:bucket/runs_korea /workspace/runs/korea 300
set -uo pipefail

SRC="${1:?usage: tb_mirror_sorted.sh <remote:path> <dest_dir> [interval_s]}"
DST="${2:?missing destination}"
INTERVAL="${3:-300}"

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

mkdir -p "$DST"
echo "[tb-mirror-sorted] every ${INTERVAL}s: $SRC -> $DST (relabelled)"
while :; do
  runs=$(rclone lsf --dirs-only "$SRC" 2>/dev/null | sed 's:/$::')
  for r in $runs; do
    lab=$(label_for "$r")
    rclone copy "$SRC/$r" "$DST/$lab" \
        --filter '+ tb/**' --filter '+ config.json' --filter '+ metrics.jsonl' \
        --filter '+ summary.json' --filter '+ diverged.json' --filter '- *' \
        --transfers 8 --checkers 16 --stats 0 2>&1 | tail -1
  done
  echo "[tb-mirror-sorted] $(find "$DST" -name 'events.out.tfevents*' 2>/dev/null | wc -l)" \
       "event file(s) over $(echo "$runs" | wc -w) run(s)  $(date -Is)"
  sleep "$INTERVAL"
done
