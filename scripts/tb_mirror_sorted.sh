#!/usr/bin/env bash
# Mirror another machine's TensorBoard runs in, in a chosen order.
#
# TensorBoard orders runs by `run.start_time` -- the wall_time of the FIRST
# event inside the event file -- ascending, with the run name only as a
# tiebreaker (tensorboard/plugins/core/core_plugin.py, `_serve_runs`).  Run
# names, file mtimes and directory order have no effect, and there is no sort
# flag.  A live mirror therefore interleaves with the local box's own runs,
# because both started at about the same wall-clock time.
#
# So: stage the rclone copy (true bytes, so rclone's sync state stays correct),
# then publish each run with its event stream rewritten to a synthetic start
# time -- one hour apart, in (family, K, J) order, all anchored far enough ahead
# that they sort after every local run.  Relative timing inside a run is
# preserved; step-indexed charts are untouched.  Run names stay real, so one
# regex in TensorBoard's filter box still selects the matching runs on both
# boxes.
#
#   scripts/tb_mirror_sorted.sh gdrive:bucket/runs_korea /workspace/runs/korea 300
set -uo pipefail

SRC="${1:?usage: tb_mirror_sorted.sh <remote:path> <dest_dir> [interval_s] [stage_dir]}"
DST="${2:?missing destination}"
INTERVAL="${3:-300}"
STAGE="${4:-/workspace/mirror_$(basename "$DST")}"
SHIFT_PY="${SHIFT_PY:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tb_shift_walltime.py}"
PYTHON="${PYTHON:-/venv/main/bin/python}"
# a fixed anchor (2027-01-15), so the order is stable across cycles and well
# after any real run on the local box
BASE_EPOCH="${BASE_EPOCH:-1800000000}"
STRIDE="${STRIDE:-3600}"

sort_key () {
  # "<family> <K padded> <J padded>": the kappa sweep in (K, J) order first,
  # then the hard runs.  Bash regex, not sed: "\+" is a GNU extension that
  # silently matches nothing under BSD sed.
  local name="$1" k=0 j=0 fam=9
  [[ "$name" =~ _k([0-9]+) ]] && k="${BASH_REMATCH[1]}"
  [[ "$name" =~ _j([0-9]+) ]] && j="${BASH_REMATCH[1]}"
  case "$name" in
    *rblapsum_kappa*) fam=0 ;;
    *rblapsum_through_rank*) fam=1 ;;
    *rout_soft*) fam=2 ;;
    *rout_hard*) fam=3 ;;
  esac
  printf "%d %04d %04d\n" "$fam" "$k" "$j"
}

mkdir -p "$DST" "$STAGE"
echo "[tb-mirror-sorted] every ${INTERVAL}s: $SRC -> $STAGE -> $DST" \
     "(start times from $BASE_EPOCH, ${STRIDE}s apart)"
while :; do
  runs=$(rclone lsf --dirs-only "$SRC" 2>/dev/null | sed 's:/$::')
  for r in $runs; do
    rclone copy "$SRC/$r" "$STAGE/$r" \
        --filter '+ tb/**' --filter '+ config.json' --filter '+ metrics.jsonl' \
        --filter '+ summary.json' --filter '+ diverged.json' --filter '- *' \
        --transfers 8 --checkers 16 --stats 0 2>&1 | tail -1
  done

  i=0
  while IFS= read -r line; do
    r="${line#* * * }"                      # strip the three sort fields
    [ -d "$STAGE/$r" ] || continue
    mkdir -p "$DST/$r"
    for meta in config.json metrics.jsonl summary.json diverged.json; do
      [ -f "$STAGE/$r/$meta" ] && cp -f "$STAGE/$r/$meta" "$DST/$r/$meta"
    done
    if [ -d "$STAGE/$r/tb" ]; then
      "$PYTHON" "$SHIFT_PY" --src "$STAGE/$r/tb" --dst "$DST/$r/tb" \
          --start-epoch "$((BASE_EPOCH + i * STRIDE))" >/dev/null 2>&1
    fi
    i=$((i + 1))
  done < <(for r in $runs; do printf "%s %s\n" "$(sort_key "$r")" "$r"; done | sort)

  echo "[tb-mirror-sorted] published $i run(s) in (family,K,J) order  $(date -Is)"
  sleep "$INTERVAL"
done
