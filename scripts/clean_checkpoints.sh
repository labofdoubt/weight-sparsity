#!/usr/bin/env bash
# Free disk by dropping roughly half of each run's checkpoints.
#
# Per run directory: never touches `latest.pt` nor the highest-step
# `ckpt_step*.pt`; of what remains it deletes the OLDEST floor(total/2) files,
# so a run holding {16000, 18000, 20000, latest} loses 16000 and 18000.
#
# Every deletion is gated on the file being present on the rclone remote.  The
# backup uses `rclone copy`, which never mirrors a local delete, so the remote
# copy survives -- but a file that never made it up there is kept locally.
#
# --keep-latest-only drops EVERY numbered checkpoint, keeping just latest.pt.
# Intended for finished runs whose checkpoints are already on the remote: it
# gives up mid-run restart points in exchange for roughly halving the footprint
# again.  A run still training keeps its rolling window unless you pass it.
#
# Usage: clean_checkpoints.sh <runs_dir> <remote:path> [--dry-run] [--keep-latest-only]
set -uo pipefail

SRC="${1:?usage: clean_checkpoints.sh <runs_dir> <remote:path> [--dry-run] [--keep-latest-only]}"
DST="${2:?missing remote}"
DRY=0; LATEST_ONLY=0
for arg in "${@:3}"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --keep-latest-only) LATEST_ONLY=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

MANIFEST=$(mktemp)
trap 'rm -f "$MANIFEST"' EXIT
echo "[clean] listing remote $DST ..."
if ! rclone lsf --files-only -R --include "*.pt" "$DST" > "$MANIFEST" 2>/dev/null; then
  echo "[clean] FATAL: cannot list $DST -- refusing to delete anything" >&2
  exit 1
fi
echo "[clean] remote holds $(wc -l < "$MANIFEST") checkpoint(s)"

freed=0; removed=0; kept_unbacked=0
for d in "$SRC"/*/; do
  run=$(basename "$d")
  mapfile -t numbered < <(
    ls "$d"ckpt_step*.pt 2>/dev/null |
      sed -E 's/.*ckpt_step([0-9]+)\.pt/\1 &/' | sort -n | cut -d' ' -f2-
  )
  total=$(ls "$d"*.pt 2>/dev/null | wc -l)
  n=${#numbered[@]}

  if (( LATEST_ONLY )); then
    # only safe if latest.pt is actually there to fall back on
    if [ ! -f "$d/latest.pt" ]; then
      printf '%-30s %s\n' "$run" "no latest.pt, leaving its $n checkpoint(s) alone"
      continue
    fi
    budget=$n
  else
    (( n < 2 )) && { printf '%-30s %s\n' "$run" "only $n numbered ckpt, skipping"; continue; }
    # candidates = every numbered ckpt except the newest; budget = half of all .pt
    budget=$(( total / 2 ))
    (( budget > n - 1 )) && budget=$(( n - 1 ))
  fi
  printf '%-30s %d .pt -> dropping %d\n' "$run" "$total" "$budget"

  for (( i = 0; i < budget; i++ )); do
    f="${numbered[$i]}"; base=$(basename "$f")
    if ! grep -qxF "$run/$base" "$MANIFEST"; then
      echo "    KEEP  $base  (not on remote)"
      kept_unbacked=$(( kept_unbacked + 1 ))
      continue
    fi
    sz=$(du -m "$f" | cut -f1)
    if (( DRY )); then
      echo "    would delete $base  (${sz} MiB)"
    else
      rm -f "$f" && echo "    deleted $base  (${sz} MiB)"
    fi
    freed=$(( freed + sz )); removed=$(( removed + 1 ))
  done
done

echo "[clean] $( ((DRY)) && echo 'would remove' || echo removed ) $removed file(s), $(( freed / 1024 )) GiB"
(( kept_unbacked )) && echo "[clean] kept $kept_unbacked unbacked file(s)"
df -h "$SRC" | tail -1
