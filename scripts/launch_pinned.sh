#!/usr/bin/env bash
# Launch one training per line of a manifest, each pinned to its own GPU, and
# wait for all of them.  Unlike gpu_queue.sh (a FIFO over free cards) the card
# of every job is fixed by the manifest, so a campaign table "GPU g runs cell
# c" holds as written.
#
#   scripts/launch_pinned.sh MANIFEST OUT_DIR [THREADS]
#
# MANIFEST lines:  gpu|run_name|config.yaml|--extra --overrides   (# comments ok)
# Each job runs `python $ENTRY --config CFG EXTRA --train.run_name=NAME
# --train.out_dir=OUT_DIR` with its log in OUT_DIR/NAME.log.  ENTRY defaults to
# scripts/train_guard.py (divergence guard, non-finite abort, --stop-step).
set -uo pipefail
MANIFEST=${1:?usage: launch_pinned.sh MANIFEST OUT_DIR [THREADS]}
OUT=${2:?usage: launch_pinned.sh MANIFEST OUT_DIR [THREADS]}
THREADS=${3:-}
PYTHON="${PYTHON:-/venv/main/bin/python}"
ENTRY="${ENTRY:-scripts/train_guard.py}"
"$PYTHON" -c "import wsparse" || { echo "$PYTHON cannot import wsparse" >&2; exit 2; }
n=$(grep -cv '^\s*\(#\|$\)' "$MANIFEST")
if [ -z "$THREADS" ]; then
  quota=$(awk '{print $1/$2}' /sys/fs/cgroup/cpu.max 2>/dev/null)
  [ -z "$quota" ] && quota=$(nproc)
  THREADS=$(( ${quota%.*} / n )); [ "$THREADS" -lt 1 ] && THREADS=1
fi
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
echo "######## $n pinned jobs, $THREADS threads each, entry $ENTRY  $(date -Is) ########"
pids=()
while IFS='|' read -r gpu name conf extra; do
  case "$gpu" in ''|\#*) continue ;; esac
  [ -f "$conf" ] || { echo "missing config $conf" >&2; exit 2; }
  if [ -e "$OUT/$name/metrics.jsonl" ]; then
    echo "refusing to start $name: $OUT/$name already holds a run" >&2; exit 2
  fi
  echo "[gpu$gpu] START $name ($conf $extra)  $(date -Is)"
  ( CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" $ENTRY --config "$conf" $extra \
        --train.run_name="$name" --train.out_dir="$OUT" > "$OUT/$name.log" 2>&1
    echo "[gpu$gpu] DONE  $name rc=$?  $(date -Is)" ) &
  pids+=($!)
done < "$MANIFEST"
wait "${pids[@]}"
echo "######## all done $(date -Is) ########"
