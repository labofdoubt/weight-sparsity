#!/usr/bin/env bash
# Run a job file of toy_mnist training runs with N concurrent workers on one
# machine (the runs are small; several share one GPU).  FIFO with flock: lines
# are consumed as they start, so the file is also "what is left".
#
#   bash toy_mnist/run_jobs.sh --jobs toy_mnist/configs/pop_sweep/jobs.txt \
#       --out /workspace/runs_mnist/pop_sweep --workers 2 \
#       --shared "data.data_dir=/workspace/data/mnist"
#
# jobs.txt: one job per line    run_name|config.yaml|section.key=value ...
# Each run logs to $OUT/logs/<run_name>.log; a run without summary.json did
# not finish.  Run from the repository root, inside tmux.
set -uo pipefail

JOBS= ; OUT= ; WORKERS=1 ; SHARED= ; THREADS=
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) JOBS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --workers) WORKERS=$2; shift 2 ;;
    --shared) SHARED=$2; shift 2 ;;
    --threads) THREADS=$2; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
[ -n "$JOBS" ] && [ -n "$OUT" ] || { echo "need --jobs and --out" >&2; exit 2; }
[ -f "$JOBS" ] || { echo "no such job file: $JOBS" >&2; exit 2; }
PYTHON="${PYTHON:-$(command -v python || echo /venv/main/bin/python)}"
"$PYTHON" -c "import toy_mnist.train" || { echo "$PYTHON cannot import toy_mnist (run from the repo root)" >&2; exit 2; }
if [ -z "$THREADS" ]; then
  quota=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | awk '$1 != "max" {print int($1/$2)}')
  THREADS=$(( ${quota:-$(nproc)} / WORKERS )); [ "$THREADS" -lt 1 ] && THREADS=1
fi
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
mkdir -p "$OUT/logs"
LOCK="$OUT/.queue.lock"
echo "[queue] $(wc -l < "$JOBS") job(s), $WORKERS worker(s), $THREADS thread(s) each, out $OUT"

next_job() {  # pops the first non-empty, non-comment line under the lock
  (
    flock 9
    local line
    while IFS= read -r line; do
      [ -z "$line" ] || [ "${line:0:1}" = "#" ] && continue
      echo "$line"; break
    done < "$JOBS"
    [ -n "${line:-}" ] && { tail -n +2 "$JOBS" > "$JOBS.tmp" && mv "$JOBS.tmp" "$JOBS"; }
  ) 9>"$LOCK"
}

worker() {
  local id=$1 line name cfg extra
  while :; do
    line=$(next_job); [ -z "$line" ] && break
    IFS='|' read -r name cfg extra <<< "$line"
    echo "[w$id] $(date +%H:%M:%S) start $name"
    # shellcheck disable=SC2086
    "$PYTHON" -m toy_mnist.train --config "$cfg" train.run_name="$name" train.out_dir="$OUT" \
        $SHARED $extra > "$OUT/logs/$name.log" 2>&1
    if [ -f "$OUT/$name/summary.json" ]; then
      echo "[w$id] $(date +%H:%M:%S) done  $name: $(tail -1 "$OUT/logs/$name.log")"
    else
      echo "[w$id] $(date +%H:%M:%S) FAILED $name (no summary.json); log tail:"; tail -5 "$OUT/logs/$name.log"
    fi
  done
}

for i in $(seq 1 "$WORKERS"); do worker "$i" & done
wait
echo "[queue] all workers finished"
