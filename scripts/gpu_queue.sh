#!/usr/bin/env bash
# Run a queue of training jobs across several GPUs, one job per GPU at a time.
#
# Each run is a single-process single-GPU job, so parallelism is just
# CUDA_VISIBLE_DEVICES per worker -- no DDP, no launcher, no shared state.
# Measured on 4x RTX 4090: four concurrent runs cost under 1% throughput each.
#
# A FIFO queue with flock, rather than dealing the jobs out round-robin: run
# time varies with K, and round-robin leaves cards idle on the tail.
#
#   scripts/gpu_queue.sh --jobs jobs.txt --out /workspace/runs \
#       --gpus 0,1,2,3 --shared "$SHARED" [--clean gdrive:bucket/runs]
#
# jobs.txt has one job per line:   run_name|config.yaml|--extra --overrides
# Lines are consumed as they start, so the file also serves as "what is left".
set -uo pipefail

JOBS= ; OUT= ; GPUS=0,1,2,3 ; SHARED= ; CLEAN= ; THREADS=
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) JOBS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --shared) SHARED=$2; shift 2 ;;
    --clean) CLEAN=$2; shift 2 ;;       # remote to verify against before pruning
    --threads) THREADS=$2; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
[ -n "$JOBS" ] && [ -n "$OUT" ] || { echo "need --jobs and --out" >&2; exit 2; }
[ -f "$JOBS" ] || { echo "no such job file: $JOBS" >&2; exit 2; }

IFS=, read -ra CARDS <<< "$GPUS"
# torch spawns one compute thread per *host* core, but a container is capped by
# a cgroup quota it cannot see; oversubscribing costs more than it buys.
if [ -z "$THREADS" ]; then
  quota=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | awk '{print $1/$2}')
  [ -z "$quota" ] && quota=$(awk '{print $1/100000}' /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)
  [ -z "$quota" ] || [ "$quota" = "0" ] && quota=$(nproc)
  THREADS=$(( ${quota%.*} / ${#CARDS[@]} ))
  [ "$THREADS" -lt 1 ] && THREADS=1
fi
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT"
LOCK="$OUT/.queue.lock"; touch "$LOCK"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "######## $(wc -l < "$JOBS") jobs over ${#CARDS[@]} GPUs, $THREADS threads each  $(date -Is) ########"

worker () {
  local gpu=$1 job NAME REST CONF EXTRA rc
  while :; do
    job=$(flock "$LOCK" -c "head -1 '$JOBS'; sed -i 1d '$JOBS'")
    [ -z "$job" ] && { echo "[gpu$gpu] queue empty $(date -Is)"; return 0; }
    NAME="${job%%|*}"; REST="${job#*|}"; CONF="${REST%%|*}"; EXTRA="${REST#*|}"
    echo "[gpu$gpu] START $NAME  $(date -Is)"
    CUDA_VISIBLE_DEVICES=$gpu python -m wsparse.train --config "$CONF" \
        $SHARED $EXTRA --train.run_name="$NAME" > "$OUT/$NAME.log" 2>&1
    rc=$?
    echo "[gpu$gpu] DONE  $NAME rc=$rc  $(date -Is)"
    sleep 15   # a separate process: exiting returns all device memory
    if [ -n "$CLEAN" ]; then
      # Verified against the remote, so a checkpoint that has not been backed
      # up yet -- including a concurrently running job's newest -- is skipped.
      flock "$LOCK" -c \
        "bash '$HERE/clean_checkpoints.sh' '$OUT' '$CLEAN' --keep-latest-only >/dev/null 2>&1" || true
    fi
  done
}

for g in "${CARDS[@]}"; do worker "$g" & done
wait

echo "######## queue drained $(date -Is) ########"
for f in "$OUT"/*/summary.json; do
  [ -f "$f" ] || continue
  printf "  %-34s %s\n" "$(basename "$(dirname "$f")")" \
    "$(python -c "import json;print(round(json.load(open('$f'))['best_val_ce'],4))" 2>/dev/null)"
done
