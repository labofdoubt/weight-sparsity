#!/usr/bin/env bash
# Run a phased GPU plan: phases in order; within a phase one chain per GPU
# set (jobs on the same set run one after another, different sets in
# parallel); a phase ends when all of its chains have ended.  A job on more
# than one GPU runs under torchrun (for a dust config: the draws split over
# the ranks, docs/dust-topk.md section 6).
#
#   bash scripts/gpu_phases.sh configs/small_dust/plan.txt [--extra.overrides ...]
#
# Plan line:  <phase> <comma-separated GPU ids> <config.yaml>   (# comments)
# Every job runs scripts/train_guard.py; its log is $LOGDIR/<config name>.log
# (default /workspace/runs/_logs).  Run from the repo root, in tmux.
set -uo pipefail
PLAN=${1:?plan file}; shift
EXTRA=("$@")
LOGDIR=${LOGDIR:-/workspace/runs/_logs}
PYTHON=${PYTHON:-}
if [ -z "$PYTHON" ]; then
  if [ -x /venv/main/bin/python ]; then PYTHON=/venv/main/bin/python
  else PYTHON=$(command -v python); fi
fi
mkdir -p "$LOGDIR"

# preflight: every config exists and loads, the code has the draw groups
fail=0
while read -r phase gpus cfg; do
  [ -z "${phase:-}" ] && continue
  case "$phase" in \#*) continue ;; esac
  [ -f "$cfg" ] || { echo "[phases] missing config $cfg" >&2; fail=1; }
done < "$PLAN"
"$PYTHON" -c "from wsparse.bottleneck.dust import DustDrawGroup" \
  || { echo "[phases] this wsparse has no DustDrawGroup" >&2; fail=1; }
[ $fail -eq 0 ] || exit 2

# per-process CPU threads: the cgroup quota (v2 cpu.max, else v1 cfs) over the GPUs
if [ -r /sys/fs/cgroup/cpu.max ]; then
  quota=$(awk '{ if ($1 == "max") print 0; else print int($1 / $2) }' /sys/fs/cgroup/cpu.max)
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
  quota=$(awk -v q="$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)" \
    '{ if (q <= 0) print 0; else print int(q / $1) }' /sys/fs/cgroup/cpu/cpu.cfs_period_us)
else
  quota=0
fi
[ "${quota:-0}" -gt 0 ] || quota=$(nproc)
ngpu_total=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${quota:-0}" -gt 0 ] && [ "${ngpu_total:-0}" -gt 0 ]; then
  export OMP_NUM_THREADS=$(( quota / ngpu_total > 0 ? quota / ngpu_total : 1 ))
  export MKL_NUM_THREADS=$OMP_NUM_THREADS
fi
echo "[phases] python $PYTHON, OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}, logs in $LOGDIR"

run_job() {  # gpus cfg
  local gpus=$1 cfg=$2 n name rc
  n=$(tr ',' '\n' <<< "$gpus" | wc -l)
  name=$(basename "$cfg" .yaml)
  echo "[phases] $(date -u +%FT%TZ) start $name on GPU $gpus"
  if [ "$n" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=$gpus "$PYTHON" -m torch.distributed.run --standalone \
      --nproc_per_node="$n" scripts/train_guard.py --config "$cfg" "${EXTRA[@]}" \
      > "$LOGDIR/$name.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpus "$PYTHON" scripts/train_guard.py --config "$cfg" "${EXTRA[@]}" \
      > "$LOGDIR/$name.log" 2>&1
  fi
  rc=$?
  echo "[phases] $(date -u +%FT%TZ) end $name exit $rc"
  tail -n 3 "$LOGDIR/$name.log" | sed "s/^/[phases]   $name | /"
}

for phase in $(awk '$1 !~ /^#/ && NF { print $1 }' "$PLAN" | awk '!seen[$0]++'); do
  echo "[phases] $(date -u +%FT%TZ) phase $phase"
  for gpus in $(awk -v p="$phase" '$1 == p { print $2 }' "$PLAN" | awk '!seen[$0]++'); do
    (
      for cfg in $(awk -v p="$phase" -v g="$gpus" '$1 == p && $2 == g { print $3 }' "$PLAN"); do
        run_job "$gpus" "$cfg"
      done
    ) &
  done
  wait
done
echo "[phases] $(date -u +%FT%TZ) plan done"
