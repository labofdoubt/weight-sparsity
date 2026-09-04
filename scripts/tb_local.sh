#!/usr/bin/env bash
# Run TensorBoard on this machine over a local copy of the box's event files.
#
#   bash scripts/tb_local.sh                 # sync, then serve on :6006
#   bash scripts/tb_local.sh --no-sync       # serve what is already local
#   bash scripts/tb_local.sh --port 6007
#
# Why bother, when `ssh -L 16006:localhost:16006` already works: a local copy
# keeps working after the box is stopped or destroyed, survives the SSH session
# dropping, and does not compete with training for the box's CPU. The tunnel is
# still the right tool for a quick look at a live run.
#
# Only event files and small metadata are pulled (~100 MB), never checkpoints.
set -euo pipefail

HOST=${TB_HOST:-174.164.26.93}
PORT_SSH=${TB_SSH_PORT:-45324}
REMOTE=${TB_REMOTE:-/workspace/runs}
LOCAL=${TB_LOCAL:-$HOME/tb-logs/weight-sparsity}
VENV=${TB_VENV:-$HOME/.venvs/tb}
PORT=6006
SYNC=1

while [ $# -gt 0 ]; do
  case "$1" in
    --no-sync) SYNC=0; shift ;;
    --port) PORT="$2"; shift 2 ;;
    --local) LOCAL="$2"; shift 2 ;;
    --remote) REMOTE="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ---- 1. tensorboard, in its own venv ------------------------------------- #
# This machine has no tensorboard and no uv/pipx, and the system Python should
# not be installed into, so it gets a dedicated venv. Built once; reused after.
if [ ! -x "$VENV/bin/tensorboard" ]; then
  echo "[tb] creating $VENV (one-off, ~100 MB download)"
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  "$VENV/bin/python" -m pip install -q tensorboard
fi
echo "[tb] $("$VENV/bin/tensorboard" --version_tb 2>/dev/null || echo tensorboard) in $VENV"

# ---- 2. pull the event files --------------------------------------------- #
if [ "$SYNC" = 1 ]; then
  mkdir -p "$LOCAL"
  echo "[tb] syncing $HOST:$REMOTE -> $LOCAL (events + metadata only)"
  # tar over ssh rather than rsync: macOS ships rsync 2.6.9, and the box may not
  # have rsync at all. --ignore-failed-read so a checkpoint being written
  # concurrently cannot abort the whole transfer.
  ssh -p "$PORT_SSH" "root@$HOST" \
    "cd '$REMOTE' && tar cz --ignore-failed-read \
       \$(find . -type f \\( -name 'events.out.tfevents*' -o -name '*.json' \
          -o -name '*.yaml' -o -name '*.jsonl' -o -name '*.md' \\) -print) 2>/dev/null" \
    | tar xz -C "$LOCAL" 2>&1 | grep -v 'Ignoring unknown extended header' || true
  echo "[tb] local size: $(du -sh "$LOCAL" | cut -f1), $(find "$LOCAL" -name 'events.out.tfevents*' | wc -l | tr -d ' ') event files"
fi

if [ -z "$(find "$LOCAL" -name 'events.out.tfevents*' -print -quit 2>/dev/null)" ]; then
  echo "[tb] no event files under $LOCAL -- run without --no-sync first" >&2
  exit 1
fi

# ---- 3. serve ------------------------------------------------------------- #
# TensorBoard names a run by its path relative to --logdir, so pointing at the
# parent keeps the box's own subdirectory layout (dc/..., lens_1/...) intact and
# runs from different machines stay separate series.
echo "[tb] serving $LOCAL on http://localhost:$PORT   (ctrl-C to stop)"
exec "$VENV/bin/tensorboard" --logdir "$LOCAL" --port "$PORT" --bind_all=false
