#!/usr/bin/env bash
# Data-parallel guarded training: one process per GPU via torchrun, every
# process running scripts/train_guard.py (rank 0 logs/validates/checkpoints
# and hosts the divergence guard; stops are all-reduced so every rank exits
# together -- see train()'s should_stop).
#
#   NGPU=8 bash scripts/train_ddp.sh --config configs/fineweb_rbk_500m.yaml \
#       --train.run_name=my_run [--activation_bottleneck.k=128 ...]
#
# batch_size / micro_batch_size / grad_accum_steps stay PER-RANK semantics:
# global tokens per step = world * batch_size * seq_len.
set -euo pipefail
NGPU=${NGPU:-8}
PYTHON=${PYTHON:-}
if [ -z "$PYTHON" ]; then
  if [ -x /venv/main/bin/python ]; then PYTHON=/venv/main/bin/python
  else PYTHON=$(command -v python); fi
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" \
    "$HERE/train_guard.py" "$@"
