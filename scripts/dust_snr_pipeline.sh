#!/usr/bin/env bash
# One SNR-probe model end to end on one GPU: train 20 steps with checkpoints at
# init and after every DUST step (analysis/dust_snr_train.py), then the
# per-coordinate SNR probe of every checkpoint (analysis/dust_snr_probe.py)
# into /workspace/analysis/dust_snr/<run>, which the analysis watcher uploads.
#
#   bash scripts/dust_snr_pipeline.sh <population> <gpu> [sequences] [chunk]
set -uo pipefail
P=${1:?population}; GPU=${2:?gpu}; SEQ=${3:-4}; CHUNK=${4:-8}
NAME=pa_snr_k32_jall_sg0p2_p$P
cd /workspace/weight-sparsity && source /venv/main/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1
echo "[pipeline] $NAME on GPU $GPU  $(date -u +%FT%TZ)"
python analysis/dust_snr_train.py --config configs/cr_dust/snrprobe/$NAME.yaml --steps 20 \
    > /workspace/runs/$NAME.log 2>&1 || { echo "[pipeline] training FAILED, see /workspace/runs/$NAME.log"; exit 1; }
echo "[pipeline] trained  $(date -u +%FT%TZ); checkpoints: $(ls /workspace/runs/$NAME/ckpt_step*.pt | wc -l)"
mkdir -p /workspace/analysis/dust_snr
python analysis/dust_snr_probe.py --run-dir /workspace/runs/$NAME --out /workspace/analysis/dust_snr/$NAME \
    --sequences $SEQ --chunk $CHUNK > /workspace/analysis/dust_snr/$NAME.probe.log 2>&1 \
    || { echo "[pipeline] probe FAILED, see /workspace/analysis/dust_snr/$NAME.probe.log"; exit 1; }
echo "[pipeline] probed   $(date -u +%FT%TZ): $(ls /workspace/analysis/dust_snr/$NAME/snr_step*.npz | wc -l) files"
