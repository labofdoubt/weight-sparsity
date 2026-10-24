#!/bin/bash
# MNIST DUST SNR viewer (streamlit, toy_mnist/snr_viewer.py).  Bound to
# 127.0.0.1 on purpose: reached by SSH local forwarding (-L 8502:localhost:8502),
# so there is no open port and no Caddy token to manage.  Its data, the 17
# snr.npz of the 2026-10-23 sweep, is committed under toy_mnist/reports/snr_sweep,
# so a checkout is all it needs (no Drive restore).
#
# Install on a vast.ai box (docs/streamlits.md):
#   bash scripts/streamlit_install.sh install mnist_snr
# SNR_ROOT and STREAMLIT_SERVER_PORT may be preset in the supervisor conf's
# environment= line; the values below are the defaults.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

source /venv/main/bin/activate
export SNR_ROOT="${SNR_ROOT:-/workspace/weight-sparsity/toy_mnist/reports/snr_sweep}"
export STREAMLIT_SERVER_ADDRESS=127.0.0.1
export STREAMLIT_SERVER_PORT="${STREAMLIT_SERVER_PORT:-8502}"
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_THEME_BASE=light
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll
cd /workspace/weight-sparsity
pty streamlit run toy_mnist/snr_viewer.py -- --root "$SNR_ROOT" 2>&1
