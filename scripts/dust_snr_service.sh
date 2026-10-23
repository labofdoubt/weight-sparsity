#!/bin/bash
# DUST SNR viewer (streamlit, analysis/dust_snr_viewer.py).  Bound to 127.0.0.1
# on purpose: reached by SSH local forwarding (-L 8501:localhost:8501), so
# there is no open port and no Caddy token to manage.
#
# Install on a vast.ai box:
#   cp scripts/dust_snr_service.sh /opt/supervisor-scripts/dust_snr.sh
#   cp scripts/dust_snr.supervisor.conf /etc/supervisor/conf.d/dust_snr.conf
#   supervisorctl reread && supervisorctl update
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

source /venv/main/bin/activate
export DUST_SNR_ROOT=/workspace/analysis/dust_snr
export STREAMLIT_SERVER_ADDRESS=127.0.0.1
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_THEME_BASE=light
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll
cd /workspace/weight-sparsity
pty streamlit run analysis/dust_snr_viewer.py -- --root "$DUST_SNR_ROOT" 2>&1
