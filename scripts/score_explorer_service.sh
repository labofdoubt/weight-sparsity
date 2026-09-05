#!/bin/bash
# Bottleneck score explorer (streamlit).  Bound to 127.0.0.1 on purpose: it is
# reached by SSH local forwarding, so there is no open port and no Caddy token
# to manage.  No exit_portal.sh guard -- this service has no portal.yaml entry.
#
# Install on a fresh vast.ai box (see analysis/README.md, "Surviving instance
# destruction"):
#   cp scripts/score_explorer_service.sh /opt/supervisor-scripts/score_explorer.sh
#   cp scripts/score_explorer.supervisor.conf /etc/supervisor/conf.d/score_explorer.conf
#   supervisorctl reread && supervisorctl update
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

source /venv/main/bin/activate
export SCORES_DIR=/workspace/analysis/scores
export PROBE_DIR=/workspace/analysis/probe
export WNORM_DIR=/workspace/analysis/wnorm
export STREAMLIT_SERVER_ADDRESS=127.0.0.1
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_THEME_BASE=light
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll
cd /workspace/weight-sparsity
pty streamlit run analysis/score_explorer.py 2>&1
