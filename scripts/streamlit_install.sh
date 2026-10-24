#!/bin/bash
# Install, inspect or stop one of the repo's streamlit viewers as a supervisor
# service on a vast.ai box (docs/streamlits.md lists the viewers, their data
# and ports).  Run from the repo directory on the box, after the viewer's data
# is in place:
#
#   bash scripts/streamlit_install.sh install <viewer>   # copies the service
#        script + conf into supervisor, starts it, prints the tunnel line
#   bash scripts/streamlit_install.sh status  <viewer>
#   bash scripts/streamlit_install.sh log     <viewer>   # last 40 lines
#   bash scripts/streamlit_install.sh restart <viewer>   # after a git pull
#   bash scripts/streamlit_install.sh stop    <viewer>
#
# <viewer> is the stem of scripts/<viewer>_service.sh and
# scripts/<viewer>.supervisor.conf: score_explorer, dust_snr, mnist_snr.
set -euo pipefail
action="${1:-}"; viewer="${2:-}"
here="$(cd "$(dirname "$0")" && pwd)"
service="$here/${viewer}_service.sh"
conf="$here/${viewer}.supervisor.conf"
if [ -z "$action" ] || [ -z "$viewer" ] || [ ! -f "$service" ] || [ ! -f "$conf" ]; then
    echo "usage: $0 install|status|log|restart|stop <viewer>" >&2
    echo "viewers: $(ls "$here"/*_service.sh | xargs -n1 basename | sed 's/_service\.sh$//' | tr '\n' ' ')" >&2
    exit 2
fi
port="$(grep -oE 'STREAMLIT_SERVER_PORT:-[0-9]+' "$service" | grep -oE '[0-9]+$')"
case "$action" in
    install)
        cp "$service" "/opt/supervisor-scripts/${viewer}.sh"
        chmod +x "/opt/supervisor-scripts/${viewer}.sh"
        cp "$conf" "/etc/supervisor/conf.d/${viewer}.conf"
        supervisorctl reread >/dev/null
        supervisorctl update
        sleep 3
        supervisorctl status "$viewer"
        echo "tunnel: ssh -p <port> root@<host> -L ${port}:localhost:${port}  ->  http://localhost:${port}"
        ;;
    status)  supervisorctl status "$viewer" ;;
    log)     supervisorctl tail -40 "$viewer" ;;
    restart) supervisorctl restart "$viewer" && sleep 3 && supervisorctl status "$viewer" ;;
    stop)    supervisorctl stop "$viewer" ;;
    *) echo "unknown action $action" >&2; exit 2 ;;
esac
