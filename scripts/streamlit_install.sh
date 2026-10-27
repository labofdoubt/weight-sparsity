#!/bin/bash
# Install, inspect or stop one of the repo's streamlit viewers as a supervisor
# service on a vast.ai box (docs/streamlits.md lists the viewers, their data
# and ports).  Run from the repo directory on the box, after the viewer's data
# is in place:
#
#   bash scripts/streamlit_install.sh install <viewer> [KEY=VALUE ...]
#        copies the service script + conf into supervisor, starts it, prints
#        the tunnel line; each KEY=VALUE is added to the service's environment
#        (data roots, STREAMLIT_SERVER_PORT), e.g.
#        install dust_snr DUST_SNR_ROOT=/workspace/viewers/dust_snr
#   bash scripts/streamlit_install.sh status  <viewer>
#   bash scripts/streamlit_install.sh log     <viewer>   # last 40 lines
#   bash scripts/streamlit_install.sh restart <viewer>   # after a git pull
#   bash scripts/streamlit_install.sh stop    <viewer>
#
# <viewer> is the stem of scripts/<viewer>_service.sh and
# scripts/<viewer>.supervisor.conf: score_explorer, dust_snr, mnist_snr.
set -euo pipefail
action="${1:-}"; viewer="${2:-}"
shift $(( $# < 2 ? $# : 2 ))
here="$(cd "$(dirname "$0")" && pwd)"
service="$here/${viewer}_service.sh"
conf="$here/${viewer}.supervisor.conf"
if [ -z "$action" ] || [ -z "$viewer" ] || [ ! -f "$service" ] || [ ! -f "$conf" ]; then
    echo "usage: $0 install|status|log|restart|stop <viewer>" >&2
    echo "viewers: $(ls "$here"/*_service.sh | xargs -n1 basename | sed 's/_service\.sh$//' | tr '\n' ' ')" >&2
    exit 2
fi
port="$(grep -oE 'STREAMLIT_SERVER_PORT:-[0-9]+' "$service" | grep -oE '[0-9]+$')"
extra_env=""
for kv in "$@"; do
    if [ "$action" != install ]; then
        echo "KEY=VALUE arguments are for install only" >&2; exit 2
    fi
    key="${kv%%=*}"; value="${kv#*=}"
    if ! [[ "$kv" == *=* && "$key" =~ ^[A-Z_][A-Z0-9_]*$ && "$value" =~ ^[A-Za-z0-9_./:@+-]+$ ]]; then
        echo "bad environment argument: $kv (KEY=VALUE; VALUE a path, port or name)" >&2
        exit 2
    fi
    extra_env="${extra_env},${key}=\"${value}\""
    [ "$key" = STREAMLIT_SERVER_PORT ] && port="$value"
done
case "$action" in
    install)
        cp "$service" "/opt/supervisor-scripts/${viewer}.sh"
        chmod +x "/opt/supervisor-scripts/${viewer}.sh"
        # the conf's environment= line gains the KEY=VALUE arguments
        sed "s|^environment=\(.*\)\$|environment=\1${extra_env}|" "$conf" \
            > "/etc/supervisor/conf.d/${viewer}.conf"
        grep '^environment=' "/etc/supervisor/conf.d/${viewer}.conf"
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
