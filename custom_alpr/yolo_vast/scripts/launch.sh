#!/usr/bin/env bash
set -euo pipefail
YOLO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
YOLO_STAGE="${1:-smoke}"
case "$YOLO_STAGE" in smoke|pair|control|moderate) ;; *) printf 'Stage must be smoke/pair/control/moderate\n' >&2; exit 2 ;; esac
YOLO_PROGRAM="yolo-robustness-$YOLO_STAGE"
mkdir -p "$YOLO_ROOT/logs"
if ! command -v supervisorctl >/dev/null; then
  printf 'supervisorctl unavailable; run run_stage.sh in tmux instead.\n' >&2
  exit 1
fi
if supervisorctl status "$YOLO_PROGRAM" 2>/dev/null | grep -q RUNNING; then
  printf 'Already running: %s\n' "$YOLO_PROGRAM"; exit 1
fi
cat > "/etc/supervisor/conf.d/$YOLO_PROGRAM.conf" <<EOF
[program:$YOLO_PROGRAM]
directory=$YOLO_ROOT
command=/bin/bash $YOLO_ROOT/custom_alpr/yolo_vast/scripts/run_stage.sh $YOLO_STAGE
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stopsignal=INT
stopwaitsecs=120
redirect_stderr=true
stdout_logfile=$YOLO_ROOT/logs/$YOLO_PROGRAM.log
stdout_logfile_maxbytes=0
EOF
supervisorctl reread
supervisorctl update "$YOLO_PROGRAM"
supervisorctl start "$YOLO_PROGRAM"
supervisorctl status "$YOLO_PROGRAM"
printf 'Log: %s/logs/%s.log\n' "$YOLO_ROOT" "$YOLO_PROGRAM"
