#!/usr/bin/env bash
set -euo pipefail
YOLO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
YOLO_PRIOR_PID="${1:?prior PID required}"
YOLO_PRIOR_MATCH="${2:?prior run-name required}"
YOLO_PRIOR_LOG="${3:?absolute prior log path required}"
YOLO_STAGE="${4:-pair}"
[[ "$YOLO_PRIOR_PID" =~ ^[0-9]+$ ]] || exit 2
[[ "$YOLO_PRIOR_MATCH" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
[[ "$YOLO_PRIOR_LOG" =~ ^/[A-Za-z0-9/._-]+$ ]] || exit 2
case "$YOLO_STAGE" in pair|moderate|control) ;; *) exit 2 ;; esac
mkdir -p "$YOLO_ROOT/logs"
if supervisorctl status yolo-robustness-queue 2>/dev/null | grep -q RUNNING; then
  printf 'Queue already running; refusing to replace it.\n' >&2
  exit 1
fi
cat > /etc/supervisor/conf.d/yolo-robustness-queue.conf <<EOF
[program:yolo-robustness-queue]
directory=$YOLO_ROOT
command=/bin/bash $YOLO_ROOT/custom_alpr/yolo_vast/scripts/queue_after_ocr.sh --prior-pid $YOLO_PRIOR_PID --prior-match $YOLO_PRIOR_MATCH --prior-log $YOLO_PRIOR_LOG --stage $YOLO_STAGE
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stopsignal=INT
stopwaitsecs=120
redirect_stderr=true
stdout_logfile=$YOLO_ROOT/logs/queue.log
stdout_logfile_maxbytes=0
EOF
supervisorctl reread
supervisorctl update yolo-robustness-queue
supervisorctl start yolo-robustness-queue
supervisorctl status yolo-robustness-queue
