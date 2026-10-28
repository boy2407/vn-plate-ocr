#!/usr/bin/env bash
set -euo pipefail
YOLO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
YOLO_BASE_PYTHON="${YOLO_BASE_PYTHON:-/venv/main/bin/python}"
cd "$YOLO_ROOT"
exec "$YOLO_BASE_PYTHON" -u -m custom_alpr.yolo_vast.wait_for --root "$YOLO_ROOT" "$@"
