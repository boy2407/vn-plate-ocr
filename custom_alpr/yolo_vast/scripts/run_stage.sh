#!/usr/bin/env bash
set -euo pipefail
YOLO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
YOLO_PYTHON="${YOLO_PYTHON:-$YOLO_ROOT/.venv/bin/python}"
cd "$YOLO_ROOT"
mkdir -p logs
case "${1:-smoke}" in
  smoke)
    "$YOLO_PYTHON" -m custom_alpr.yolo_vast.run --root "$YOLO_ROOT" --arm control --smoke --batch 8
    "$YOLO_PYTHON" -m custom_alpr.yolo_vast.run --root "$YOLO_ROOT" --arm moderate --smoke --batch 8
    ;;
  pair)
    "$YOLO_PYTHON" -m custom_alpr.yolo_vast.run --root "$YOLO_ROOT" --arm moderate --name moderate-s42
    "$YOLO_PYTHON" -m custom_alpr.yolo_vast.run --root "$YOLO_ROOT" --arm control --name control-s42
    ;;
  control|moderate)
    "$YOLO_PYTHON" -m custom_alpr.yolo_vast.run --root "$YOLO_ROOT" --arm "$1" --name "$1-s42"
    ;;
  resume)
    "$YOLO_PYTHON" -m custom_alpr.yolo_vast.run --root "$YOLO_ROOT" --resume "${2:?experiment folder required}"
    ;;
  *) printf 'Unknown stage: %s\n' "$1" >&2; exit 2 ;;
esac
