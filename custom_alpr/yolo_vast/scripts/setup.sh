#!/usr/bin/env bash
set -euo pipefail
YOLO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
YOLO_BASE_PYTHON="${YOLO_BASE_PYTHON:-/venv/main/bin/python}"
YOLO_PYTHON="${YOLO_PYTHON:-$YOLO_ROOT/.venv/bin/python}"
cd "$YOLO_ROOT"
if ! test -x "$YOLO_PYTHON"; then
  "$YOLO_BASE_PYTHON" -m venv --system-site-packages "$YOLO_ROOT/.venv"
fi
if ! "$YOLO_PYTHON" -c 'import torch, torchvision'; then
  "$YOLO_PYTHON" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
fi
"$YOLO_PYTHON" -m pip install -r custom_alpr/yolo_vast/requirements.txt
"$YOLO_PYTHON" -m pytest tests -q
"$YOLO_PYTHON" -m custom_alpr.yolo_vast.run --root "$YOLO_ROOT" --preflight-only
"$YOLO_PYTHON" -m custom_alpr.yolo_vast.preview --root "$YOLO_ROOT"
