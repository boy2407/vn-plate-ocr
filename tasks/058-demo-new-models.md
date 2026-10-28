# Task 058: Use augmented YOLO and OCR D in demo

## Goal
Switch demo defaults and running server to new checkpoints.

## Checklist
- [x] Update defaults and visible model labels.
- [x] Run checks and actual inference.
- [x] Restart web demo.

## Done criteria
Demo serves successfully and infers with augmented YOLO and OCR D.

## Progress log
- 2026-10-28: Started; both checkpoints available locally.

- 2026-10-28: Completed. Both demo defaults point to moderate-s42/best.pt and OCR run D/best.pt; visible labels updated. 96 tests and Ruff passed. Actual sample-frame inference succeeded with both models. Restarted server on 127.0.0.1:8766 and verified HTTP page labels.
