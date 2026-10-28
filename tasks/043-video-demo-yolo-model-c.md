# Task 043: Demo trained YOLO and OCR model C on video clips

## Goal

Use the trained YOLO detector and OCR model C (PARSeq run 04) to read plate text in video clips and export annotated video plus per-frame detections.

## Checklist

- [x] Implement video inference with batched OCR crops and training-matched preprocessing.
- [x] Add CLI usage and document the YOLO/model C checkpoints.
- [x] Run the demo on an available clip and inspect the output.
- [x] Run repository tests and Ruff checks.

## Done criteria

A user can run the detector + model C on a video clip and receive an annotated video and CSV; a sample run succeeds.

## Progress log

- 2026-10-28: Identified YOLO checkpoint in sibling `vn-plate-detector` and OCR model C as `run-20261028-04-parseq-aug-noise/best.pt` (PlateOCRParseq, epoch 55). Existing `PlateReader` only supports CCT-XS ONNX, so the video demo will load model C directly.
- 2026-10-28: Added `custom_alpr/video_demo.py`; it runs YOLO per frame, batches plate crops into model C, and exports annotated MP4, CSV, and JSON.
- 2026-10-28: Ran a 60-frame/15-fps sample made from contiguous highway test frames on MPS. Result: 212 detections, 6.45 s processing; outputs are under `custom_alpr/outputs/video-demo/`.
- 2026-10-28: Repository tests passed (33); Ruff check and format check passed using detector venv's Ruff installation.
