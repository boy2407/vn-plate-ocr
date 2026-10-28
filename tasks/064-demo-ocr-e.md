# Task 064: Switch running demo from OCR D to OCR E

## Goal
Use run E (combined synthetic + tilt + small-image augmentation), run-20261028-08-parseq-e-all/best.pt, for new demo inference.

## Checklist
- [x] Update checkpoint defaults and current model labels/docs.
- [x] Verify loading and actual recognition, run required checks.
- [x] Restart live demo and verify an HTTP inference job.

## Done criteria
Demo serves on port 8766 and new inference uses OCR E. Existing results remain historical; YOLO detector selection is unchanged.

## Progress log
- 2026-10-28: Started at user request. Task 063 identifies E as run 08 (combined augmentation), distinct from E1/E2. Checkpoint already downloaded locally.

- 2026-10-28: Completed. Video/web defaults now use run-20261028-08-parseq-e-all/best.pt; visible labels and current-model documentation updated while historical D validation measurements remain identified as D. Restarted localhost:8766. Actual HTTP image and 60-frame sample-video inference completed with checkpoint provenance confirming OCR E and the augmented YOLOv8 checkpoint. 116 tests, Ruff and JS syntax check passed. Existing historical jobs are preserved and require rerunning to obtain E predictions. Evidence: outputs/web-demo/ocr-e-average-validation/.
