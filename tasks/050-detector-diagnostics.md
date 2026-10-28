# Task 050: Diagnose YOLO quality and compare inference resolutions

## Goal

Measure detector precision/recall and speed at fixed inference resolutions on labeled validation images, inspect the actual checkpoint configuration, and provide evidence-based web controls.

## Checklist

- [x] Inspect training config and existing operating-point results.
- [x] Compare 640 / 960 / 1280 inference on a fixed labeled validation sample.
- [x] Add explicit resolution selection and record the setting per job.
- [x] Document measured tradeoffs and limits on unlabelled uploaded clips.
- [x] Run integration checks and restart the website.

## Done criteria

Resolution behavior is explicit and measured; no unsupported claim that filtering or resolution alone fixes the detector's domain errors.

## Progress log

- 2026-10-28: Started. Copied `args.yaml` reports training size 320; current web prediction implicitly uses Ultralytics' 640 default. Inspect checkpoint metadata to confirm.
- 2026-10-28: Checkpoint train arguments confirm 320. On fixed 192 validation images / 483 GT boxes, seed 20261028, conf .25, NMS .7 and match IoU .5: P/R 320=89.85/97.10%, 640=94.34/96.69%, 960=89.07/92.75%, 1280=77.50/83.44%. Kept 640; higher input resolution is not automatically better for these weights.
- 2026-10-28: At 640/conf .40, P/R=95.46/95.86%; web default now .40 with explicit size selector and per-job recording. Historical results keep their original settings. CLI has `--imgsz`. Metrics are validation sample diagnostics, not proof of clip accuracy.
- 2026-10-28: Verified current YOLO dataset includes exact pixel hashes of Kaggle `duydieunguyen/licenseplates` v1: 3,592 train / 449 validation / 450 test, unchanged splits. OCR model C uses a separate Kaggle OCR dataset.
- 2026-10-28: Visually reviewed 12 time samples across two uploaded clips. Examples: clip 4K at ~5.56s crops a person's shirt; ~2.48s includes body lettering and actual plate together. Clip 1080p correctly tracks the yellow `51H21510`, but later blur affects OCR. Contact sheet: `custom_alpr/outputs/web-demo/detector-clip-review.jpg`.
- 2026-10-28: 66 tests and integration checks passed; backend restarted with completed-job recovery. A robustness training plan is recorded in task 051; no new training has launched and the existing detector's domain errors remain to be addressed by that work.
