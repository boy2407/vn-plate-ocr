# Task 048: Optional denoise and sharpening before OCR

## Goal

Allow light denoising and sharpening of the selected plate crop before OCR, with exact model-input previews alongside original crops. Preserve real time streaming and one-box selection.

## Checklist

- [x] Share deterministic preprocessing between model inputs and preview artifacts.
- [x] Add independent web controls and CLI flags; record per-job settings.
- [x] Show original vs actual OCR input in live view and enlarged review.
- [x] Check tensor/preview equality, disabled path, option validation, and real image/video jobs.
- [x] Measure processing cost and compare a fixed labeled validation sample without tuning on test.
- [x] Run required checks, restart web and open a result.

## Done criteria

Users can enable denoise/sharpen before OCR and inspect exactly which pixels the recognizer receives. Accuracy improvement is reported only if supported by labeled evaluation.

## Progress log

- 2026-10-28: Started. Apply processing to the selected crop after YOLO, keeping detector input as the original frame. Keep controls independent and avoid claiming sharpening recovers missing detail.
- 2026-10-28: Added training-matched bilinear RGB resize, optional light bilateral denoising and unsharp mask, shared by web and CLI. Web records per-result settings and saves both original crop and exact model input PNG; disabled filters preserve the training input pixels. Added side-by-side LIVE/modal images and independent controls, off by default.
- 2026-10-28: Fixed-seed 256-image validation check: original 220 correct, denoise 218, sharpen 220, both 218. CER respectively 0.0434 / 0.0449 / 0.0425 / 0.0444. One preprocessing pass averaged 0.066 / 0.131 / 0.108 / 0.347 ms per crop. Evidence does not justify enabling filtering universally; no test split tuning.
- 2026-10-28: Real HTTP image checks verified settings propagation and exact preview pixels. MP4 with both filters enabled streamed before completion, retained at most one OCR box per processed frame and reached 22.8 FPS updates on the user's 25 FPS / 4K clip (262 processed, 26 skipped). CLI flags succeeded on 60-frame sample.
- 2026-10-28: 57 tests, Ruff check/format, JavaScript syntax and diff checks passed. Restarted server on port 8766; opened and verified comparison dialog in browser. Screenshot and measurements are under `custom_alpr/outputs/web-demo/`.
- 2026-10-28: Review also exposed a high-confidence YOLO crop of a vehicle logo in the supplied clip. Filtering that crop does not fix the detector's incorrect region selection; OCR may still produce a plausible plate string.
