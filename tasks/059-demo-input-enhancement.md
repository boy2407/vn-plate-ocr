# Task 059: Enhance images immediately before demo inference

## Goal
Add crop margin and conservative perspective rectification immediately before OCR, preserving original pixels and an explicit off mode.

## Checklist
- [x] Add 5% crop margin and guarded perspective rectification.
- [x] Wire upload/video settings and exact OCR previews.
- [x] Validate coordinate mapping, preview fidelity, accuracy sample and latency.
- [x] Restart and verify live demo.

## Done criteria
Demo can process enhanced inputs before both models; original crops stay intact and observed accuracy/runtime limitations are recorded.

## Progress log
- 2026-10-28: Started. User clarified inference-time preprocessing, not dataset preparation or retraining. Enhancement remains opt-in pending measured benefit.

- 2026-10-28: User finalized margin + rectification before OCR. Gamma/CLAHE deferred; YOLO continues receiving original frames. FastALPR source reviewed: direct bounded crop and OCR, no temporal tracking in ALPR.predict.

- 2026-10-28: Completed. Web default adds 5% source-image margin (minimum one pixel, clamped); detector box remains distinct from crop_box. Perspective rectification is opt-in with convexity, size, tilt and edge-support guards, fallback to original crop, and original-resolution warp. Saved OCR PNG matches actual tensor pixels. Final 256-image validation seed 42 with OCR D: 234 correct both modes, CER 0.02117 both, 8 rectified crops, ~0.34 ms preprocessing/crop. Padding accuracy is not established by pre-cropped OCR validation. Actual HTTP image and 60-frame sample-video jobs completed; OCR preview and selected CSV returned successfully. Full checks: 106 tests, Ruff and JS syntax pass. Final server restarted on port 8766. Evidence in custom_alpr/outputs/web-demo/crop-rectification-validation/.
