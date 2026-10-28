# Task 060: Review FastALPR and select clearer temporal representatives

## Goal
Review FastALPR and improve the existing spatial tracker/temporal OCR consensus with image-quality-aware crop selection.

## Checklist
- [x] Read upstream ALPR, detector and OCR sources.
- [x] Score original crops for sharpness, contrast and exposure.
- [x] Preserve observed majority text and choose a matching clearer crop.
- [x] Verify raw observations and text/crop alignment remain available.

## Done criteria
Source findings documented, representative selection considers image quality and temporal tests pass.

## Progress log
- 2026-10-28: User approved tracking + best crop + temporal OCR. Upstream https://github.com/ankandrew/fast-alpr/blob/master/fast_alpr/alpr.py predict uses per-image detection, bounded box crop and OCR; it contains no multi-frame tracker or consensus. Existing demo has simple spatial association and majority text voting; this task improves representatives without replacing models or changing one-box-per-frame behavior.

- 2026-10-28: User additionally requested trying OCR D confidence selection. Added separate review mode choosing highest-confidence observed OCR per spatial track; consensus+quality mode stays available. Switching modes reuses raw observations and needs no model rerun; CSV follows selected mode.

- 2026-10-28: Completed. Gallery offers OCR-confidence mode (default UI) and consensus+quality mode. Highest-confidence minority selection and matching observed crop are tested; consensus retains majority. Original raw results stay unchanged, selected CSV matches selected mode, and invalid modes are rejected. FastALPR remains a source reference; no extra model or dependency installed. Existing association is lightweight spatial linking, not full multi-object identity tracking. Actual HTTP checks and 106 tests pass.
