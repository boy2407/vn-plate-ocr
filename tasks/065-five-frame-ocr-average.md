# Task 065: Combine OCR results over five recent observations

## Goal
Try OCR E temporal consensus over the latest five processed observations in each spatial track, reporting mean confidence of observations that match the winning text. Preserve original per-frame predictions.

## Checklist
- [x] Add bounded five-observation windows and mean confidence, with best matching crop evidence.
- [x] Apply selected mode in live video/overlay and completed gallery/CSV.
- [x] Keep maximum-confidence and whole-track consensus alternatives.
- [x] Verify mixed OCR strings, short tracks, track boundaries and actual OCR E web inference.

## Done criteria
Live and recorded demo support five-frame aggregation with truthful confidence/crop provenance and raw observations retained. No accuracy gain is claimed without labeled video evaluation.

## Progress log
- 2026-10-28: Started at user request to try a few frames and average results. Interpret as aggregating OCR outputs, not blending image pixels. Use at most five most recent processed detections per spatial track; majority string, mean confidence of matching strings, deterministic tie-break by mean confidence.

- 2026-10-28: Completed. Added bounded deque of five actual processed detections per spatial track, majority string with mean-confidence tie-break, average only scores matching the winning text, representative crop original confidence stored separately. Fresh tracks start empty; text-deduplicated gallery retains a chosen track window and never mixes averages across tracks. Selected mode applies to live preview/inspector and exported video; raw per-frame predictions remain intact. Gallery/CSV supports average, maximum OCR confidence and whole-track consensus. HTTP sample video: 60 raw observations, 22 inspected live updates, averages and matching crop provenance verified for all gallery results; maximum available window in that short real clip was four. Synthetic unit tests validate full five-observation eviction/mixed strings/track boundaries. Actual HTTP image uses one observation correctly. 116 tests, Ruff and JS syntax pass. No labeled-video accuracy improvement claimed.
