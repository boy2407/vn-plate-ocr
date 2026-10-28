# Extract model C incorrect test images

## Goal
Copy all test images where run C's greedy autoregressive prediction differs from the test reference, with useful metadata.

## Checklist
- [x] Copy incorrect images and write CSV metadata.
- [x] Verify image and row counts; update task status.

## Done criteria
All run C incorrect test images are available for visual review with reference, prediction, confidence, and character edit distance.

## Progress log
- 2026-10-28: Started extraction from run C (`run-20261028-04-parseq-aug-noise`) test predictions.
- 2026-10-28: Extracted all 127 incorrect predictions from 1,203 test images; created CSV metadata and an HTML gallery.
