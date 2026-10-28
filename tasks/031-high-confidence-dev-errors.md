# Extract high-confidence incorrect dev predictions

## Goal
Run the best checkpoint autoregressively on dev and extract incorrect predictions with confidence >= 0.90.

## Checklist
- [x] Evaluate dev and save per-image confidence predictions.
- [x] Copy high-confidence incorrect images with metadata.
- [x] Verify counts and update task tracking.

## Done criteria
Provide the number of dev plates that are incorrect at confidence >= 0.90 and a reviewable image set with metadata.

## Progress log
- 2026-10-28: Started evaluation using the best epoch-33 checkpoint and the established 0.90 confidence threshold.
- 2026-10-28: Evaluated all 1,187 dev images with greedy autoregressive decoding; extracted 100 incorrect predictions at confidence >= 0.90.
