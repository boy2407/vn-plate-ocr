# Analyze confusion, layout, confidence, and speed

## Goal
Extend the held-out test evaluation with a character confusion matrix, one-row/two-row breakdown, confidence-threshold exact accuracy, and inference speed.

## Checklist
- [x] Compute the character confusion matrix and insertion/deletion counts.
- [x] Determine whether reliable one-row/two-row ground truth is available; report only supportable subgroup metrics.
- [x] Generate sequence confidence and exact-accuracy/coverage by threshold.
- [x] Benchmark batch-1 latency and batched throughput on desktop CUDA.
- [x] Save reports, document methodology/limitations, and update task status.

## Done criteria
The user receives reproducible reports for all requested measurements, with any unavailable grouping called out rather than guessed.

## Progress log
- 2026-10-28: Existing test artifacts include greedy predictions but no token scores, confusion alignment, or timing. Kaggle `Type` is explicitly not a reliable plate-row label.
- 2026-10-28: CUDA evaluation completed for all 1,203 test plates. Confusion alignment: 539 substitutions, 51 insertions, 17 deletions. Confidence threshold 0.90 yielded 92.55% exact accuracy at 88.11% coverage. Speed: batch-1 median 16.73 ms, p95 18.31 ms; batch-16 throughput 635.74 plates/s. Layout accuracy remains unavailable without ground-truth single/double labels; generated a 1,203-row annotation template.
