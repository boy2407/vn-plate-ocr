# Count character-level false positives on test

## Goal
Re-evaluate the saved PlateOCR2D checkpoint on the held-out test set and report false-positive character counts with an explicit alignment definition.

## Checklist
- [x] Generate greedy predictions for all test samples on the desktop CUDA device.
- [x] Align predictions against references with LCS and count unmatched predicted/reference characters.
- [x] Report false-positive convention, counts, and denominator.
- [x] Update project tracking with the evaluation result.

## Done criteria
The user receives reproducible test-set character-level FP counts and understands whether extra characters or all incorrect predicted characters are being counted.

## Progress log
- 2026-10-28: Training log contains aggregate CER and exact accuracy but no character-level confusion/FP counts.
- 2026-10-28: Re-evaluated checkpoint epoch 33 on CUDA over 1,203 test samples. LCS counts: TP 9,221, FP 543, FN 509; precision 94.44%, recall 94.77%, F1 94.60%. Exact accuracy 82.63% and CER 6.24% match the original run. Saved the report and per-image predictions with the checkpoint artifacts.
