# Summarize character errors per plate on test

## Goal
Compute the distribution of Levenshtein character errors per test plate from the saved greedy predictions.

## Checklist
- [x] Calculate edit distance for each reference/prediction pair.
- [x] Summarize count and percentage by error count.
- [x] Save the distribution and per-plate distances with the evaluation artifacts.
- [x] Update project tracking and report the result.

## Done criteria
The user receives an error-count histogram for the 1,203 test plates, including exact reads.

## Progress log
- 2026-10-28: Per-image greedy predictions are available; previous report only contained aggregate exact accuracy, CER, and character TP/FP/FN.
- 2026-10-28: Computed Levenshtein errors for all 1,203 test plates. Exact reads: 994; mean 0.505 errors/plate; median 0; maximum 9. Saved the histogram JSON and per-plate CSV beside the checkpoint.
