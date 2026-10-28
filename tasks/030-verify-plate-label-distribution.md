# Verify `99A99999` share in OCR training labels

## Goal
Check whether labels matching the normalized pattern `99A99999` account for 54% of the 9,541 training examples.

## Checklist
- [x] Inspect the prepared training manifest and count labels matching the pattern.
- [x] Report the count, percentage, and interpretation of the pattern.

## Done criteria
Give the exact count and percentage over the training set, with the matching rule stated. **Met:** 5,169 of 9,541 labels (54.18%) match `^\d{2}[A-Z]\d{5}$`.

## Progress log
- 2026-10-28: Counted the 9,541 rows in `train.csv`; 5,169 labels match the eight-character structure of two digits, one uppercase Latin letter, then five digits (54.18%).
