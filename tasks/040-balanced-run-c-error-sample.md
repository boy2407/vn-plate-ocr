# Create balanced review sample for model C errors

## Goal
Create an easy-to-review balanced subset of run C's incorrect test images across plate groups.

## Checklist
- [x] Sample up to three incorrect images per filename group.
- [x] Create a gallery and CSV with predictions and labels.
- [x] Verify balance and update task tracking.

## Done criteria
An HTML gallery and CSV provide a balanced representative subset of run C test errors.

## Progress log
- 2026-10-28: Started stratified sampling by plate filename family.
- 2026-10-28: Created a 26-image sample: up to three each from car, mb, type1–type7, spread across confidence within each group.
