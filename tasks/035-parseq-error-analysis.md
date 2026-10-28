# Error analysis and confusion matrix for run B (PARSeq)

## Goal
Break down the remaining errors of run-20261028-03-parseq-aug on test and dev: character confusion matrix, error types, positions, plate formats, image size, confidence, and suspected label errors.

## Checklist
- [x] Confusion matrix summary (top pairs, letter/digit groups) for test and dev.
- [x] Error breakdown by type, position, format, group, image height, confidence.
- [x] Visual confusion-matrix page and list of high-confidence errors.

## Done criteria
A written breakdown with numbers and a viewable confusion matrix.

## Progress log
- 2026-10-28: Started from existing analysis-test / analysis-val outputs.
- 2026-10-28: Done. Test: 150 wrong (12.5%); substitutions 264 digit->digit, 58 letter->letter, 26 letter<->digit; 17 ins, 15 del. Top pairs 8->9, 8->6, 1->0, 2->0. Errors are almost all on real photos: real 77.7% vs _lp 99.4% (test); real <25 px 66.8%, >=60 px 90.2%. `_lp` files are clean twins of real images with the same label (5290 pairs, often across splits), but real images whose twin is in train score lower, not higher, so no inflation found. Worst groups type2/type3/type7 real (~68-70%). Suspect labels: car_390 BOE90955, mb_275 B537011. Matrix page: custom_alpr/outputs/train-custom-ocr/run-20261028-03-parseq-aug/confusion-matrix.html.
- 2026-10-28: Follow-up: B test errors by height <25: 80, 25-39: 34, >=40: 36 (val 85/27/38). Viewed all 36 test errors >=40 px: 17 readable images misread by the model (8->6/0, 8->B, N->H, H->U, rare R/Z/HR in _lp, blue type4/type6 plates), 5 ambiguous, 13 bad images (blank/blur/tilt/crop), 1 label/crop issue (mb_275).
