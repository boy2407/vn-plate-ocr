# Analyze OCR errors by image size and plate group

## Goal
Find whether small or large plate crops are harder for the two new OCR models, and characterize recurring mistakes.

## Checklist
- [x] Stratify test exact accuracy by source image dimensions.
- [x] Identify recurring character/sequence errors and inspect representative examples.
- [x] Summarize limitations and update task tracking.

## Done criteria
Report comparable size-based metrics for both new runs and concrete, verified error examples.

## Progress log
- 2026-10-28: Started quantitative analysis of the held-out test predictions and source crop dimensions.
- 2026-10-28: Run B exact accuracy by height: <25 px 66.8% (161/241), 25–39 px 82.8% (164/198), >=40 px 95.3% (728/764). Natural photos are 77.7%; clean `_lp` renders 99.4%. Reviewed existing large-image error audit and confusion summary.
