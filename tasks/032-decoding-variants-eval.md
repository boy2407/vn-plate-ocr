# Measure decode-time improvements (regex, constrained, beam, threshold)

## Goal
Without retraining, measure how plate-format checks, constrained decoding, and beam search change exact accuracy and confidence-threshold coverage of the epoch-33 checkpoint on dev and test.

## Checklist
- [x] Derive allowed plate patterns from train labels only.
- [x] Implement greedy, greedy+regex reject, constrained greedy, beam, constrained beam.
- [x] Run on desktop CUDA for dev and test.
- [x] Report exact/CER and coverage/accuracy per threshold (incl. 0.99) per variant.

## Done criteria
A JSON/CSV report per split comparing all variants against the greedy baseline.

## Progress log
- 2026-10-28: Started. Running on desktop `C:\Users\trnghia\vn-plate-ocr` over SSH.
- 2026-10-28: First beam version ignored EOS probability in ranking (exact 0.58/0.08); fixed so ranking includes EOS while confidence still excludes it.
- 2026-10-28: Done. Formats from train (count >= 5): 10 patterns. Dev: all variants exact 0.807-0.808, @0.99 cov 0.741 acc 0.982 (16 wrong) unchanged. Test: greedy 0.826, beam 0.825, @0.99 cov 0.752 acc 0.983 (15 wrong) unchanged. Format check rejects only 2 dev / 1 test plates; beam changes ~41-52 plates but fixes 0. Errors are inside valid formats (e.g. 52LB5552 vs 52L85552 are both valid), so decode-time constraints do not help; retraining (step 2) is needed. ruff passes; pytest not run (no pytest in desktop venv or local env). Reports: custom_alpr/outputs/train-custom-ocr/run-20261028-01/decoding-variants/{dev,test}.
