# Task 047: Compare global CCT-XS v2 on OCR test split

## Goal

Run `cct-xs-v2-global-model` from fast-plate-ocr on the exact held-out test manifest used by OCR model C (PARSeq run 04), then compare plate exact accuracy and character error rate.

## Checklist

- [x] Run CCT inference on all test crops with RGB input through the model's configured 128x64 preprocessing.
- [x] Compare predictions against model C on the same manifest and report exact accuracy/CER.
- [x] Record limitations, including alphabet/config differences.

## Done criteria

Both recognizers are evaluated on the same 1,203 test samples and the comparison can be reproduced from saved per-image predictions.

## Progress log

- 2026-10-28: Started evaluation; isolated Python environment created and global CCT-XS v2 model/config downloaded.
- 2026-10-28: Evaluated all 1,203 test crops. PARSeq model C: 89.44% exact / 3.73% CER; global CCT-XS v2: 78.72% exact / 9.11% CER. Saved per-image predictions and summary under `custom_alpr/outputs/train-custom-ocr/cct-xs-v2-global-test-comparison/`.
