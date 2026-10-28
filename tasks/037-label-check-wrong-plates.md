# Check labels of plates run B gets wrong

## Goal
Find label errors among run B's wrong predictions on test and dev.

## Checklist
- [x] Flag suspects: label not a known plate format, or run01/A/B all give the same non-label prediction.
- [x] View every flagged image and record a verdict.

## Done criteria
A review CSV with verdicts; no dataset file edited without user approval.

## Progress log
- 2026-10-28: 17 flagged (8 test, 9 dev) of 300 wrong. Confirmed 4 label errors (type2_146, car_390, mb_33, type2_922), 1 crop cut (mb_275), 6 correct labels misread by models, 6 ambiguous. Review: custom_alpr/outputs/label-review-20261028.csv. Only flagged plates were reviewed; other errors may hide more label mistakes. Dataset not modified.
- 2026-10-28: User approved the label review. Applied to custom_alpr/data/ocr/kaggle-v9-prepared (backup in backup-before-label-fix-20261028/, metadata.json records the fixes): val car_390 BOE90955->30E90955, mb_33 59K722599->59K122599, type2_922 60AA25379->50AA25379; test type2_146 29Z26430->29Z26340; removed cropped-plate mb_275 from test (1203 -> 1202). Train `_lp` twins left unchanged: they are renders of the label text, so their labels match their pixels. Rescored all runs from saved predictions (test/dev): run01 0.828/0.810, A 0.881/0.874, B 0.877/0.876, C 0.896/0.878, D 0.897/0.899; D vs C test p=1.00, dev p<0.001. vast.ai copy of the manifests not yet updated.
