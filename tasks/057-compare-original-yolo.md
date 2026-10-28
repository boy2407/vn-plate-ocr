# Task 057: Compare original YOLO with retrained checkpoints

## Goal
Evaluate original initial.pt on the same held-out test at 640px and compare frozen checkpoints.

## Checklist
- [x] Evaluate original checkpoint.
- [x] Save results locally and report comparison.

## Done criteria
Same-split, same-settings test metrics for all three checkpoints are available.

## Progress log
- 2026-10-28: Started at user request. Original checkpoint is bundle weights/initial.pt, copied from desktop-yolov8n-pretrained-20261028/best.pt.

- 2026-10-28: Completed original clean test at 640px on 1,228 images / 2,926 boxes. P=97.6831%, R=96.5400%, AP50=99.0568%, AP50-95=70.4862%. Moderate AP50-95 gain +13.1572 percentage points. Evaluation size differs from original training size (320); all comparison evaluations use 640. No stress robustness claim.
