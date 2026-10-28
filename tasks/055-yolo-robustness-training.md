# Task 055: Train paired YOLO robustness experiments

## Goal
Train and compare moderate augmentation and control on the rented GPU.

## Checklist
- [x] Wait for existing OCR training and held-out test to finish.
- [x] Validate both GPU smoke runs.
- [x] Finish moderate augmentation run (30 epochs).
- [x] Finish control run (30 epochs).
- [ ] Back up checkpoints and compare clean/stress validation.

## Done criteria
Both experiments complete with backed-up checkpoints and validation results.

## Progress log
- 2026-10-28: Running on root@47.164.117.172 port 43353, /workspace/vn-plate-yolo. Moderate run completed epoch 7/30 (~72–75 seconds/epoch); control queued afterward. Input 640, batch 16, 9,249 train images. Supervisor yolo-robustness-queue running.

- 2026-10-28: Moderate completed 30/30 in 36.95 minutes. Best clean validation P=98.98%, R=99.09%, mAP50=99.47%, mAP50-95=82.22%. Control completed 23/30 and remains running. Started rsync backup of completed moderate run. These are validation metrics, not test or robustness comparison results.

- 2026-10-28: Both runs complete. Started held-out clean test evaluation of both frozen best checkpoints at 640px, without tuning on test. Validation/stress comparison still pending.

- 2026-10-28: Clean held-out test completed for both frozen best checkpoints. Moderate: P=99.3456%, R=98.5796%, AP50=99.4729%, AP50-95=83.6433%. Control: P=98.9393%, R=98.9405%, AP50=99.4782%, AP50-95=83.4714%. Small clean AP gain (+0.172 percentage points), higher precision/lower recall; corruption stress evaluation remains pending. P/R are Ultralytics summary operating-point metrics, not fixed conf=0.40 counts.
