# 073 — Train plate corner pose and evaluate rectified OCR

Status: done (measurement only). Started/completed: 2026-10-30.

Convert original Kaggle polygons to TL/TR/BR/BL pose labels, preserving detection splits/exclusions, test ordering/flip semantics, and visually review 30 annotations. Train isolated YOLOv8n-pose on desktop GPU for up to 100 epochs with patience. Compare detector box mAP, keypoint errors, E OCR on task071 camera crops, and CPU latency. Select warp margin using Đà Nẵng only before one Sài Gòn evaluation. No integration, old checkpoint changes, or commit.

## Preparation

- Source archive raw: 4857 true quadrilaterals, 286 axis-aligned rectangles, 57 non-four-point polygons. Detection membership preserves its original split, 44 duplicate exclusions, and other source rejections.
- Pose retained: train 3379 images / 3821 plates; val 1117 images / 1278 plates. 54 unsupported polygons remain after detection exclusions and are skipped; 35 images with no supported quadrilateral are omitted rather than taught as background. Rectangles retained: 213 train + 71 val.
- Ordered TL/TR/BR/BL clockwise using atan2 around center, start at the upper edge (lowest midpoint y), invariant under horizontal reflection; kpt_shape [4,3], flip_idx [1,0,3,2]. All visibility=2. Contact sheet of 30 annotations visually inspected.
- Seven data/flip tests and two synthetic warp tests pass without GPU. Full pytest still stops during collection at pre-existing missing custom_alpr.yolo_vast.corruptions.
- Isolated desktop C:/Users/admin/Documents/plate-pose-073; .venv uses installed torch 2.14.1+cu132 / Ultralytics8.4.191, RTX5060 8GB. Pretrained pose and AMP-check weights downloaded on Mac then copied. PYTHONIOENCODING=utf-8 set by launch PS script.
- Final train configuration: 100 epochs, patience20, imgsz640, batch32, AMP, workers4, seed73. Initial FP32 startup and AMP weights-path startup attempts were stopped before using any result; final run selects best by pose validation fitness, never video OCR.

### Corner convention correction before final training

The initial min(x+y) corner start was ambiguous after horizontal flip for 5/5143 source quadrilaterals. Changed to the start of the upper edge in clockwise order; source-wide reflection audit now has 0 mismatches. A skew regression test was added. Rebuilt labels at data/pose-073-v2 and transferred labels to the isolated desktop dataset, clearing only its generated caches. Earlier startup runs were discarded; final valid run is runs/plate-pose-073-5. No video OCR was evaluated during this correction.

## Current detector baseline

Measured independently on all detection val: 1127 images / 1295 boxes, mAP50=0.994656 and mAP50-95=0.888274 on CPU. Checkpoint hash recorded in outputs/pose-073/default-detector-map.json.

Evaluation protocol frozen before video inference: pose conf>=.25; associate maximum-IoU pose box intersecting original box (no high IoU cutoff because original boxes are loose); failed detection/invalid or low-confidence corners count as incorrect/missing, no silent exclusions. Fixed margins 0/5/10% selected on Đà Nẵng exact, then CER, then lower margin. Sài Gòn marker prevents a second run. Pipeline timing includes every full-frame detection/valid warp, excludes OCR.

## Final training and measurement notes

100 epochs completed (4758.34 seconds reported in results.csv); best checkpoint corresponds to epoch91 by combined box+pose mAP50-95 fitness (CSV precision). Final checkpoint copied to outputs/pose-073/best.pt.

A first Đà Nẵng measurement attempt failed before producing scores because YOLO xy keypoints were a noncontiguous NumPy view. Fixed by ascontiguousarray and added a synthetic strided-view regression test. Completed Đà Nẵng run selected margin5% (0/5/10% exact=14/16/15 of37), then Sài Gòn was run once with that margin and identical checkpoint hash. No Sài Gòn rerun or parameter changes.

For box mAP only, DetectionValidator initially interpreted pose extras as class scores (NMS nc inferred from channels). This invalid metric attempt was stopped. PoseBoxValidator now retains detection GT parsing and explicitly selects pose-aware nc for NMS; a one-class/12-extra-coordinate tensor regression test verifies only class0 and all12 extra fields. OCR used the standard PosePredictor and is unaffected; no video inference was repeated for this correction.

## Final measurements

Exact/missing/fixed/broken intervals use Wilson95. CER uses sample bootstrap10000; paired differences use bootstrap100000 seed73. Units are crop_id, not independent vehicles. Đà Nẵng selected the margin; Sài Gòn was run once with that fixed margin/checkpoint.

| Model / detection val | Images | mAP50 | mAP50-95 |
|---|---|---|---|
| current_detector | 1127 | 0.9947 | 0.8883 |
| pose | 1127 | 0.9944 | 0.9026 |

Pose corner median: **1.53 px**, **3.33% GT bounding-box height**, or 3.70% mean quad side height. Matched 1266/1278 = 99.1% [98.4, 99.5], 5064 corners; unique greedy IoU≥.5, conf≥.25. Unmatched GT excluded only from conditional corner error, not hidden: coverage reported here.

Selected margin: **5% per edge**; maximize Đà Nẵng exact, break ties by CER then smaller margin. Width/height use maximum opposite side lengths; getPerspectiveTransform maps TL/TR/BR/BL to the rectangle; margins sample context through the homography (black beyond image boundary).

### E — Đà Nẵng — selection set

| Crop | Exact [Wilson95] | CER [bootstrap95] | Fixed / broken vs (a), both [Wilson95] | Δ pp vs (a) [paired95] | Missing [Wilson95] |
|---|---|---|---|---|---|
| tight | 9/37 = 24.3% [13.4, 40.1] | 30.8% [22.3, 40.2], chars=295 | 0/37 = 0.0% [0.0, 9.4] / 0/37 = 0.0% [0.0, 9.4] | +0.0 [+0.0, +0.0] | 0/37 = 0.0% [0.0, 9.4] |
| refine | 20/37 = 54.1% [38.4, 69.0] | 11.9% [6.8, 18.0], chars=295 | 11/37 = 29.7% [17.5, 45.8] / 0/37 = 0.0% [0.0, 9.4] | +29.7 [+16.2, +45.9] | 0/37 = 0.0% [0.0, 9.4] |
| pose_box | 17/37 = 45.9% [31.0, 61.6] | 22.7% [13.2, 34.1], chars=295 | 8/37 = 21.6% [11.4, 37.2] / 0/37 = 0.0% [0.0, 9.4] | +21.6 [+8.1, +35.1] | 5/37 = 13.5% [5.9, 28.0] |
| warp_0.0 | 14/37 = 37.8% [24.1, 53.9] | 32.9% [21.1, 45.9], chars=295 | 5/37 = 13.5% [5.9, 28.0] / 0/37 = 0.0% [0.0, 9.4] | +13.5 [+2.7, +24.3] | 8/37 = 21.6% [11.4, 37.2] |
| warp_0.05 | 16/37 = 43.2% [28.7, 59.1] | 30.8% [18.9, 44.2], chars=295 | 7/37 = 18.9% [9.5, 34.2] / 0/37 = 0.0% [0.0, 9.4] | +18.9 [+8.1, +32.4] | 8/37 = 21.6% [11.4, 37.2] |
| warp_0.1 | 15/37 = 40.5% [26.3, 56.5] | 31.9% [19.9, 45.3], chars=295 | 6/37 = 16.2% [7.7, 31.1] / 0/37 = 0.0% [0.0, 9.4] | +16.2 [+5.4, +29.7] | 8/37 = 21.6% [11.4, 37.2] |

Selected warp vs (c) refine: fixed 2/37 = 5.4% [1.5, 17.7]; broken 6/37 = 16.2% [7.7, 31.1]; Δ -10.8 [-24.3, +2.7] pp.

CPU pipeline n=37: detector+refine mean/median 216.14/215.26 ms/frame; pose+warp 72.23/72.39 ms/frame. All detections in each full frame, OCR/read/decode excluded; warm models, 4 CPU threads.

### E — Sài Gòn — fixed margin, one run

| Crop | Exact [Wilson95] | CER [bootstrap95] | Fixed / broken vs (a), both [Wilson95] | Δ pp vs (a) [paired95] | Missing [Wilson95] |
|---|---|---|---|---|---|
| tight | 11/58 = 19.0% [10.9, 30.9] | 37.2% [30.0, 44.8], chars=494 | 0/58 = 0.0% [0.0, 6.2] / 0/58 = 0.0% [0.0, 6.2] | +0.0 [+0.0, +0.0] | 0/58 = 0.0% [0.0, 6.2] |
| refine | 15/58 = 25.9% [16.3, 38.4] | 22.7% [16.9, 28.7], chars=494 | 6/58 = 10.3% [4.8, 20.8] / 2/58 = 3.4% [1.0, 11.7] | +6.9 [-1.7, +17.2] | 0/58 = 0.0% [0.0, 6.2] |
| pose_box | 11/58 = 19.0% [10.9, 30.9] | 37.2% [28.7, 46.4], chars=494 | 7/58 = 12.1% [6.0, 22.9] / 7/58 = 12.1% [6.0, 22.9] | +0.0 [-12.1, +12.1] | 10/58 = 17.2% [9.6, 28.9] |
| warp_0.05 | 13/58 = 22.4% [13.6, 34.7] | 35.8% [26.7, 45.5], chars=494 | 8/58 = 13.8% [7.2, 24.9] / 6/58 = 10.3% [4.8, 20.8] | +3.4 [-8.6, +15.5] | 11/58 = 19.0% [10.9, 30.9] |

Selected warp vs (c) refine: fixed 6/58 = 10.3% [4.8, 20.8]; broken 8/58 = 13.8% [7.2, 24.9]; Δ -3.4 [-15.5, +8.6] pp.

CPU pipeline n=58: detector+refine mean/median 128.02/119.70 ms/frame; pose+warp 69.93/66.29 ms/frame. All detections in each full frame, OCR/read/decode excluded; warm models, 4 CPU threads.

### CPU all 95 frames

Detector+refine mean/median **162.34/149.52 ms/frame**; pose+warp **70.82/69.20 ms/frame**. Detector confidence .4, pose .25, keypoint confidence .25; per-frame detection counts saved alongside timing. No concurrent heavy CPU job during measurements.

### Artifacts and validation

`custom_alpr/outputs/pose-073/`: best.pt, report.json, detection-report.json, corner-rows.json, selected-margin.json, saigon-run-started.json, camera rows/reports/timings and changed-result contact sheets, conversion-report.json and labels-contact.png. New files pass Ruff; 11 data/flip/warp/validator tests pass without GPU. Full pytest collection remains blocked by the existing missing custom_alpr.yolo_vast.corruptions. No integration, old checkpoint changes, web/deploy edits or commit.

## Missing cases and decision

Pose box missing: Đà Nẵng5/37, Sài Gòn10/58. Additional keypoint-confidence failures:3/37 and1/58. Total unavailable warp:8/37 and11/58. All count as incorrect, not silently excluded; report.json missing_cases and camera rows list every crop ID/reason/confidence.

Selected warp5% vs refine: Đà Nẵng fixes2 / breaks6, net−4, Δ−10.8pp [−24.3,+2.7]; Sài Gòn fixes6 / breaks8, net−2, Δ−3.4pp [−15.5,+8.6]. Both intervals contain0 and changes are below about10 crops: chưa đủ bằng chứng. Relative to tight box, gains are7/37 and2/58 (Sài Gòn fixed8/broken6), not evidence that pose+warp beats refine.

Observed mAP50-95 is higher on the identical detection val (.9026 vs .8883), and mean CPU pipeline time is lower (70.82 vs162.34ms/frame). Neither establishes higher OCR accuracy on the video domain. Contact sheets of Đà Nẵng labels and Sài Gòn changed reads were visually inspected; examples show valid warps that still confuse characters and missed plates.

**Kết luận: Chưa nên thay detector + refine bằng pose + warp vì chưa đủ bằng chứng cải thiện OCR video, dù nhanh hơn trên CPU.**
