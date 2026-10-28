# Task 061: Compare FastALPR detector with augmented YOLO

## Goal
Evaluate the FastALPR default detector and project augmented YOLO on the exact same held-out detection test split, including AP, fixed-threshold false positives/misses, and same-machine latency.

## Checklist
- [x] Inspect/pin upstream runtime and model metadata.
- [x] Run both detectors on all 1,228 test images with common matching/metrics.
- [x] Compare fixed confidence 0.40 and AP50/AP50-95; document model input differences.
- [x] Record same-machine timings, artifacts and results.

## Done criteria
Reproducible results for all test images and clear limitations, with no changes to demo models.

## Progress log
- 2026-10-28: Started at user request. FastALPR default verified from upstream DefaultDetector: yolo-v9-t-384-license-plate-end2end. Include 640 variant if available to separate resolution from model effects; compare project moderate-s42/best.pt at 640. Do not retune on test.

- 2026-10-28: Evaluation running under Supervisor fastalpr-detector-test on Vast CPU. Isolated environment: open-image-models 0.6.0, ORT 1.31.1, ONNX 1.23.2; no GPU use or training environment changes. Upstream graphs have NMS IoU 0.45, score floor 0.001 and max 100 outputs. Project checkpoint exported to ONNX with identical controls; all three use common RGB letterbox, float coordinate restoration and CPU provider. Thread timing selected 16/16/8 threads for project640/Fast384/Fast640. First 100 images completed in ~24 seconds. Comparison is detector-only, no OCR, filters or best-box restriction.

- 2026-10-28: Complete. All three ONNX detectors evaluated on the same verified 1,228 test images / 2,926 GT boxes, all boxes scored (no demo best-box restriction), NMS .45, floor .001, max100, float coordinates, FP32 CPU. Project640 AP50/AP50-95=99.3672%/82.9403%; Fast384=77.1605%/43.6250%; Fast640=95.7899%/61.5569%. At conf .40 / match IoU .50, project TP/FP/FN=2900/94/26, Fast384=1565/30/1361, Fast640=2233/33/693. Mean CPU latencies 92.35/48.31/92.63 ms. Other GPU training jobs shared host, so timing is indicative. Native API parity checks passed on 8 project and 8 Fast sample calls; project ONNX coordinate differences <0.0001px, Fast coordinates identical after native integer conversion. Source PT SHA256 agrees with local deployed model. Full results/predictions/parity/log/hardware backed up locally under custom_alpr/outputs/fastalpr-detector-comparison/. Common square letterbox/NMS protocol differs from earlier native validation; do not mix AP values. 108 tests and Ruff passed. Demo models unchanged.
