# Task 062: Fine-tune FastALPR YOLOv9-t 384 with project augmentation

## Goal
Fine-tune the actual FastALPR YOLOv9-t 384 plate checkpoint on the exact YOLOv8 dataset/split and moderate corruption policy, then export and compare frozen test results.

## Checklist
- [x] Obtain corresponding trainable checkpoint, pin official architecture/trainer and verify baseline parity.
- [x] Reuse 9,249 train images / 25,364 boxes, excluding 580 unreviewed empty-label images; preserve val/test.
- [x] Integrate same moderate image corruptions and geometry/Mosaic policy at 384px.
- [x] Run meaningful label/gradient checks and GPU smoke training.
- [x] Start full 30-epoch fine-tune safely after current GPU jobs, with persistent logs/checkpoints.
- [x] Export trained model, evaluate untouched test, back up results.

## Done criteria
Correct YOLOv9-t 384 checkpoint fine-tuned, exported and evaluated with artifacts safely copied locally. Queue/launch alone is not completion.

## Progress log
- 2026-10-28: Started at user request. Existing Vast host root@47.164.117.172 port43353. Preserve ongoing OCR jobs; use separate directory/environment. Need native YOLOv9 trainer rather than silently substituting an Ultralytics YOLOv9-c model.

- 2026-10-28: Downloaded official yolo-v9-t-384-license-plates.pt, SHA256 4d525da3ebe44fb6e6bfe8b05ca717b777db3feb2b44b472250f8b32f4754fdb. Pinned WongKinYiu/yolov9 at 5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff. Correct DDetect single-class converted model with 1,913,443 parameters and 221 BN layers; all 1339/1339 state items transfer. PT vs release ONNX parity passes on 8 validation images (<0.0001px input-space error).
- 2026-10-28: Native adapter injects the exact existing corruption budget after geometry/HSV/flips, preserves labels, disables implicit Albumentations, and uses a shared epoch flag to close Mosaic in worker copies for the final 10 epochs. Dataset hashes and exact quarantined train list verified.
- 2026-10-28: CPU smoke found destructive native SGD bias warmup default 0.1 with AdamW; changed warmup_bias_lr to 0.0 for fine-tuning. Added compatibility for removed NumPy trapz (equivalent trapezoid) and pinned setuptools 80.9.0 for legacy pkg_resources. CPU smoke completed on 32 train / 16 val with finite losses and AP50 .678 (smoke subset only, no improvement claim). Failed trial artifacts retained in separate directories.
- 2026-10-28: Upstream end-to-end export stub has missing random import and incompatible box dimensions. Added a tested single-class Nx7 ONNX NMS wrapper with common IoU .45 / conf .001 / max100 controls. Real CPU smoke export and 8-image PT/ONNX parity pass. Supervisor fastalpr-yolov9t-queue launched; it verifies prior OCR train+test completion and GPU idle, then GPU smoke -> full30 epochs -> CPU export/test. App heartbeat theo-d-i-fine-tune-yolov9-t-384 checks every 15 minutes, backs up final results and disables itself after completion. Demo unchanged.

- 2026-10-28: Both CPU and real GPU smoke runs completed with finite training losses and saved optimizer checkpoints. GPU smoke AP50=.6117 on just 16 validation images (not a final score). Full GPU training verified in epoch 1/30 at batch 336/579; batch16, imgsz384, 4 workers, AdamW lr0 .001, warmup_bias_lr0. Sampled corruption counters for 5,120 images: clean 2,033 / one effect 2,310 / two effects 777, confirming the requested on-the-fly augmentation is active. Full source tests: 112 passed, Ruff passed. Heartbeat will finish backup/test reporting; task remains doing until all done criteria are met.

- 2026-10-28: First full epoch completed (1/30) with finite losses and validation AP50=.9191, AP50-95=.5282; these are preliminary validation metrics, not test or evidence of final superiority. Keep task doing. Timing caution for final report: future single-model CPU evaluation and earlier interleaved baseline timings have different host load; rebenchmark together before claiming a speed improvement.

- 2026-10-28 21:36 Asia/Saigon: Heartbeat verified healthy training, 7/30 epochs complete and epoch8 underway. Preliminary validation AP50=.9576/AP50-95=.6644, not final test results. Backed up immutable epoch6.pt (after seven completed epochs) locally to outputs/fastalpr-yolov9t-20261028/intermediate; local/remote SHA256 both a3d9127696a4452652577f243a14bf6f23edf8d76a2a7cb81583f8c754efe864. No code/config/data or process changes.

- 2026-10-28 21:51 Asia/Saigon: Healthy training verified at 14/30 completed. Backed up immutable epoch13.pt locally; local/remote SHA256 both 1e87aeb0e282a832ac903f834369f8b1ecdd90f9c30ea327a107e8896e0c5b08. Task remains doing; no code/config/data changes.

- 2026-10-28 22:10 Asia/Saigon: Healthy at 24/30 completed, epoch25 underway. Backed up immutable epoch23.pt; matching local/remote SHA256 b826639b318c7de2f0118797676981addfd90d358d7efed1f0278e86a92f0fd9. Actual worker counters confirm 43,008 samples with Mosaic disabled after the close boundary; corruption budget still active (227,328 sampled observations, ~40% clean /45% one /15% two effects). Preliminary validation AP50-95=.7136 is not a final test result. No process/code/config/data changes.

- 2026-10-28: COMPLETED. Native fine-tune finished all 30 epochs; queue completed export and held-out test at 2026-10-28T15:24:37Z (22:24 Asia/Saigon). Common protocol: 1,228 images / 2,926 boxes, ONNX CPU FP32, letterbox at 384, NMS .45, max100, confidence floor .001; fixed counts at conf .40 / match IoU .50. Fine-tuned YOLOv9-t384 test AP50=96.3637%, AP50-95=68.9249%, TP/FP/FN=2509/63/417, precision=97.5505%, recall=85.7485%. Original Fast384 AP50-95=43.6250%, FP/FN=30/1361; augmented YOLOv8n640 AP50-95=82.9403%, FP/FN=94/26. Fine-tuning gains +25.30 AP50-95 percentage points over Fast384; YOLOv8n still has higher AP/recall on this test. Do not compare separate-run CPU speed measurements as controlled benchmarks.
- 2026-10-28: Final backup completed at custom_alpr/outputs/fastalpr-yolov9t-20261028/final, including best.pt, last.pt, both stripped variants, ONNX, CSV, experiment/result metadata, counters, export parity and all test predictions/metrics. Verified all five weight hashes against remote final-hashes.json; best.pt=7fd1c3554e0e68f9279e882fe65d4ab0ccee682611522fad8a62c103a5435caf, ONNX=83be8f116508fcb0d98db3ca1b4e16770695a9d5e4764cdadbc207789ab3da3d. Verified 1,228 prediction records / 2,926 targets, identical baseline manifest hash, and 8 PT/ONNX parity cases (<0.0001px). Logs/configs and actual source modules backed up; upstream provenance has pinned commit and no tracked edits. Local verified-summary.json records checks. Demo remains augmented YOLOv8n with OCR E and five-frame averaging, as separately authorized. Task is done; recurring follow-up should be deleted.
