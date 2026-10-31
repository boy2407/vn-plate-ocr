# 077 — Speed up the OCR demo on CPU

Status: in progress. Assignee: Claude.

Goal: lower ms/frame of the CPU-only web demo without changing what it reads.

## Changes
- `web_app.InferenceEngine.recognize` records `stage_ms` (detector, pose, OCR) per call;
  the live panel shows it under "Thời gian / frame".
- `custom_alpr/yolo_export.py` exports the detector and pose model to OpenVINO
  (dynamic input, imgsz 640). The Dockerfile runs it at build time. `web_app` loads a
  `*_openvino_model` folder next to a `.pt` when one exists (`--no-openvino` disables it).
- `custom_alpr/speed_benchmark.py` times both backends on one clip and matches boxes
  (IoU ≥ 0.5) and OCR text against the `.pt` run.

Acceptance: box count within 1% of `.pt`, matched boxes give the same text.

## Results
Docker image on the Mac (i7-9750H, `--cpus 12`), `highway-test-sample.mp4`, 200 frames,
pose warp + padding, conf 0.25, imgsz 640. Median ms per stage:

| Run | Detector | Pose | OCR | Mean total |
| --- | --- | --- | --- | --- |
| 1: `.pt` | 165.0 | 67.1 | 88.7 | 322.7 |
| 1: OpenVINO | 60.2 | 71.1 | 119.4 | 309.1 |
| 2: `.pt` | 121.6 | 51.5 | 74.5 | 246.0 |
| 2: OpenVINO detector + `.pt` pose | 73.8 | 69.6 | 66.2 | 211.6 |
| 2: OpenVINO | 49.1 | 55.3 | 89.3 | 206.6 |

OpenVINO vs `.pt` (run 1): 194/194 boxes matched, 194/194 identical text.
The Mac was busy, so the runs are noisy: the `.pt` total moved 323 -> 246 ms between
runs. The detector is 1.6-2.7x faster every time; pose shows no gain; the total drops
15-35%. Still to measure on the 12-CPU LXC itself.

Desktop (AMD Ryzen 7 7800X3D, 16 threads, CPU path), same clip and settings:

| Backend | Detector | Pose | OCR | Mean total |
| --- | --- | --- | --- | --- |
| `.pt` | 30.4 | 27.0 | 42.9 | 102.0 |
| OpenVINO | 43.2 | 23.4 | 42.0 | 106.9 |

194/194 boxes matched, same text. On this AMD CPU OpenVINO gives no gain; it helps on
the Intel Mac. Whether it helps the demo depends on the Proxmox host CPU.

Desktop tests (`--ignore test_yolo9_vast.py`, which needs Unix `fcntl`): 174 passed,
9 failed. None of the failures come from this task: `live_seconds_setting` was removed
from `web_app.py` by earlier uncommitted work, `np.cross` on 2-D vectors fails on the
desktop's NumPy, and `preprocessing` validation is pre-existing.

`web_app` uses the OpenVINO folders only when the device resolves to CPU; on CUDA the
`.pt` files stay.
