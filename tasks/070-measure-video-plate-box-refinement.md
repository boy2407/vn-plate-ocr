# 070 — Measure video plate box refinement

**Status:** in progress  
**Started:** 2026-10-30

## Scope

Measure tight YOLO crops, 5% padded crops, and `refine_box` crops at context 0.5 / size 320 on the 101 person-confirmed video labels. Recover original frames by the metadata frame index, verify seeks, and save them under `custom_alpr/data/collected/frames/`. Evaluate E and E-v1, report exact match, correct/wrong changes relative to tight crops, fallback rate, changed-result contact sheets, and CPU frame overhead.

Only integrate refinement into `InferenceEngine.recognize` behind an enabled-by-default switch if the measurement shows a clear improvement. Use a separate YOLO instance for the second pass and add GPU-independent logic tests. Fine-tune a new robust OCR model only after the refinement measurements, with preview and camera-held-out/Kaggle evaluation.

## Findings and blockers

- Read `docs/models/plate-ocr/video-ocr-diagnosis.md`, `custom_alpr/refine_box.py`, and `custom_alpr/refine_eval.py`.
- `refine_box.py` defaults match the requested configuration: context 0.5 and size 320.
- All 101 rows in `confirmed-tight.csv` map to a row in collected metadata. The metadata supplies video ID, frame index, and detector box.
- The E-v1 checkpoint is present at `custom_alpr/outputs/train-custom-ocr/run-20261030-10-parseq-e-v1-video/best.pt`.
- User confirmed E is `custom_alpr/outputs/train-custom-ocr/run-20261029-09-parseq-f-compress/best.pt`. Despite the directory name containing `f`, this is model E used for the 24% baseline; checkpoint identity is resolved.
- Frame recovery and three-way measurement completed; see results below.
- The checkout already contained user changes in `custom_alpr/web_app.py`, web assets, and `deploy/`; these were left untouched.

## Completion record

Measurement stage completed. Per user instruction, stop before editing web_app.py; task remains in progress for conditional integration / training. Required checks executed; see validation limitations below.

## CPU measurement run (2026-10-30)

- `.venv-cpu` uses Python 3.12.16, torch 2.2.2 and numpy 1.26.4. System pip rejected `--user` due to Homebrew externally managed environment; uv was bootstrapped in `/tmp/plate-uv-bootstrap`.
- E and E-v1 loaded through `load_ocr` / `build_model`, and both produced predictions on a blank OCR input. DEFAULT_DETECTOR loaded and predicted on a blank input on CPU.
- Source stream resolution uses `web_app.resolve_stream` (H.264 preferred). First recovered RjeyplZXDCo frames match saved crops pixel for pixel.
- Required pytest command executed with `.venv-cpu/bin/python`: collection fails because `custom_alpr.yolo_vast.corruptions` is absent. Ruff reports 116 errors in the existing tree; the new measurement script passes its own check.

## Measurement results

101/101 source frames recovered across five videos, no stream/seek errors. Exact frame position verified before/after read, and saved crops aligned at the metadata coordinates (minimum correlation 0.923). Two videos match pixels exactly; three differ in encoding, MAE up to 18.97. All three compared variants use the same recovered frame.

| Model | tight | padding 5% | refine | padding fixed/broken | refine fixed/broken |
| --- | --- | --- | --- | --- | --- |
| E | 21/101 | 15/101 | 37/101 | 0/6 | 18/2 |
| E-v1 (seen data) | 75/101 | 84/101 | 62/101 | 10/1 | 6/19 |

Fallback 12/101 (11.9%). CPU refine mean 37.65 ms/frame, median 29.96 ms/frame (101 unique frames, one crop each). E net +16 clears the user threshold of about 10, but E-v1 net −13 prevents assuming improvement for the current demo. No integration or training started.

Original saved tight crops independently reproduce E 25/101 and E-v1 94/101; recovered stream encoding differences materially affect OCR. Main table therefore uses consistent recovered frames for all variants.

Artifacts: `custom_alpr/outputs/refine-confirmed-070/{report.json,rows.json,saved-baseline.json,E-changed-*.png,E-v1-changed-*.png}`. Visually inspected E contact sheet; changed outputs show tighter plate regions, remaining misreads, and fallback examples. Seven E and four E-v1 sheets cover all changed reads.
