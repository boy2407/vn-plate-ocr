# 056 Synthetic plates for run E

## Goal
Generate synthetic training plates for every layout (type1..type7), filling letters that are scarce in train and weighting digits run D misreads, so run E can learn them. Synthetic images go to train only.

## Checklist
- [x] Generator `custom_alpr/synth_plates.py`: swap characters inside Kaggle `_lp` renders (same layout colour/border/spacing), 1/2-row detected by aspect ratio, glyphs shared across layouts with stroke matching, gradient-aware background fill, province codes from train labels.
- [x] Letters fill gaps up to 400 plates each (I, J target 40, only 99999A(A) layout); digits 8 6 9 3 2 0 weighted x2.
- [x] Preview 42 plates (clean + augmented) in custom_alpr/outputs/synthetic-preview/.
- [x] User approves preview ("khá ổn").
- [x] Tests, ruff; full set (2,400) + combined train manifest (custom_alpr/data/ocr/synthetic-v1/train-plus-synthetic.csv, 11,941 rows).
- [x] Run E on vast.ai (task 063).

## Done criteria
User-approved synthetic set and combined manifest exist; run E trained and compared with D.

## Progress log
- 2026-10-28: Counts per layout: type1 200, type2 500, type3 300, type4 300, type5 300, type6 500, type7 300 (2,400). Full dry run: Y R U Z X S V N Q P G reach ~400 plates each; I/J got only +1 each (few 99999A(A) templates). Waiting for preview approval.
- 2026-10-28: Preview approved. Generated synthetic-v1 (2,400 plates, 14 MB) and train-plus-synthetic.csv. Added `--strong-downscale` (p=0.5, 8-24 px) and, after angle previews (custom_alpr/outputs/rotation-preview/), `--view-tilt` (rotation +/-12 deg + side/vertical view tilt 10-25%, far half squeezed, shrink-to-fit; user chose 10-25%). 48 tests pass (test_project/test_yolo_vast need ultralytics, not installed locally), ruff clean.
- 2026-10-28: User refined tilt: rotation only 10-40 deg, view tilt only 10-40%, combined +/-12 deg + 10-25% (modes equally likely, p=0.3), applied only to straight plates: renders/synthetic always, real crops when `looks_straight` (Hough lines level <3 deg and top/bottom parallel <3 deg; ~2,038/5,236 real crops; no lines -> not straight). Preview custom_alpr/outputs/rotation-preview/tilt-detection.png. Added `add_finish` to synthetic plates (embossed characters, gloss gradient, soft cast shadow on half), preview custom_alpr/outputs/synthetic-preview/finish-28.png; regenerated synthetic-v1. 50 tests pass, ruff clean.
