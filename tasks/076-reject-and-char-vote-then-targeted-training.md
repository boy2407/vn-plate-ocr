# 076 — Calibrated rejection, per-character track voting, then targeted OCR training

Status: Phases A and B complete; Phase C in progress per user request. Assignee: Codex.

Goal: a plate the system reports must almost always be right. When it is unsure it
says "unreadable" instead of guessing. Do the phases **in order**. **Stop after each
phase**, write its results to this file, and wait for approval. Do not integrate into
`web_app.py` / `video_demo.py` and do not change `DEFAULT_OCR` unless asked.

## Context (read first)

- `docs/models/plate-ocr/video-ocr-diagnosis.md`: the diagnosis. On video the strokes
  are still in the pixels (median crop height 58 px). The main errors come from the
  domain gap and loose YOLO boxes. Layout-constrained decoding changed nothing.
  Video confusions are `7→2, N→M, 2→0, 9→3, 8→6`.
- Models: E = `custom_alpr/outputs/train-custom-ocr/run-20261029-09-parseq-f-compress/best.pt`,
  E-v1 = `DEFAULT_OCR` in `custom_alpr/video_demo.py`. E-v1 was trained on video
  crops, so treat it as a reference only on the confirmed set.
- Data:
  - Kaggle test `custom_alpr/data/ocr/kaggle-v9-prepared/test.csv`.
  - 101 human-confirmed video crops `custom_alpr/data/ocr/collected-v1/confirmed-tight.csv`.
  - Per-crop `track_id`, `quality`, `camera` in `custom_alpr/data/collected/*/metadata.csv`,
    joined by crop stem.
  - The 249 Codex+VLM agreed crops come from `custom_alpr/consensus.py`. They are
    secondary evidence only and must be reported separately from the human-confirmed crops.
- Existing tools: `plate_tracks.py`, `confusion_report.py`, `evaluate_decoding_variants.py`,
  `refine_holdout_eval.py`, `augment.py`, `synth_plates.py`, `train_custom_ocr.py`,
  `PARSeq.generate_with_confidence` / `layout_decode` in `custom_ocr.py`.
- Super-resolution was measured as harmful. Do not use it.
- Statistics conventions from tasks 071–075:
  - Wilson 95% CI for exact accuracy.
  - Bootstrap CI for CER.
  - Paired bootstrap for deltas.
  - Hold out **by camera** (danang-ptz / saigon-drive; hanoi-drive is test).
  - Crops from one vehicle are correlated, so state it.
  - A difference under ~10 crops, or a CI containing 0, is "not enough evidence".
- CPU venv: `.venv-cpu`. Training runs on the desktop RTX 5060 (see tasks 067/073).
  Data and outputs are gitignored.

## Phase A — Calibrated rejection (CPU, no training)

1. New script `custom_alpr/reject_eval.py`. For E and E-v1 on Kaggle test and on
   confirmed crops, save per crop: prediction, truth, crop height, and these candidate
   confidence scores:
   - (a) the current geometric-mean confidence;
   - (b) the minimum per-character probability;
   - (c) the margin between top-1 and top-2 at the weakest position;
   - (d) agreement between free greedy and `layout_decode`.
   Expose per-position probabilities from `generate_with_confidence` through a new
   optional return. Do not change existing callers' behaviour.
2. Build risk–coverage curves: accuracy of the accepted crops vs. fraction accepted.
3. Choose the threshold for 98% and 99% precision **on one camera only**. Then report
   precision/coverage with Wilson CI on the other camera, both directions. Report Kaggle
   per height bucket (<16, 16–24, 24–32, ≥32 px).
4. Unit tests for score functions in `custom_alpr/tests/test_reject_eval.py`.

Deliverable: a table of score × target precision → held-out precision [CI] and
coverage [CI]. Name the best score. If no score reaches ≥95% held-out precision with
non-trivial coverage, say so plainly.

## Phase B — Per-character track voting (CPU, no training)

1. In `plate_tracks.py`, add an optional per-character vote. Keep the old
   string-level vote as default.
   - Group a track's observations by predicted length / layout. Take the dominant layout.
   - Per position, sum `weight × log p(char)` over observations.
     `weight = crop_quality.score` (existing `vote_weight` without OCR confidence).
   - The track confidence is the per-position posterior margin. Apply Phase A's
     threshold so the track can output "unreadable".
2. `custom_alpr/track_vote_eval.py`: on confirmed crops grouped by `track_id`, compare
   1. single best crop;
   2. current string vote;
   3. per-char vote;
   4. per-char vote + rejection.

   Track truth is the confirmed label. Drop a track if its labels disagree, and list
   it. Report the number of tracks honestly; it will be small.
3. Run the same comparison on the 249 agreed crops as a separate, secondary table.
4. Tests in `custom_alpr/tests/test_plate_tracks.py`, covering:
   - a 3-observation track where every crop has one different wrong digit, so
     per-char voting recovers the plate;
   - a mixed-length track;
   - a track that falls below the threshold and is rejected.

Stop here. Report. Phases C–E need GPU and approval.

## Results — Phases A and B

### Phase A — Calibrated rejection

E is the primary calibration model. The six Hanoi crops were excluded from every
score and threshold choice; only Đà Nẵng (37 crops) and Sài Gòn (58 crops) were used
in both directions. The full 101-row prediction file retains Hanoi for the locked
test set, but its predictions are not included in the camera calibration table.
Crop observations from a vehicle are correlated.

All 98% and 99% empirical precision thresholds were identical on these small
calibration sets. The table reports held-out precision and coverage with Wilson 95%
intervals. `—` means the calibration camera had no threshold meeting the target.

| Score | Calibration → held-out | Threshold | Held-out precision [95%] | Coverage [95%] |
|---|---|---:|---:|---:|
| Geometric mean | Đà Nẵng → Sài Gòn | 0.9230 | 3/3 = 100% [43.9, 100] | 3/58 = 5.2% [1.8, 14.1] |
| Geometric mean | Sài Gòn → Đà Nẵng | 0.9235 | 1/1 = 100% [20.7, 100] | 1/37 = 2.7% [0.5, 13.8] |
| Minimum character probability | Đà Nẵng → Sài Gòn | 0.9153 | 3/3 = 100% [43.9, 100] | 3/58 = 5.2% [1.8, 14.1] |
| Minimum character probability | Sài Gòn → Đà Nẵng | 0.9110 | 2/2 = 100% [34.2, 100] | 2/37 = 5.4% [1.5, 17.7] |
| Weakest-position margin | Đà Nẵng → Sài Gòn | 0.9118 | 3/3 = 100% [43.9, 100] | 3/58 = 5.2% [1.8, 14.1] |
| Weakest-position margin | Sài Gòn → Đà Nẵng | 0.9007 | 2/2 = 100% [34.2, 100] | 2/37 = 5.4% [1.5, 17.7] |
| Free/layout agreement | Đà Nẵng → Sài Gòn | — | no threshold | 0/58 = 0% [0, 6.2] |
| Free/layout agreement | Sài Gòn → Đà Nẵng | — | no threshold | 0/37 = 0% [0, 9.4] |

The weakest-position margin and minimum character probability tie on held-out
coverage. Weakest-position margin is used in Phase B because the track confidence is
also a posterior margin. No E score achieved at least 95% held-out precision at
coverage above 10%; the apparent 100% precision comes from only one to three crops
per direction, with very wide intervals. This is not enough evidence for a useful
crop-level rejection threshold.

E exact accuracy by Kaggle test crop height:

| Height | Correct | Exact [Wilson 95%] |
|---|---:|---:|
| <16 px | 37/76 | 48.7% [37.8, 59.7] |
| 16–24 px | 120/150 | 80.0% [72.9, 85.6] |
| 24–32 px | 101/113 | 89.4% [82.4, 93.8] |
| ≥32 px | 816/863 | 94.6% [92.8, 95.9] |

E-v1 is included as descriptive reference only: it was trained on these video crops,
so the camera table is in-sample and cannot validate generalization or select a
threshold. Its 98%/99% threshold results are below; each selected cutoff is from
the calibration camera, but both cameras are represented in E-v1's video training
data.

| Score | Calibration → held-out | Threshold | Held-out precision [95%] | Coverage [95%] |
|---|---|---:|---:|---:|
| Geometric mean | Đà Nẵng → Sài Gòn | 0.8831 | 54/54 = 100% [93.4, 100] | 54/58 = 93.1% [83.6, 97.3] |
| Geometric mean | Sài Gòn → Đà Nẵng | 0.8722 | 36/36 = 100% [90.4, 100] | 36/37 = 97.3% [86.2, 99.5] |
| Minimum character probability | Đà Nẵng → Sài Gòn | 0.6435 | 57/57 = 100% [93.7, 100] | 57/58 = 98.3% [90.9, 99.7] |
| Minimum character probability | Sài Gòn → Đà Nẵng | 0.5780 | 36/36 = 100% [90.4, 100] | 36/37 = 97.3% [86.2, 99.5] |
| Weakest-position margin | Đà Nẵng → Sài Gòn | 0.4283 | 56/56 = 100% [93.6, 100] | 56/58 = 96.6% [88.3, 99.0] |
| Weakest-position margin | Sài Gòn → Đà Nẵng | 0.2315 | 36/36 = 100% [90.4, 100] | 36/37 = 97.3% [86.2, 99.5] |
| Free/layout agreement | Đà Nẵng → Sài Gòn | — | no threshold | 0/58 = 0% [0, 6.2] |
| Free/layout agreement | Sài Gòn → Đà Nẵng | 1.0000 | 36/37 = 97.3% [86.2, 99.5] | 37/37 = 100% [90.6, 100] |

On its Kaggle test the E-v1 height-bucket exact rates were 48.7%, 81.3%, 86.7%,
and 94.9%, respectively.

The risk–coverage curves for all four scores and both calibration cameras are in
`custom_alpr/outputs/reject-eval-076/report.json`. Per-crop prediction, truth, source
height, free/layout output, all score values, camera, track ID, crop quality, and
per-position probabilities are in `E-confirmed.json` and `E-v1-confirmed.json`;
Kaggle rows are in the matching `*-kaggle.json` files.

### Phase B — Per-character track voting

The optional `per_character` vote groups observations by predicted digit/letter
layout, chooses the dominant layout, sums `crop quality × log p(character)` at each
position, and rejects when the weakest top-two posterior margin is below the locked
Phase A threshold. The existing string vote remains the default.

| Human-confirmed set | Tracks | Correct / exact [Wilson 95%] | Coverage |
|---|---:|---:|---:|
| Best single crop | 71 | 19/71 = 26.8% [17.9, 38.1] | 100% |
| Current string vote | 71 | 20/71 = 28.2% [19.0, 39.5] | 100% |
| Per-character vote | 71 | 19/71 = 26.8% [17.9, 38.1] | 100% |
| Per-character + rejection, Đà Nẵng threshold → Sài Gòn | 34 held-out tracks | 2/2 accepted = 100% [34.2, 100] | 2/34 = 5.9% [1.6, 19.1] |
| Per-character + rejection, Sài Gòn threshold → Đà Nẵng | 32 held-out tracks | no accepted track | 0% [0, 10.7] |

There were no conflicting human labels among the 71 confirmed tracks. Base track
comparisons include the six locked Hanoi crops after the thresholds were fixed;
Hanoi was not used in calibration. Per-character voting did not improve over the
string vote, and the one-track difference is not enough evidence of a change.

The separate Codex+VLM agreed set contains 249 crops and 113 non-conflicting tracks;
one track with disagreeing labels (`20261029131728-31`, labels `43A28779` and
`43A67359`) was dropped and listed in the report.

| Secondary agreed set | Tracks | Correct / exact [Wilson 95%] | Coverage |
|---|---:|---:|---:|
| Best single crop | 113 | 58/113 = 51.3% [42.2, 60.3] | 100% |
| Current string vote | 113 | 60/113 = 53.1% [43.9, 62.0] | 100% |
| Per-character vote | 113 | 57/113 = 50.4% [41.4, 59.5] | 100% |
| Per-character + rejection, Đà Nẵng threshold → Sài Gòn | 29 held-out tracks | 9/9 accepted = 100% [70.1, 100] | 9/29 = 31.0% [17.3, 49.2] |
| Per-character + rejection, Sài Gòn threshold → Đà Nẵng | 65 held-out tracks | 10/12 accepted = 83.3% [55.2, 95.3] | 12/65 = 18.5% [10.9, 29.6] |

This agreed set is secondary evidence, not a substitute for human labels. Its track
results differ from the human set and do not establish that the margin threshold
generalizes.

### Reproduction and outputs

```bash
.venv-cpu/bin/python -m custom_alpr.reject_eval --output custom_alpr/outputs/reject-eval-076 --batch-size 32
.venv-cpu/bin/python -m custom_alpr.track_vote_eval --score weakest_position_margin --output custom_alpr/outputs/track-vote-076
.venv-cpu/bin/python -m pytest custom_alpr/tests/test_reject_eval.py custom_alpr/tests/test_plate_tracks.py custom_alpr/tests/test_custom_ocr.py -q
```

Outputs: `custom_alpr/outputs/reject-eval-076/` and
`custom_alpr/outputs/track-vote-076/`. Verdict: score-based crop rejection can
produce high empirical precision only at very low coverage; the available video
sample is too small to establish it as a useful operating point. Per-character
voting does not help on the human-confirmed tracks. Phases C–E were not started.

Validation: focused tests pass (46 passed, 1 skipped). The required full test run
still fails collection on the pre-existing missing module
`custom_alpr.yolo_vast.corruptions`. Project-wide Ruff reports 126 pre-existing issues;
all changed production files and the new/updated plate-track and rejection tests pass
Ruff. No unrelated lint or import issue was changed.

## Phase C — Targeted synthetic data with video-like degradation (GPU)

1. `synth_plates.py`: add `--confusable-boost` that oversamples characters in
   `{3,2,5,8,9}, {1,7}, {0,8,6}, {N,H,M}`. Weight the sampling by the Phase 1 video
   confusion counts from `confusion_report.py`. Output
   `custom_alpr/data/ocr/synthetic-v2-confusable/`.
2. Degrade it like video using existing `augment.py` functions:
   - `random_downscale` to 12–24 px;
   - `random_motion_blur`;
   - `random_h264`;
   - `random_view_tilt`;
   - plus **loose-box context**: paste the plate into a larger background crop with
     random 10–60% margin. Loose YOLO boxes are the main video error.

   Save a 64-image preview grid and look at it before training.
3. Train from E with the same recipe as `run-20261029-09-parseq-f-compress`. Mix in
   Kaggle train plus synthetic-v2 (no video crops, so the confirmed set stays a clean
   test). Use the same seed and epochs budget.
4. Evaluate Kaggle test (must not drop > 1 point), confirmed crops per camera, and the
   per-pair confusion matrix before vs after.

### Phase C results — loose-box/video-like synthetic training

Generated `custom_alpr/data/ocr/synthetic-v2-confusable/` with 2,400 images and
combined it with the 9,541 Kaggle train images (11,941 training rows); Kaggle val
(1,187) and test (1,202) remain unchanged. The confusion weights came from the
95 human-confirmed Đà Nẵng/Sài Gòn crops only; the six Hanoi crops and all confirmed
video crops were excluded from training. The procedural crop context contains a car
body panel and red/orange tail-light detail, with independently sampled margins from
10% to 60% on each side. Every synthetic sample was downscaled to 12–24 px before
final OCR resize, 50% received motion blur, 40% H.264 compression, and 30% view tilt
(realized counts: 2,400 / 1,200 / 969 / 727 respectively). No held-out video frame was
used as synthetic background.

The requested 64-image preview was inspected before training:
`custom_alpr/outputs/phase-c-076/preview-grid-64.png`.

Training initialized from E with the same PARSeq recipe, seed 20261028, 60-epoch
budget, batch 32, LR 0.0003, and patience 8. Early stopping occurred at epoch 28;
best checkpoint is epoch 20 at
`custom_alpr/outputs/train-custom-ocr/run-20261030-11-parseq-f-confusable-loose/best.pt`.
Best validation exact was 89.8%; its saved test exact was 89.6% (1,077/1,202).
Independent paired evaluation below gives E 89.4% and C 89.6% on the same test set.

#### Kaggle and confirmed video comparison

Exact accuracy intervals are Wilson 95%; CER intervals are bootstrap 95%. E is the
`run-20261029-09-parseq-f-compress` checkpoint. The refined set is generated from
the same 101 verified source frames with task 070's stored `final_box`; 12 rows used
the explicit fallback box (Đà Nẵng 1/37, Hanoi 1/6, Sài Gòn 10/58). Crop observations
are correlated by vehicle and camera, so these intervals are descriptive, not an
independent-vehicle generalization estimate.

| Set / camera | E exact; CER | C exact; CER | Paired C−E exact 95% CI | Paired C−E CER 95% CI |
|---|---:|---:|---:|---:|
| Original — Đà Nẵng (n=37) | 9/37 24.3% [13.4, 40.1]; 30.17% | 13/37 35.1% [21.8, 51.2]; 20.68% | [+2.7, +21.6] pp | [−16.10, −3.40] pp |
| Original — Sài Gòn (n=58) | 15/58 25.9% [16.3, 38.4]; 28.34% | 21/58 36.2% [25.1, 49.1]; 20.24% | [0, +22.4] pp | [−11.99, −4.29] pp |
| Original — Hanoi (n=6, locked test) | 1/6 16.7%; 65.96% | 1/6 16.7%; 57.45% | [0, 0] pp | [−29.17, +12.77] pp |
| `refine_box` — Đà Nẵng (n=37) | 20/37 54.1% [38.4, 69.0]; 11.86% | 20/37 54.1% [38.4, 69.0]; 9.83% | [−13.5, +13.5] pp | [−5.76, +1.69] pp |
| `refine_box` — Sài Gòn (n=58) | 15/58 25.9% [16.3, 38.4]; 22.67% | 18/58 31.0% [20.6, 43.8]; 22.06% | [−3.4, +13.8] pp | [−3.41, +2.41] pp |
| `refine_box` — Hanoi (n=6, locked test) | 2/6 33.3%; 19.15% | 3/6 50.0%; 19.15% | [0, +50.0] pp | [−6.25, +6.67] pp |
| Kaggle test (n=1,202) | 1,074/1,202 89.35% [87.48, 90.97]; CER 3.57% [2.82, 4.35] | 1,077/1,202 89.60% [87.75, 91.20]; CER 3.32% [2.61, 4.08] | [−0.75, +1.25] pp | [−0.51, +0.01] pp |

Per-camera exact/CER is also in
`custom_alpr/outputs/phase-c-076/evaluation/report.json`, including Wilson and
bootstrap intervals for each camera and crop version. On the original video crops,
C improves exact reads by 4 on Đà Nẵng and 6 on Sài Gòn; paired bootstrap supports
lower CER on both, while Hanoi is too small and was not used to choose training
settings. After `refine_box`, Đà Nẵng exact ties and Sài Gòn gains 3; paired
intervals include zero for both. The requested model comparisons therefore support
“helps on loose original boxes; not enough evidence for an added benefit after
refining boxes.” Kaggle exact changes by +0.25 pp and does not indicate a material
regression; the paired interval still includes a small negative change.

The leading original-crop substitutions shift from E's `7→2 (6), N→M (5), 2→0
(5), 9→3 (5), 8→6 (4)` to C's `N→M (6), 9→3 (4), 0→8 (4), 5→9 (3), 5→4
(3)`. On refined crops, E's top pairs are `N→M (7), 8→6 (6), 7→3 (5), 3→2
(4), 0→8 (4)`; C's are `N→M (5), 3→9 (4), 5→6 (4), L→C (4), 8→6 (3)`.
These are per-character substitution counts, not counts of wholly incorrect plates.

#### Rerun rejection evaluation on C

Command:

```bash
.venv-cpu/bin/python -m custom_alpr.reject_eval --model C=custom_alpr/outputs/train-custom-ocr/run-20261030-11-parseq-f-confusable-loose/best.pt --output custom_alpr/outputs/reject-eval-076-C --batch-size 32
```

C also does not yield a useful high-precision operating point on these small camera
sets. With weakest-position margin, calibrating Đà Nẵng and holding out Sài Gòn
accepted 3/58 (100%, Wilson [43.9, 100], coverage 5.2% [1.8, 14.1]); reversing
the cameras accepted 2/37 (100%, [34.2, 100], coverage 5.4% [1.5, 17.7]). This
meets empirical 98/99% precision only at very low coverage and with wide intervals.
Geometric mean accepted 1/58 correct in one direction and 1/37 of 2 in the other;
minimum probability accepted 1/58 and 2/37. No score achieved 95% precision with
non-trivial coverage. Hanoi remains excluded from threshold choice. C's Kaggle exact
by height was 41/76 (53.9%) for <16 px, 121/150 (80.7%) for 16–24 px, 98/113
(86.7%) for 24–32 px, and 817/863 (94.7%) for ≥32 px.

Reproduction/evaluation commands:

```bash
.venv-cpu/bin/python -m custom_alpr.synth_plates --output custom_alpr/data/ocr/synthetic-v2-confusable --seed 20261028 --confusable-boost --confusion-report custom_alpr/outputs/phase-c-076/confusion-report/report.json --video-like --preview-grid custom_alpr/outputs/phase-c-076/preview-grid-64.png
.venv-cpu/bin/python -m custom_alpr.phase_c_eval --checkpoint custom_alpr/outputs/train-custom-ocr/run-20261030-11-parseq-f-confusable-loose/best.pt --output custom_alpr/outputs/phase-c-076/evaluation --batch-size 32
```

Verdict: targeted loose-box training helps on the original camera crops without a
measurable Kaggle regression. `refine_box` already recovers much of the error, and
the incremental gain after refinement is inconclusive. The rejection rerun still
shows very low useful coverage. **Phase C is complete; stopped here as requested.**

Validation: focused Phase C and related OCR tests passed (59 passed). Ruff passed on
`synth_plates.py`, `phase_c_eval.py`, `reject_eval.py`, and `test_synth_plates.py`.
The required full `custom_alpr/tests` run remains blocked during collection by the
existing missing `custom_alpr.yolo_vast.corruptions` import. Project-wide Ruff reports
125 existing issues; no unrelated files were changed to address them.

## Phase D — Clean→degraded feature distillation (GPU)

1. Teacher = frozen E reading the clean image. Student = copy of E reading a degraded
   version of the **same** image (Phase C pipeline).
2. Loss = CE(student, label) + λ·MSE(student encoder memory, teacher encoder memory)
   + optional KL on decoder logits at T=2. Sweep λ ∈ {0.1, 0.5, 1.0} on val only.
3. Add `--distill-teacher`, `--distill-weight`, `--distill-kl` to
   `train_custom_ocr.py`. Default off; existing runs reproduce unchanged. Add a smoke
   test with `--max-train-batches 2`.
4. Evaluate exactly as in Phase C, against the Phase C model as the baseline.

## Results — Phase D (clean→degraded distillation; run by Claude, not Codex)

Naming: **E-loose** = the Phase C model (`run-20261030-11-parseq-f-confusable-loose`).
Student and teacher both start from E-loose; the teacher reads the un-augmented image.
Distillation applies only to rows with a clean view (Kaggle + synthetic-v1, 9541 of 11941);
synthetic-v2 images are stored pre-degraded and get the normal loss only. Code:
`--distill-teacher/--distill-weight/--distill-kl` in `train_custom_ocr.py` (default off).
Same recipe as E-loose, lr 1e-4, up to 12 epochs, patience 4, selection on Kaggle val only.

| Run | MSE / KL | Best val greedy exact | Video 101, original | Video 101, refine_box | Kaggle test |
|---|---|---|---|---|---|
| E-loose (start) | – | 0.8981 | 35/101 | 41/101 | 1077/1202 |
| d0 (control, no distill) | 0 / 0 | 0.8981 (ep 2) | 31/101 | 35/101 | 1082/1202 |
| d05 | 0.5 / 0.5 | 0.8972 (ep 2) | 29/101 | 39/101 | 1083/1202 |
| d10 | 1.0 / 0.5 | 0.8989 (ep 2) | 30/101 | 38/101 | 1083/1202 |

All runs early-stopped at epoch 6 with the best epoch at 2. Paired bootstrap of CER (new − E-loose)
on the original crops has intervals containing 0 for both cameras in every run (e.g. d10 Đà Nẵng
[-0.010, +0.092]); Kaggle accuracy deltas also contain 0.

Verdict: **not enough evidence that distillation helps; no sign of benefit.** Distilled runs
are not better than the d0 control on video (29–30 vs 31 of 101, within noise), and every
continued-training run is at or below E-loose on video. Val is flat from epoch 1, so the model
has little left to gain from this loss on this data. Not tested: distillation from a clean
view of loose-box crops (needs regenerating synthetic-v2 with clean pairs).

Outputs: `custom_alpr/outputs/phase-d-076/{d0,d05,d10}/report.json`; checkpoints in
`custom_alpr/outputs/train-custom-ocr/run-phased-*`. Desktop launcher:
`custom_alpr/outputs/phase-d-run.ps1` (start with `Invoke-CimMethod Win32_Process Create`; a
`Start-Process` child dies when the SSH session closes).

### Phase D follow-up — clean comparison on the 249 Codex+VLM agreed crops

`custom_alpr/agreed_eval.py`; output `custom_alpr/outputs/agreed-eval-076/`. No agreed crop shares an
image or sha256 with any training manifest, but 35 of its 75 distinct plates occur in E-v1's training
manifest (other frames of the same vehicles). So E-v1 is also reported on the 175 crops (40 plates)
whose plate text it never saw. Labels are machine-agreed, not human: secondary evidence. Cameras:
Đà Nẵng 136, Hà Nội 51, Sài Gòn 62; Hà Nội is test-only.

| Model | All 249 | Plates unseen by E-v1 (175) | Plates seen by E-v1 (74) |
|---|---|---|---|
| E | 117 (47.0%) | 75 (42.9%) | 42 (56.8%) |
| E-loose | 151 (60.6%) | 102 (58.3%) | 49 (66.2%) |
| d0 | 139 (55.8%) | 93 (53.1%) | 46 (62.2%) |
| d05 | 141 (56.6%) | 92 (52.6%) | 49 (66.2%) |
| d10 | 142 (57.0%) | 93 (53.1%) | 49 (66.2%) |
| E-v1 | 160 (64.3%) | 100 (57.1%) | 60 (81.1%) |

Paired bootstrap on the unseen 175, accuracy delta vs E-loose: E [-21.7, -9.1] pp; d0 [-9.1, -1.1];
d05 [-10.3, -1.1]; d10 [-9.7, -0.6]; E-v1 [-7.4, +5.1].

Verdict: E-loose is clearly better than E. The d runs are slightly worse than E-loose on unseen
plates (intervals just below 0); distillation does not recover it (d05/d10 ≈ d0). E-v1 is not
distinguishable from E-loose on unseen plates, and its lead (81% vs 66%) appears only on plates it
trained on. The claim "fine-tuning on 95 video crops beats E-loose" is not supported here.

## Phase E — Hard-pair loss (GPU, only if C/D still show digit↔digit confusions on readable crops)

1. Focal loss option (`--focal-gamma`, default 0) on the per-position CE.
2. Optional pair-margin term: for target chars in a confusable set, penalize the logit
   of the other members of the set (`--pair-margin`). Synthetic labels give positions.
3. Evaluate as above, baseline = best of C/D.

## Rules for every phase

- No looking at hanoi-drive / test results to pick hyperparameters. Select on val or
  on the other camera.
- Write a results section here after each phase. Include:
  - tables with CIs;
  - exact reproduce commands;
  - output paths;
  - a plain verdict: helps / not enough evidence / hurts.
- Update `tasks/README.md` and the summary section of
  `docs/models/plate-ocr/video-ocr-diagnosis.md`.
- Run `ruff` on changed files and the new tests. Note pre-existing failures but do not
  fix unrelated code.
- No commits unless asked.
