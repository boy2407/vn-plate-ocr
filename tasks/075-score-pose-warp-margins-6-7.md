# 075 — Score pose warp margins 6 and 7 percent

Status: done. Started/completed: 2026-10-30.

Extend the exploratory task074 sweep to 6/7%, reusing task073 saved corners, cached frames, and model E. Compare with 5% and refine on all95 crops (Đà Nẵng37, Sài Gòn58). No detector inference, training, integration or checkpoint changes. Sài Gòn has already been examined; no fresh test claim.

## Results

5% was rerun as a control and reproduced all task073 OCR strings exactly. Same model E / saved pose corners / missing policy. Confidence intervals: exact and missing Wilson95; CER bootstrap10000; paired deltas bootstrap100000 seed73. Sài Gòn is exploratory; frame observations within a vehicle are correlated.

### danang-ptz

| Margin | Exact [Wilson95] | CER [bootstrap95] | Fixed/broken vs5%; Δ [paired95] | Fixed/broken vs refine; Δ [paired95] | Missing [Wilson95] |
|---|---|---|---|---|---|
| 5% | 16/37 = 43.2% [28.7, 59.1] | 30.8% [18.9, 44.2], chars=295 | 0/0; +0.0pp [+0.0, +0.0] | 2/6; -10.8pp [-24.3, +2.7] | 8/37 = 21.6% [11.4, 37.2] |
| 6% | 16/37 = 43.2% [28.7, 59.1] | 30.2% [18.3, 43.4], chars=295 | 0/0; +0.0pp [+0.0, +0.0] | 2/6; -10.8pp [-24.3, +2.7] | 8/37 = 21.6% [11.4, 37.2] |
| 7% | 16/37 = 43.2% [28.7, 59.1] | 30.5% [18.6, 43.9], chars=295 | 0/0; +0.0pp [+0.0, +0.0] | 2/6; -10.8pp [-24.3, +2.7] | 8/37 = 21.6% [11.4, 37.2] |

### saigon-drive

| Margin | Exact [Wilson95] | CER [bootstrap95] | Fixed/broken vs5%; Δ [paired95] | Fixed/broken vs refine; Δ [paired95] | Missing [Wilson95] |
|---|---|---|---|---|---|
| 5% | 13/58 = 22.4% [13.6, 34.7] | 35.8% [26.7, 45.5], chars=494 | 0/0; +0.0pp [+0.0, +0.0] | 6/8; -3.4pp [-15.5, +8.6] | 11/58 = 19.0% [10.9, 30.9] |
| 6% | 12/58 = 20.7% [12.3, 32.8] | 35.8% [26.9, 45.4], chars=494 | 0/1; -1.7pp [-5.2, +0.0] | 6/9; -5.2pp [-19.0, +8.6] | 11/58 = 19.0% [10.9, 30.9] |
| 7% | 12/58 = 20.7% [12.3, 32.8] | 35.4% [26.6, 44.8], chars=494 | 0/1; -1.7pp [-5.2, +0.0] | 6/9; -5.2pp [-19.0, +8.6] | 11/58 = 19.0% [10.9, 30.9] |

### pooled_exploratory

| Margin | Exact [Wilson95] | CER [bootstrap95] | Fixed/broken vs5%; Δ [paired95] | Fixed/broken vs refine; Δ [paired95] | Missing [Wilson95] |
|---|---|---|---|---|---|
| 5% | 29/95 = 30.5% [22.2, 40.4] | 34.0% [26.7, 41.7], chars=789 | 0/0; +0.0pp [+0.0, +0.0] | 8/14; -6.3pp [-15.8, +3.2] | 19/95 = 20.0% [13.2, 29.1] |
| 6% | 28/95 = 29.5% [21.2, 39.3] | 33.7% [26.6, 41.2], chars=789 | 0/1; -1.1pp [-3.2, +0.0] | 8/15; -7.4pp [-16.8, +2.1] | 19/95 = 20.0% [13.2, 29.1] |
| 7% | 28/95 = 29.5% [21.2, 39.3] | 33.6% [26.5, 41.1], chars=789 | 0/1; -1.1pp [-3.2, +0.0] | 8/15; -7.4pp [-16.8, +2.1] | 19/95 = 20.0% [13.2, 29.1] |

5/6/7% pooled exact:29/28/28 of95, vs refine35/95. Differences of1 crop are insufficient evidence of a general margin effect. No benefit demonstrated for6/7% over5%; no claim of statistically proven degradation. Missing warps remain8/37 and11/58, counted as wrong at every margin.

Reproduce: `.venv-cpu/bin/python -m custom_alpr.pose_margin_sweep --percents 5 6 7 --output custom_alpr/outputs/pose-margin-075`. The default1-5% run and task073 locked selection remain unchanged. Outputs: report.json and camera rows in outputs/pose-margin-075.

Validation: Ruff modified script passes. Full pytest still fails collection on pre-existing missing custom_alpr.yolo_vast.corruptions; full Ruff reports126 errors in the existing tree. No unrelated files fixed.
