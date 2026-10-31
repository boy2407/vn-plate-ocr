# 074 — Score pose warp margins from 1 to 5 percent

Status: done. Started/completed: 2026-10-30.

User explicitly requested all margins 1/2/3/4/5% on both task073 cameras. Reuse saved pose corners and seek-verified frames, unchanged E and pose checkpoint; do not rerun detection or train. This is exploratory: Sài Gòn has already been inspected and cannot be reused for an unbiased margin-selection claim. Preserve task073's original 5% selection and single-run report. Score missing/invalid warps as incorrect; report exact/Wilson, CER, paired changes and bootstrap against refine and 5%.

## Results

Same model E, same saved pose corners, same missing-crop policy as task073. No pose inference, training, integration or checkpoint changes. Sài Gòn is exploratory, not a new held-out result. Exact intervals are Wilson95, paired bootstrap100000 seed73; CER intervals are crop bootstrap10000. Repeated frames per vehicle and multiple margins limit statistical claims.

### danang-ptz

| Margin | Exact [Wilson95] | CER [bootstrap95] | Fixed/broken vs refine; Δ [paired95] | Fixed/broken vs5%; Δ [paired95] | Missing [Wilson95] |
|---|---|---|---|---|---|
| 1% | 14/37 = 37.8% [24.1, 53.9] | 33.2% [21.4, 46.3], chars=295 | 0/6; Δ-16.2pp [-29.7, -5.4] | 0/2; Δ-5.4pp [-13.5, +0.0] | 8/37 = 21.6% [11.4, 37.2] |
| 2% | 15/37 = 40.5% [26.3, 56.5] | 31.2% [19.6, 44.4], chars=295 | 0/5; Δ-13.5pp [-24.3, -2.7] | 1/2; Δ-2.7pp [-10.8, +5.4] | 8/37 = 21.6% [11.4, 37.2] |
| 3% | 14/37 = 37.8% [24.1, 53.9] | 31.2% [19.5, 44.2], chars=295 | 0/6; Δ-16.2pp [-29.7, -5.4] | 0/2; Δ-5.4pp [-13.5, +0.0] | 8/37 = 21.6% [11.4, 37.2] |
| 4% | 15/37 = 40.5% [26.3, 56.5] | 31.5% [19.7, 44.7], chars=295 | 1/6; Δ-13.5pp [-27.0, +0.0] | 0/1; Δ-2.7pp [-8.1, +0.0] | 8/37 = 21.6% [11.4, 37.2] |
| 5% | 16/37 = 43.2% [28.7, 59.1] | 30.8% [18.9, 44.2], chars=295 | 2/6; Δ-10.8pp [-24.3, +2.7] | 0/0; Δ+0.0pp [+0.0, +0.0] | 8/37 = 21.6% [11.4, 37.2] |

### saigon-drive

| Margin | Exact [Wilson95] | CER [bootstrap95] | Fixed/broken vs refine; Δ [paired95] | Fixed/broken vs5%; Δ [paired95] | Missing [Wilson95] |
|---|---|---|---|---|---|
| 1% | 10/58 = 17.2% [9.6, 28.9] | 38.5% [29.7, 47.8], chars=494 | 3/8; Δ-8.6pp [-19.0, +1.7] | 1/4; Δ-5.2pp [-13.8, +1.7] | 11/58 = 19.0% [10.9, 30.9] |
| 2% | 12/58 = 20.7% [12.3, 32.8] | 37.2% [28.1, 47.0], chars=494 | 5/8; Δ-5.2pp [-17.2, +6.9] | 1/2; Δ-1.7pp [-6.9, +3.4] | 11/58 = 19.0% [10.9, 30.9] |
| 3% | 11/58 = 19.0% [10.9, 30.9] | 36.2% [27.3, 45.9], chars=494 | 4/8; Δ-6.9pp [-19.0, +5.2] | 1/3; Δ-3.4pp [-10.3, +3.4] | 11/58 = 19.0% [10.9, 30.9] |
| 4% | 11/58 = 19.0% [10.9, 30.9] | 36.6% [27.6, 46.3], chars=494 | 4/8; Δ-6.9pp [-19.0, +5.2] | 0/2; Δ-3.4pp [-8.6, +0.0] | 11/58 = 19.0% [10.9, 30.9] |
| 5% | 13/58 = 22.4% [13.6, 34.7] | 35.8% [26.7, 45.5], chars=494 | 6/8; Δ-3.4pp [-15.5, +8.6] | 0/0; Δ+0.0pp [+0.0, +0.0] | 11/58 = 19.0% [10.9, 30.9] |

### pooled_exploratory

| Margin | Exact [Wilson95] | CER [bootstrap95] | Fixed/broken vs refine; Δ [paired95] | Fixed/broken vs5%; Δ [paired95] | Missing [Wilson95] |
|---|---|---|---|---|---|
| 1% | 24/95 = 25.3% [17.6, 34.8] | 36.5% [29.4, 44.0], chars=789 | 3/14; Δ-11.6pp [-20.0, -3.2] | 1/6; Δ-5.3pp [-10.5, +0.0] | 19/95 = 20.0% [13.2, 29.1] |
| 2% | 27/95 = 28.4% [20.3, 38.2] | 35.0% [27.7, 42.7], chars=789 | 5/13; Δ-8.4pp [-16.8, +0.0] | 2/4; Δ-2.1pp [-7.4, +3.2] | 19/95 = 20.0% [13.2, 29.1] |
| 3% | 25/95 = 26.3% [18.5, 36.0] | 34.3% [27.2, 41.9], chars=789 | 4/14; Δ-10.5pp [-18.9, -2.1] | 1/5; Δ-4.2pp [-9.5, +0.0] | 19/95 = 20.0% [13.2, 29.1] |
| 4% | 26/95 = 27.4% [19.4, 37.1] | 34.7% [27.6, 42.4], chars=789 | 5/14; Δ-9.5pp [-17.9, -1.1] | 0/3; Δ-3.2pp [-7.4, +0.0] | 19/95 = 20.0% [13.2, 29.1] |
| 5% | 29/95 = 30.5% [22.2, 40.4] | 34.0% [26.7, 41.7], chars=789 | 8/14; Δ-6.3pp [-15.8, +3.2] | 0/0; Δ+0.0pp [+0.0, +0.0] | 19/95 = 20.0% [13.2, 29.1] |

5% exactly reproduced task073: Đà Nẵng16/37 and Sài Gòn13/58, no changed OCR strings. No valid warp for8/37 and11/58 respectively, counted as wrong at every margin. Pixel rounding means adjacent percentages may generate identical crop dimensions on small plates; per-row output dimensions and reads are saved.

Highest observed exact is5% (29/95), followed by2% (27/95); differences among margins are below about10 crops, so chưa đủ bằng chứng that one margin is generally better. Neither beats the observed refine baseline35/95. Do not change the original locked5% selection/report or claim this sweep as a fresh Sài Gòn test.

Artifacts: custom_alpr/outputs/pose-margin-074/report.json and camera rows.json. Script: .venv-cpu/bin/python -m custom_alpr.pose_margin_sweep. Ruff new script passes. Required full pytest/ruff executed; results recorded below.

Full pytest remains blocked during collection by missing custom_alpr.yolo_vast.corruptions. Full Ruff reports126 errors in the current tree; new script passes. Existing user changes were left untouched.
