# Measure robustness to Gaussian and salt-and-pepper noise

## Goal
Before adding noise augmentation, measure how much exact accuracy of run01, run A, and run B drops when test images get Gaussian or salt-and-pepper noise.

## Checklist
- [x] Script that adds deterministic noise to test batches and evaluates greedy decoding.
- [x] Run on desktop for Gaussian sigma 5/10/20/30 and salt-and-pepper 1/3/5/10%.
- [x] Report exact/CER drop per model and level.

## Done criteria
A table of exact/CER per model and noise level, saved as JSON.

## Progress log
- 2026-10-28: Started.
- 2026-10-28: Done (desktop CUDA, test 1203, seed 20261028). Gaussian sigma<=10 costs <=0.5 pt for A/B; sigma 30: A -4.2, B -3.0, run01 -3.7. Salt-and-pepper hurts more: 3%: A -6.4, B -4.0; 10%: A -24.9, B -19.9, run01 -34.6. Result: custom_alpr/outputs/train-custom-ocr/noise-robustness-test.json. Script: custom_alpr/evaluate_noise_robustness.py. ruff clean.
