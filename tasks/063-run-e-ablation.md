# 063 Run E ablation: synthetic plates, view tilt + strong downscale

## Goal
Train three runs on top of run D in parallel on vast.ai and find which change helps: E1 = + synthetic-v1 (2,400 plates); E2 = + view tilt (straight plates only) + strong downscale; E = both.

## Checklist
- [x] Upload code, label-fixed manifests, synthetic-v1 to vast.ai; tests pass there.
- [x] Train E1/E2/E in parallel (tmux, workers 6, ~95 s/epoch, GPU ~100%).
- [x] Analyze test/dev, noise robustness for C, D, E1, E2, E; compare by groups with McNemar.
- [ ] User decides next step; vast.ai instance to be stopped by user.

## Done criteria
Comparison reported to the user.

## Progress log
- 2026-10-28: Results (label-fixed test/dev exact): C 0.896/0.878, D 0.897/0.899, E1 0.893/0.894 (early stop ep 47), E2 0.887/0.880, E 0.897/0.895. No run differs significantly from D on test (E vs D 24/24, p=1.00; E2 vs D 20/32, p=0.13). E best on small <25 px in both splits (test 0.714 vs D 0.693, dev 0.712 vs 0.686; pooled 29/18, p=0.14) and fewest confident errors (test 39 vs D 45). E2 worse on dev (p=0.001 vs D) and on salt-pepper 10% (0.831 vs 0.868). Fixed looks_straight returning numpy.bool_ (analysis only; training used truthiness). Outputs in custom_alpr/outputs/train-custom-ocr/run-20261028-0[678]-*, noise-robustness-test-C-D-E.json.
