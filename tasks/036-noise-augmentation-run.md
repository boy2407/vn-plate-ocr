# Train PARSeq with Gaussian and salt-and-pepper noise augmentation (run C)

## Goal
Make the model robust to Gaussian and salt-and-pepper noise without losing clean accuracy, by training run B's exact configuration plus noise augmentation.

## Checklist
- [x] Add random Gaussian noise (p=0.3, sigma 5-30) and salt-and-pepper (p=0.2, 0.5-5% pixels) to training augmentation; tests.
- [x] Train run C on desktop (same settings as run B).
- [x] Re-run noise robustness and error analysis; compare C vs B (all images and real-only), McNemar.

## Done criteria
Run C finished; robustness table and clean accuracy compared with run B.

## Progress log
- 2026-10-28: Started at user's request after the noise robustness check (task 034).
- 2026-10-28: Added `random_gaussian_noise`, `random_salt_pepper`, dataset `noise=` flag and `--noise-augment`; 11 tests pass on desktop, ruff clean. Launched run C `run-20261028-04-parseq-aug-noise` (run B settings + --noise-augment).
- 2026-10-28: Run C ran 60 epochs, best epoch 55. Clean test 0.8944 vs B 0.8753 (C fixes 37, breaks 14, McNemar p=0.0018); dev 0.8778 vs 0.8736 (30 vs 25, p=0.59, not significant). Real photos test 0.809 vs 0.777. Noise robustness: salt-and-pepper 5% 0.879 vs 0.789, 10% 0.848 vs 0.677; Gaussian sigma 30 0.880 vs 0.845. Outputs: run-20261028-04-parseq-aug-noise/, noise-robustness-test-B-C.json. Done.
- 2026-10-28: Dev errors: B 150 (12.6%), C 145 (12.2%); real photos B 147/652, C 140/652; both wrong 120, only B 30, only C 25. Viewed all 40 C dev errors >=40 px: 21 clear images misread (16 real + 5 _lp), 2 known label errors, 6 hard-to-read, 11 bad images. Clear-error patterns: strongly tilted/rotated large car plates (car_13, car_306, car_525), blue plates (type6_022/027/039), rare formats 99999A/99999A9 (type2_862/880), rare letters/combos (V, C, K, N, Y, HB, PT, HD, LH).
