# Run D: blue-plate recolor, rotation/perspective, rare-letter sampler

## Goal
Test whether random color inversion (dark-on-light <-> light-on-dark) helps blue plates (type4 1-row, type6 2-row) without hurting other plates. Run D = run C + inversion only.

## Checklist
- [x] Add blue-plate recolor, rotation/perspective, rare-letter repeat factor sampling; tests.
- [x] Train run D on desktop with run C settings.
- [x] Compare D vs C overall, real photos, type4/type6, noise robustness; McNemar.

## Done criteria
Run D finished and compared with run C.

## Progress log
- 2026-10-28: Started at user's request.
- 2026-10-28: Code added (random_invert, --invert-augment, test; 12 tests pass on desktop, ruff clean). Training NOT started: user replied "không" — waiting for clarification.
- 2026-10-28: User chose blue-plate recoloring instead of raw inversion and approved the run-D list. Replaced `random_invert` with `random_blue_plate` (p=0.15), added `random_rotate_perspective` (p=0.3, +/-15 deg, corner shift 8%), `rare_letter_weights` + WeightedRandomSampler (`--balance-rare-letters`; rare = letters <200 in train: IJNQRSUVXYZ, weight 200/count capped at 4). 15 tests pass, ruff clean, 1-epoch smoke passed. Full training waiting: user asked how 15%/30% were chosen.
- 2026-10-28: Final run-D settings after discussion with user: blue recolor p=0.08 (raises blue plates seen in training from 8.9% to ~16.2%, the share among real photos), skipping crops whose central region is light-on-dark; rotation +/-8 deg + 5% corner shift, shrunk to fit so no character is cut, p=0.3; Repeat Factor Sampling (LVIS) factor max(1, sqrt(200/plates containing letter)), every plate kept once per epoch -> 9,918 crops/epoch instead of 9,541. 19 tests pass, ruff clean, preview custom_alpr/outputs/augment-preview-run-d.png. Launched run-20261028-05-parseq-d.
- 2026-10-28: Desktop went offline at epoch ~8; run D re-trained on vast.ai (task 054). Results D vs C: test exact 0.895 vs 0.894 (D fixes 25, breaks 24, McNemar p=1.00); real-photo test 0.814 vs 0.809 (p=0.77); test <25 px 0.693 vs 0.710 (p=0.52). Dev (used for checkpoint selection, so optimistic) 0.896 vs 0.878 (p=0.001). Noise: salt-pepper 10% 0.858 vs 0.848, Gaussian equal. 8/21 dev clear-wrong images fixed (blue 2-row, rare letters V/C/K, tilted car_306). Conclusion: no significant gain on the held-out test set.
