# Add motion-blur augmentation helper

## Goal
Provide a training-only directional motion blur augmentation that preserves plate transcription labels.

## Checklist
- [x] Implement seeded, probabilistic motion blur for RGB plate crops.
- [x] Document intended train-only use and starting parameters.
- [x] Run required project checks where available.
- [x] Update task overview and completion log.

## Progress log
- 2026-10-27: Started after user requested motion-blur augmentation and asked what multi-frame recognition means.
- 2026-10-27: Added `random_motion_blur` with a seeded NumPy generator, direction selection, and mild 3/5/7-pixel kernels. Documented that no custom Dataset/train loop exists, so it is not automatically applied yet. Syntax/diff checks pass; pytest/Ruff unavailable in host environment.
