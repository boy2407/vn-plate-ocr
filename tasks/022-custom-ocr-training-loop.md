# Implement the custom OCR training loop

## Goal
Make the prepared Kaggle train/validation manifests consumable by `PlateOCR2D` training, with evaluation and checkpointing.

## Checklist
- [x] Add an OCR Dataset and padded collate function for prepared manifests.
- [x] Add a train/validation CLI with seeded training-only motion blur and best-checkpoint saving.
- [x] Declare dependencies and document a runnable command.
- [x] Verify the loop with a small forward/train step and run project checks.

## Done criteria
The user can run a reproducible custom `PlateOCR2D` training job from the prepared manifests and retain a best validation checkpoint.

## Progress log
- 2026-10-28: Dataset download and split manifests are complete; project has no custom OCR Dataset or train loop yet.
- 2026-10-28: Added the manifest Dataset, padding collate, training-only motion blur, seeded train/validation loop, best-checkpoint saving, and held-out greedy evaluation. One-batch MPS smoke run completed; full training has not been run.
