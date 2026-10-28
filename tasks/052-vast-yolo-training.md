# Task 052: Prepare portable YOLO robustness training for Vast.ai

## Goal

Prepare and validate a complete training package for the moderate YOLO robustness plan, then provision the user's rented Vast.ai host and run a smoke test.

## Checklist

- [x] Implement controlled on-the-fly corruption augmentation and Mosaic-aware trainer.
- [x] Audit/package the dataset, initial detector checkpoint and configs with hashes.
- [x] Add bootstrap, preflight GPU checks, preview, training, resume and evaluation commands.
- [x] Add rsync upload/download and persistent training session scripts.
- [x] Test augmentation/labels and run a real smoke epoch on the rented GPU.
- [x] Document ready-to-run commands and remaining data limitations.

## Done criteria

The rented host can run the configured control/moderate experiments with checked inputs and resumable checkpoints; local and remote smoke results are available before starting full training.

## Progress log

- 2026-10-28: Started. User selected RTX 5060 Ti 16 GB offer and supplied SSH connection. No rental or purchase is performed by Codex.

- 2026-10-28: Completed. Remote integrity checks, 20 focused tests and both GPU smoke runs passed. Full training verified through epoch 7/30; effective train list contains 9,249 images and 25,364 boxes, excluding 580 uncertain empty-label frames.
