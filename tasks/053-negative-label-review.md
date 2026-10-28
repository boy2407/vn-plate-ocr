# Task 053: Quarantine unreviewed empty-label traffic frames

## Goal

Avoid treating unreviewed full traffic images with blurred plate regions as guaranteed plate-free negatives in the new robustness run.

## Checklist

- [x] Inspect sampled empty-label training frames.
- [x] Keep original files intact and exclude uncertain empty-label frames from the default training list.
- [x] Record source/effective counts and exact exclusions; preserve validation/test.
- [x] Test runtime lists and document remaining manual review.

## Done criteria

The default bundle does not train on the 580 uncertain empty-label traffic frames; these are retained for review rather than silently relabeled.

## Progress log

- 2026-10-28: Sample review shows full traffic scenes with deliberately blurred plate locations. This does not establish that every empty-label image is wrong; it means they are not confirmed plate-free. Use a conservative quarantine pending review. User clip-derived negatives remain outside training.

- 2026-10-28: Completed. Remote integrity checks, 20 focused tests and both GPU smoke runs passed. Full training verified through epoch 7/30; effective train list contains 9,249 images and 25,364 boxes, excluding 580 uncertain empty-label frames.
