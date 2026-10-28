# Preview HSV hue/color shifts

## Goal
Show how HSV hue changes affect different plate background colors, alongside the existing value/saturation lighting examples.

## Checklist
- [x] Use white, blue, yellow, and red plate examples.
- [x] Generate original and hue-shifted visual comparisons.
- [x] Save a labeled contact sheet and parameters.

## Progress log
- 2026-10-27: Generated color-only previews with OpenCV-scale hue offsets of -8 and +8; source images are not used for training.
- 2026-10-27: Saved the contact sheet at `docs/models/plate-ocr/dataset-review/hue-shift-contact-sheet.jpg`; H±8 is an exaggerated visual comparison, not a final augmentation setting.
