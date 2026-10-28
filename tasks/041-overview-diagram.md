# Draw an overview diagram of PlateOCR2D / PARSeq

## Goal
One picture of the full model used by runs A–D: input crop → CNN → 4×32 grid → encoder ×3 → decoder ×2 → decoding, with real shapes from `custom_alpr/custom_ocr.py`.

## Checklist
- [x] Read shapes and settings from the model and training code.
- [x] Draw the diagram as an artifact page (Vietnamese).
- [x] Share the link with the user.

## Done criteria
The user has a link to a diagram whose shapes and settings match the code.

## Progress log
- 2026-10-28: Started at user's request.
- 2026-10-28: Published diagram https://claude.ai/artifact/L6rAXWmUBnyM4gUUskTtTq (input, CNN 4 blocks, 4x32 grid, encoder x3, PARSeq decoder x2, decoding; runs A-D table). Done.
