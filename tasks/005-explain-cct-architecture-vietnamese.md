# Explain CCT OCR architecture in Vietnamese

## Goal
Explain and visually show the configured detector + CCT-XS v2 recognition network, including CNN heads/tokens, downsampling, Transformer, and character decoding.

## Checklist
- [x] Read project instructions and existing architecture config/diagram.
- [x] Verify tensor shapes from config and distinguish implemented pipeline from trained weights.
- [x] Provide detailed Vietnamese diagram and explanation.

## Done criteria
The user can trace an image crop through conv/pooling, tokenization, Transformer/reducer, and per-position character prediction.

## Progress log
- 2026-10-27: Read tasks/README.md, existing architecture docs and configs. No model implementation exists in this repo; OCR architecture is configured for upstream fast-plate-ocr CCT-XS v2.
- 2026-10-27: Completed detailed layer/token diagram and clarified limitations around upstream internal details and training status. No code changes.
