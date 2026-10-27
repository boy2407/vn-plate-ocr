# Draw current OCR architecture

## Goal
Show the current CCT-XS v2 OCR architecture as an inline diagram based on project configuration.

## Checklist
- [x] Read current model and plate configuration.
- [x] Draw preprocessing, convolutional tokenizer, Transformer, and character output.

## Done criteria
Diagram matches configured values and separates the detector from OCR.

## Progress log
- 2026-10-26: Started diagram from current YAML configuration; using Mermaid for a static architecture view.
- 2026-10-26: Completed conceptual diagram with configured CNN channels, patch embedding, attention dimensions, reducer, post-reducer layers, and plate alphabet. No source code changed.
