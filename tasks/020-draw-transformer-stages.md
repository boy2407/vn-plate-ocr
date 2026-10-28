# Draw encoder and autoregressive decoder stages

## Goal
Create a focused diagram of the PlateOCR2D Transformer encoder and decoder stages.

## Checklist
- [x] Confirm layer structure and tensor shapes from code and notes.
- [x] Draw the encoder and decoder stages with their repeated sublayers.
- [x] Link the diagram from the architecture note and mark the task complete.

## Done criteria
The project has a clear diagram showing 3 encoder layers, 2 decoder layers, cross-attention to image tokens, and autoregressive character generation.

## Progress log
- 2026-10-28: Confirmed the implementation uses 128 visual tokens of dimension 192, a 3-layer encoder, and a 2-layer causal decoder.
- 2026-10-28: Added the dedicated Mermaid diagram and linked it from the layout-free architecture note.
