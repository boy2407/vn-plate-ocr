# Draw a detailed diagram of the 3 encoder layers

## Goal
Show step by step how data flows through the PlateOCR2D encoder: 3 stacked layers, and inside one layer LayerNorm, Q/K/V, 4 attention heads, concat, residual, feed-forward, ending in memory 128×192.

## Checklist
- [x] Read encoder settings from `custom_alpr/custom_ocr.py`.
- [x] Draw the diagram as an artifact page (Vietnamese) with formulas and parameter counts.
- [x] Share the link with the user.

## Done criteria
The user has a link to a diagram whose shapes, order of operations and parameter counts match the code.

## Progress log
- 2026-10-28: Started at user's request.
- 2026-10-28: Published https://claude.ai/artifact/MSbXthWLQhGXobuBQiGWEn (stack of 3 layers; one layer in detail; formulas; 444,864 params per layer, 1,334,976 for the encoder incl. final LayerNorm, counted by hand from layer sizes). Done.
