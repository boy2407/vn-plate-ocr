# Build layout-free 2D Transformer OCR

## Goal
Build a custom OCR model that consumes full one/two-row plate crops and learns the text sequence without explicit single/double layout labels.

## Checklist
- [x] Implement a 2D visual Transformer encoder and autoregressive text decoder.
- [x] Add text target encoding, sequence loss, and greedy decoding.
- [x] Document model tensor flow and how full-plate annotations are used.
- [x] Run required project checks where the environment permits.
- [x] Update task overview and mark completion.

## Progress log
- 2026-10-27: Started from the existing residual BlurPool CNN baseline. Chose a 2D encoder-decoder over rasterized CTC because raster order does not guarantee top-row-then-bottom-row reading order.
- 2026-10-27: Implemented `PlateOCR2D`, BOS/EOS target encoding, teacher-forced cross-entropy, and greedy generation. Added the layout-free model diagram and README notes. Syntax compilation and diff checks pass. Runtime forward/training and project tests could not run because host Python 3.14 lacks PyTorch/pytest and is outside the project's supported Python range; Ruff is also unavailable. Training loop and pipeline integration remain the next implementation stage.
