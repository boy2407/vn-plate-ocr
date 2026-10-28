# Build a first custom OCR network

## Goal
Add a project-owned baseline OCR architecture independent of fast-plate-ocr: CNN feature extractor, Transformer encoder, and CTC character output.

## Checklist
- [x] Inspect current repository instructions and OCR integration.
- [x] Implement standalone network and sequence decoding.
- [x] Document tensor flow and baseline limitations.
- [x] Run available syntax and diff checks; attempt required project checks.
- [x] Mark task complete and update overview.

## Progress log
- 2026-10-27: Started. Existing OCR training and inference call fast-plate-ocr; this task introduces a separate custom model module without silently changing the existing pipeline.
- 2026-10-27: Added CNN + Transformer Encoder + CTC model, text encoding/decoding helpers, and documented the baseline. Python syntax compilation and `git diff --check` pass. Required pytest and Ruff checks could not run because pytest/Ruff are unavailable and the host Python 3.14 is outside the project's declared `<3.14` range.
