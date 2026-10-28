# Extract high-confidence incorrect test images

## Goal
Copy test images with confidence >= 0.90 and an incorrect OCR prediction into a reviewable folder, with metadata and per-plate character edit distance.

## Checklist
- [x] Read existing confidence predictions and test manifest.
- [x] Copy matching original images and create review metadata.
- [x] Verify the extracted count and update task tracking.

## Done criteria
All high-confidence incorrect test images are available in a separate folder with their reference text, prediction, confidence, and character error count.

## Progress log
- 2026-10-28: Started extraction using the report's confidence threshold of 0.90.
- 2026-10-28: Copied 79 test images and wrote CSV metadata with character edit distance.
