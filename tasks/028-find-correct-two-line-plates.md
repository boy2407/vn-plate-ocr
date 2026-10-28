# Find correctly recognized two-line test plates

## Goal
Identify two-line plates in the test set and extract those whose OCR prediction exactly matches the reference.

## Checklist
- [x] Determine two-line labels from the test images.
- [x] Cross-reference exact OCR predictions and export matching images.
- [x] Verify the output and update task tracking.

## Done criteria
Provide a reviewable set of test images visually identified as two-line plates and exactly recognized by the model, with metadata.

## Progress log
- 2026-10-28: Started visual/layout filtering from the test set; existing evaluation has no reliable layout annotations.
- 2026-10-28: Identified two-line plate crops by their aspect ratio (width/height < 2.5), checked representative layout groups visually, and exported 688 exact matches.
