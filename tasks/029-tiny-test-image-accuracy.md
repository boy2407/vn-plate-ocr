# Measure OCR accuracy on tiny test images

## Goal
Count test images with either dimension below 30 pixels and how many predictions exactly match the dataset label.

## Checklist
- [x] Read the test manifest and prediction report.
- [x] Count images under 30 pixels on at least one side and exact matches.

## Done criteria
Report the image count and exact-match count with the dimension rule stated.

## Progress log
- 2026-10-28: Found 321 test images with width < 30 px or height < 30 px; 197 exact matches against the dataset labels.
