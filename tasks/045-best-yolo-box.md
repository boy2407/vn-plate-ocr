# Task 045: Send only the best YOLO box to OCR

## Goal

For each uploaded image or video frame, pass only the valid box with highest YOLO confidence to OCR. Apply the same rule to the web app and video CLI.

## Checklist

- [x] Share best-box selection across web and video demo.
- [x] Update UI and documentation.
- [x] Check OCR receives one crop, no-detection behavior, and real video; run required checks.
- [x] Restart the web server with updated behavior.

## Done criteria

Each image/frame produces at most one OCR result, chosen by YOLO confidence after the configured detection threshold.

## Progress log

- 2026-10-28: Started; interpret best box as highest YOLO confidence per image/frame.
- 2026-10-28: Shared `select_best_box` validates/clips boxes and picks the highest YOLO confidence before OCR. Web and CLI both use it; no valid box skips OCR. UI and README describe the rule.
- 2026-10-28: Added regression checks for unsorted scores, invalid boxes, one-crop OCR and skipping OCR on empty results. All 38 tests and Ruff passed. CLI sample now yields 60 results over 60 frames instead of 212. Live web results also contain at most one result per processed frame.
- 2026-10-28: Restarted the local server at port 8766 with the updated selection policy.
