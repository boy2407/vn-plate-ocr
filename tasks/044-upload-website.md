# Task 044: Upload website with enlarged plate crops

## Goal

Run the trained YOLO and OCR model C on uploaded images/MP4, and compare enlarged original plate crops with OCR text in a browser.

## Checklist

- [x] Add local upload server and background inference jobs.
- [x] Build responsive UI with annotated media, enlarged crops, confidence, and video timestamps.
- [x] Check real image and video uploads, invalid input, tests and lint.
- [x] Open website and document startup.

## Done criteria

An image or MP4 uploaded in the website produces an annotated result and enlarged original plate crops next to recognized text.

## Progress log

- 2026-10-28: Started. Reuse trained YOLO + model C and existing video inference; host locally because weights run in Python/PyTorch on this machine.
- 2026-10-28: Added Flask backend, single model worker, upload validation, progress polling, annotated media, per-frame crop PNGs, paginated results and CSV download. Models are cached after the first job.
- 2026-10-28: Added responsive frontend with dropzone, confidence slider, sample clip, enlarged crop dialog, smoothing toggle and timestamp seek. Added web dependency extra and packaged static assets.
- 2026-10-28: Real HTTP checks passed for JPG (1 detection), uploaded MP4 (212 detections), no-plate PNG, malformed JPG, unsupported suffix, invalid confidence, media range requests and path traversal rejection. Existing 33 tests and Ruff passed.
- 2026-10-28: Opened http://127.0.0.1:8766 in Codex browser, ran sample through UI and checked enlarged crop dialog. Server is running in a retained local session; startup instructions are in README.
