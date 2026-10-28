# Task 046: Stream actual clip inference in real time

## Goal

Show annotated frames and the best plate crop/OCR while the model processes an uploaded clip, rather than waiting for a completed output video.

## Checklist

- [x] Add paced video inference and stream newly inferred frames to the browser.
- [x] Show latest crop/text and actual processing FPS, latency and skipped frame counts.
- [x] Preserve downloadable artifacts and end-of-job review.
- [x] Verify live frames arrive before completion and measure real inference speed.
- [x] Restart and open the updated website.

## Done criteria

During a running clip job the browser shows actual new inference frames and OCR; if inference cannot keep up, old frames are skipped and the UI reports this.

## Progress log

- 2026-10-28: User explicitly requested real time processing. Retain one highest-confidence box per frame from task 045.
- 2026-10-28: Added MJPEG of actual inferred frames, live crop/text, source clock pacing, stale-frame skipping, FPS/frame-time counters and resume-by-job URL. Models warm once before the clip clock starts. Original crops retain resolution; display/export video has maximum side 1280 to reduce 4K rendering cost.
- 2026-10-28: Confirmed live JPEG arrives while status is running, before output completion. Verified in browser with the user's uploaded 3840×2160 / 25 FPS clip, not a simulated replay. Final run: 288 source frames, 265 processed, 23 skipped, 43 OCR results; live update rate 23.0 FPS on MPS. 40.0 processing FPS includes frames with no detection; total job time 16.13 s also includes warmup and final H.264 export.
- 2026-10-28: 38 tests, Ruff check/format and real HTTP/video checks passed. Live screenshot and measurements are in `custom_alpr/outputs/web-demo/`. Server remains running at http://127.0.0.1:8766; browser opened on the completed user clip.
