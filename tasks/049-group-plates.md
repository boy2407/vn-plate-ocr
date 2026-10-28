# Task 049: Group repeated plate observations across video frames

## Goal

Show one review card per spatially continuous plate track instead of one card per frame. Retain raw observations and choose an observed consensus text and representative crop.

## Checklist

- [x] Associate nearby detections over time; vote observed OCR strings without inventing text.
- [x] Add grouped gallery/CSV, observation counts and first/last timestamps; keep frame view.
- [x] Apply grouping to existing user results and persist completed job metadata across restarts.
- [x] Check repeated plates, OCR fluctuations, nearby different vehicles, gaps and representative alignment.
- [x] Run checks and reopen the user's clip results.

## Done criteria

The repeated `51H21510` observations in the supplied clip are grouped while frame-level evidence remains accessible.

## Progress log

- 2026-10-28: Started after user screenshot showed the same plate once per processed frame.
- 2026-10-28: Added spatial/time tracks with majority voting and best observed matching crop. Gallery additionally merges equal consensus strings across tracks; track IDs remain available. This groups OCR strings, not verified vehicle identities.
- 2026-10-28: Added grouped/frame selector, `plates.json`/CSV, observation counts, intervals and agreement. Persisted seven existing jobs and added completed-job recovery for future restarts. Original frame evidence is retained.
- 2026-10-28: User runs: 185 raw observations → 34 grouped cards; latest 188 → 37. `51H21510` appears once in the latest gallery, representing 102 observations with 89% agreement. Browser verified switching 188 frame records / 37 groups.
- 2026-10-28: New real clip run succeeded with grouping and live streaming, 186 observations → 34 groups. 66 tests plus Ruff, format, JS syntax and diff checks passed. Website remains running at port 8766.
