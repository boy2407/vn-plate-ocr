# Task 066: Make CCT comparison reproducible and push results

## Goal

Commit a reusable evaluation script and a tracked summary comparing global CCT-XS v2 with PARSeq model C on the same held-out OCR test split.

## Checklist

- [x] Add a CLI script that reruns CCT inference and computes paired metrics against saved model C predictions.
- [x] Add a concise tracked report with results and limitations.
- [x] Run the script and verify its report matches the completed evaluation.
- [x] Commit and push to `origin/main`.

## Done criteria

The repository contains the reproducible evaluator and result summary, and the commit is pushed to the configured remote.

## Progress log

- 2026-10-28: Started after confirming task 047 notes were already committed and output artifacts were ignored by Git.
- 2026-10-28: Added the reusable evaluator and reran it on the current 1,202-image test manifest; updated the report after accounting for the reviewed-label split change.
- 2026-10-28: Pushed implementation and report to `origin/main` in commit `0a438db`.
