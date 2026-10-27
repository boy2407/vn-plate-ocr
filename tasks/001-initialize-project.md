# Initialize standalone project

## Goal
Independent YOLOv8n + custom CCT OCR project, extracted from fast-alpr.

## Checklist
- [x] Copy code, configs and prepared data.
- [x] Add independent project metadata and instructions.
- [x] Initialize Git and verify execution.

## Done criteria
CLI and tests pass from this repository; dataset config points into this project.

## Progress log
- 2026-10-26: Created standalone project. Training still requires manual OCR transcriptions and a training environment.
- 2026-10-26: Initialized independent Git repository on main; added packaging metadata and CLI entry point. Copied prepared images and OCR CSVs, updated detection dataset path to this checkout. Verified 14 tests, Ruff check/format and CLI help from the new root using an existing test interpreter. No Git remote configured; no training run or model weights produced.
