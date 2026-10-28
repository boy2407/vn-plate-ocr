# Download Kaggle OCR data and prepare training set

## Goal
Download the public Vietnamese license plate OCR dataset and prepare deduplicated, stratified OCR training/validation data for the custom `PlateOCR2D` model.

## Checklist
- [x] Download and inspect the Kaggle archive.
- [x] Convert source labels into the model's supported transcription format.
- [x] Deduplicate before assigning train/validation/test splits.
- [x] Prepare reproducible training manifests and preprocessing documentation.
- [x] Record limitations and update project status.

## Done criteria
The dataset is available locally in a reproducible prepared layout with labels, safe splits, and documented next steps for training.

## Progress log
- 2026-10-28: Confirmed dataset was previously only audited and 12 review samples were checked in; full data was not downloaded. Kaggle CLI is installed and 38 GB disk space is available.
- 2026-10-28: Downloaded the 35.4 MB Kaggle archive (12,190 labeled rows). Validated image signatures and labels; removed 255 exact duplicate rows and excluded 2 conflicting duplicate image groups. Created deterministic source × Type stratified manifests: 9,541 train / 1,187 validation / 1,203 test. Documented that training Dataset and loop remain to be implemented.
