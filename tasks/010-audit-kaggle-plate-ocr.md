# Audit Kaggle Vietnamese plate OCR dataset

## Goal
Inspect the public Kaggle archive for OCR labels, one/two-line layout labels, file quality, and compatibility with the custom OCR project.

## Checklist
- [x] Read Kaggle metadata and enumerate archive files.
- [x] Inspect annotation schema, layout clues, and character compatibility.
- [x] Report whether the dataset can train one/two-line OCR directly.

## Findings
- Kaggle dataset v9, Apache 2.0, 2024-11-11; archive approximately 36.5 MB.
- `cropped/` has 6,643 images and `labels/crop_labels.csv`; `generated/` has 5,547 images and `labels/gen_labels.csv`.
- Both CSVs have `Name,Label,Type`; labels are nonempty but no train/val/test split is provided.
- `Type` has values 0-3 and is not a safe one/two-row indicator: sampled one-row and two-row images have the same Type values.
- Sampled images include both single and double rows. `Label` stores a space separator but does not encode line layout reliably.
- 199 `Đ` characters occur in each CSV. The custom OCR alphabets were subsequently expanded to include `Đ`; the existing CCT config still needs checking before training on these labels.
- Exact byte duplicate copies found: 93 in `cropped/`, 164 in `generated/`; deduplicate before splitting to avoid leakage.
- Generated image files use `.jpg` names while their content is PNG encoded; verify decoder compatibility during import.
- Archive was inspected in memory; it was not added to the repository.

## Progress log
- 2026-10-27: Inspected the public Kaggle API metadata and current archive; verified sample row layouts visually and summarized compatibility constraints.
