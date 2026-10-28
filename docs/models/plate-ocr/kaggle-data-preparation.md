# Kaggle OCR dataset preparation

Dataset: [Vietnamese License Plate OCR](https://www.kaggle.com/datasets/topkek69/vietnamese-license-plate-ocr), Kaggle slug `topkek69/vietnamese-license-plate-ocr`, Apache-2.0.

## Local layout

The Kaggle archive is downloaded under the ignored local path:

```text
custom_alpr/data/raw/kaggle-vietnamese-license-plate-ocr/
├── cropped/
├── generated/
└── labels/
```

No source images are copied or modified by the preparation step. Prepared manifests refer to those originals and are written to the ignored path `custom_alpr/data/ocr/kaggle-v9-prepared/`.

## Reproduce

```bash
kaggle datasets download \
  -d topkek69/vietnamese-license-plate-ocr \
  -p custom_alpr/data/raw/kaggle-vietnamese-license-plate-ocr \
  --unzip
python3 custom_alpr/prepare_ocr_dataset.py
```

The preparation script uses the fixed seed `20261028` by default. Pass `--seed` to create a different split, or `--output-dir` to write to a new location. It refuses to overwrite an existing output directory.

## Preparation decisions

- Labels from `crop_labels.csv` and `gen_labels.csv` are normalized to uppercase, with spaces, hyphens, and periods removed. Vietnamese `Đ` is preserved.
- Image files are found by their label CSV names and validated by PNG/JPEG signature rather than extension. Some generated PNG files have a `.jpg` extension.
- Exact byte duplicates are removed globally before splitting. For the same image and label, the original `cropped/` record is preferred over `generated/`.
- Two image groups had conflicting transcriptions and were excluded: `car_338.jpg`/`car_352.jpg` and `car_522.jpg`/`car_553.jpg`.
- Splits are approximately 80/10/10 and stratified by source (`cropped`/`generated`) × Kaggle `Type`. `Type` is retained only to preserve source distribution; it is not a reliable single-row/double-row label.
- Every manifest contains `image_path`, `plate_text`, `source`, `type`, and `sha256`. Image paths are relative to the manifest file.

## Prepared inventory

| Split | Images |
| --- | ---: |
| Train | 9,541 |
| Validation | 1,187 |
| Test | 1,203 |
| **Total** | **11,931** |

The directory also contains `metadata.json` and `preparation-report.json` with the seed, counts, duplicate/conflict details, and source inventory.

## Training input requirements

`PlateOCR2D` expects RGB tensors shaped `[batch, 3, 64, 128]` (height × width in tensor dimensions). A training Dataset still needs to decode image contents, convert RGBA to RGB, resize to 128 × 64, convert to float, and scale pixel values to `[0, 1]`. Targets should use the existing `encode_target()` helper and be padded with `PAD_INDEX`; optimize `sequence_loss()` with teacher forcing. Apply random motion blur only to training samples.

This repository does not yet contain that Dataset/collate function or a training/validation loop. The manifests prepare the data split, but do not themselves train the model.
