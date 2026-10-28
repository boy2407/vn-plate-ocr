"""Compare global CCT-XS v2 with a saved OCR model on one test manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "custom_alpr/data/ocr/kaggle-v9-prepared/test.csv"
DEFAULT_REFERENCE_PREDICTIONS = (
    ROOT
    / "custom_alpr/outputs/train-custom-ocr/run-20261028-04-parseq-aug-noise/analysis-test/"
    / "confidence-predictions.csv"
)
DEFAULT_OUTPUT = ROOT / "custom_alpr/outputs/train-custom-ocr/cct-xs-v2-global-test-comparison"


def edit_distance(reference: str, prediction: str) -> int:
    """Return Levenshtein distance for character error rate."""
    previous = list(range(len(prediction) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(prediction, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def score(rows: list[dict[str, str]], prediction_key: str) -> dict[str, int | float]:
    exact = sum(row[prediction_key] == row["reference"] for row in rows)
    errors = sum(edit_distance(row["reference"], row[prediction_key]) for row in rows)
    characters = sum(len(row["reference"]) for row in rows)
    return {
        "exact_plates": exact,
        "exact_accuracy": exact / len(rows),
        "edit_errors": errors,
        "reference_characters": characters,
        "cer": errors / characters,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-predictions", type=Path, default=DEFAULT_REFERENCE_PREDICTIONS)
    parser.add_argument("--model", default="cct-xs-v2-global-model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.manifest, args.reference_predictions):
        if not path.is_file():
            raise SystemExit(f"Missing input file: {path}")

    from fast_plate_ocr import LicensePlateRecognizer

    with args.manifest.open(encoding="utf-8", newline="") as manifest_file:
        manifest_rows = list(csv.DictReader(manifest_file))
    with args.reference_predictions.open(encoding="utf-8", newline="") as predictions_file:
        reference_rows = {row["image"]: row for row in csv.DictReader(predictions_file)}
    recognizer = LicensePlateRecognizer(args.model, device=args.device)
    results: list[dict[str, str]] = []
    for index, row in enumerate(manifest_rows, 1):
        image_path = args.manifest.parent / row["image_path"]
        if not image_path.is_file():
            raise ValueError(f"Missing image: {image_path}")
        reference = reference_rows.get(image_path.name)
        if reference is None:
            raise ValueError(f"Reference prediction mismatch for {image_path.name}")
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        prediction = recognizer.run_one(rgb, return_confidence=True)
        results.append(
            {
                "image": image_path.name,
                "reference": row["plate_text"],
                "cct_prediction": prediction.plate,
                "reference_model_prediction": reference["prediction"],
            }
        )
        if index % 200 == 0:
            print(f"Evaluated {index}/{len(manifest_rows)}")

    paired = {
        "both_correct": sum(
            row["cct_prediction"] == row["reference"]
            and row["reference_model_prediction"] == row["reference"]
            for row in results
        ),
        "reference_model_only_correct": sum(
            row["reference_model_prediction"] == row["reference"]
            and row["cct_prediction"] != row["reference"]
            for row in results
        ),
        "cct_only_correct": sum(
            row["cct_prediction"] == row["reference"]
            and row["reference_model_prediction"] != row["reference"]
            for row in results
        ),
        "both_wrong": sum(
            row["cct_prediction"] != row["reference"]
            and row["reference_model_prediction"] != row["reference"]
            for row in results
        ),
    }
    summary = {
        "test_samples": len(results),
        "manifest": str(args.manifest),
        "reference_model": str(args.reference_predictions),
        "cct_model": args.model,
        "input_note": "Same RGB crop files and labels; each recognizer uses its own configured preprocessing.",
        "cct_alphabet_note": "Global CCT config alphabet is 0-9, A-Z, underscore; the dataset includes Đ.",
        "reference_model_metrics": score(results, "reference_model_prediction"),
        "cct_metrics": score(results, "cct_prediction"),
        "paired_cases": paired,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
