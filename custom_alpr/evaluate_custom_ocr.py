"""Evaluate a saved PlateOCR2D checkpoint and count character-level false positives."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from custom_alpr.custom_ocr import build_model
from custom_alpr.custom_ocr_data import PlateOCRDataset, plate_ocr_collate

DEFAULT_DATA = Path("custom_alpr/data/ocr/kaggle-v9-prepared")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA / "test.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda[:index]")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers cannot be negative")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if not args.manifest.is_file():
        parser.error(f"Manifest does not exist: {args.manifest}")
    return args


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lcs_length(reference: str, prediction: str) -> int:
    """Return the number of exactly matched characters in a longest common subsequence."""
    previous = [0] * (len(prediction) + 1)
    for expected in reference:
        current = [0]
        for column, actual in enumerate(prediction, 1):
            current.append(
                previous[column - 1] + 1
                if expected == actual
                else max(previous[column], current[-1])
            )
        previous = current
    return previous[-1]


def edit_distance(reference: str, prediction: str) -> int:
    """Return Levenshtein distance for the standard character error rate."""
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


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dataset = PlateOCRDataset(args.manifest)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=plate_ocr_collate,
    )
    predictions: list[tuple[str, str, str]] = []
    with torch.inference_mode():
        for images, _, _, references in loader:
            decoded = model.generate(images.to(device))
            predictions.extend(
                (str(dataset.samples[index][0].name), reference, prediction)
                for index, (reference, prediction) in enumerate(
                    zip(references, decoded, strict=True),
                    start=len(predictions),
                )
            )

    true_positives = false_positives = false_negatives = edit_errors = reference_chars = 0
    exact_plates = 0
    for _, reference, prediction in predictions:
        matches = lcs_length(reference, prediction)
        true_positives += matches
        false_positives += len(prediction) - matches
        false_negatives += len(reference) - matches
        edit_errors += edit_distance(reference, prediction)
        reference_chars += len(reference)
        exact_plates += reference == prediction

    predicted_chars = true_positives + false_positives
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "manifest": str(args.manifest),
        "device": str(device),
        "test_samples": len(predictions),
        "reference_characters": reference_chars,
        "predicted_characters": predicted_chars,
        "character_true_positives": true_positives,
        "character_false_positives": false_positives,
        "character_false_negatives": false_negatives,
        "false_positive_definition": (
            "Predicted characters not matched exactly to a reference character by longest "
            "common subsequence; includes insertions and substitutions."
        ),
        "character_precision": true_positives / max(predicted_chars, 1),
        "character_recall": true_positives / max(reference_chars, 1),
        "character_f1": 2
        * true_positives
        / max(2 * true_positives + false_positives + false_negatives, 1),
        "exact_plate_accuracy": exact_plates / max(len(predictions), 1),
        "cer": edit_errors / max(reference_chars, 1),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "character-fp-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "test-predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as prediction_file:
        writer = csv.writer(prediction_file)
        writer.writerow(("image", "reference", "prediction", "exact_match"))
        writer.writerows(
            (name, reference, prediction, reference == prediction)
            for name, reference, prediction in predictions
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
