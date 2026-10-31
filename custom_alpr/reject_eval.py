"""Evaluate OCR rejection scores and camera-held-out precision/coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from custom_alpr.custom_ocr import DECODER_CHAR_TO_INDEX, build_model, layout_decode
from custom_alpr.custom_ocr_data import PlateOCRDataset, plate_ocr_collate

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "custom_alpr/outputs/train-custom-ocr/run-20261029-09-parseq-f-compress/best.pt"
EV1 = ROOT / "custom_alpr/outputs/train-custom-ocr/run-20261030-10-parseq-e-v1-video/best.pt"
CONFIRMED = ROOT / "custom_alpr/data/ocr/collected-v1/confirmed-tight.csv"
KAGGLE = ROOT / "custom_alpr/data/ocr/kaggle-v9-prepared/test.csv"
SCORE_NAMES = (
    "geometric_mean",
    "minimum_probability",
    "weakest_position_margin",
    "free_layout_agreement",
)


def score_positions(probabilities: torch.Tensor, free: str, layout: str) -> dict[str, float]:
    """Score one crop using its final per-position token probabilities."""
    length = len(free)
    if not length:
        return dict.fromkeys(SCORE_NAMES[:3], 0.0) | {SCORE_NAMES[3]: float(free == layout)}
    ids = torch.tensor([DECODER_CHAR_TO_INDEX[c] for c in free], device=probabilities.device)
    positions = probabilities[0, :length]
    selected = positions.gather(1, ids[:, None]).squeeze(1)
    top_two = positions.topk(2, dim=1).values
    weakest = int(selected.argmin())
    return {
        "geometric_mean": float(selected.clamp_min(1e-12).log().mean().exp()),
        "minimum_probability": float(selected.min()),
        "weakest_position_margin": float(top_two[weakest, 0] - top_two[weakest, 1]),
        "free_layout_agreement": float(free == layout),
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not total:
        return [0.0, 1.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z * ((proportion * (1 - proportion) / total + z * z / (4 * total**2)) ** 0.5) / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - radius)
    upper = 1.0 if successes == total else min(1.0, center + radius)
    return [lower, upper]


def choose_threshold(rows: list[dict], score: str, target: float) -> float | None:
    candidates = sorted({row[score] for row in rows})
    valid = []
    for threshold in candidates:
        accepted = [row for row in rows if row[score] >= threshold]
        if accepted and sum(row["correct"] for row in accepted) / len(accepted) >= target:
            valid.append(threshold)
    return min(valid) if valid else None


def summarize_threshold(rows: list[dict], score: str, threshold: float | None) -> dict:
    if threshold is None:
        return {
            "threshold": None,
            "accepted": 0,
            "precision": None,
            "precision_ci": None,
            "coverage": 0.0,
            "coverage_ci": wilson(0, len(rows)),
        }
    accepted = [row for row in rows if row[score] >= threshold]
    correct = sum(row["correct"] for row in accepted)
    return {
        "threshold": threshold,
        "accepted": len(accepted),
        "correct": correct,
        "precision": correct / len(accepted) if accepted else None,
        "precision_ci": wilson(correct, len(accepted)),
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "coverage_ci": wilson(len(accepted), len(rows)),
    }


def risk_coverage(rows: list[dict], score: str) -> list[dict]:
    """Return empirical accuracy and Wilson interval at each accepted coverage level."""
    curve = []
    for threshold in sorted({row[score] for row in rows}, reverse=True):
        accepted = [row for row in rows if row[score] >= threshold]
        correct = sum(row["correct"] for row in accepted)
        curve.append(
            {
                "threshold": threshold,
                "accepted": len(accepted),
                "coverage": len(accepted) / len(rows) if rows else 0.0,
                "accuracy": correct / len(accepted) if accepted else None,
                "accuracy_ci": wilson(correct, len(accepted)),
            }
        )
    return curve


def camera_index() -> dict[str, dict]:
    result = {}
    for metadata in (ROOT / "custom_alpr/data/collected").glob("*/metadata.csv"):
        with metadata.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                result[row["crop_id"]] = row
    return result


@torch.inference_mode()
def infer(model, manifest: Path, name: str, batch_size: int) -> list[dict]:
    dataset = PlateOCRDataset(manifest)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=plate_ocr_collate)
    metadata = camera_index() if manifest == CONFIRMED else {}
    rows, index = [], 0
    for images, _, _, references in loader:
        free, confidence, probabilities = model.generate_with_confidence(
            images, return_positions=True
        )
        fitted, _ = layout_decode(probabilities)
        for local, (reference, prediction, geometric, layout) in enumerate(
            zip(references, free, confidence, fitted, strict=True)
        ):
            path, _ = dataset.samples[index]
            with Image.open(path) as image:
                height = image.height
            details = metadata.get(path.stem, {})
            probs = probabilities[local].detach().cpu()
            scores = score_positions(probs.unsqueeze(0), prediction, layout)
            # The per-position categorical distribution is retained for Phase B.
            char_probs = [probs[pos, 3:].tolist() for pos in range(len(prediction))]
            rows.append(
                {
                    "model": name,
                    "dataset": "confirmed" if manifest == CONFIRMED else "kaggle",
                    "image": str(path),
                    "height": height,
                    "reference": reference,
                    "prediction": prediction,
                    "layout_prediction": layout,
                    "correct": prediction == reference,
                    **scores,
                    "camera": details.get("camera", "kaggle"),
                    "track_id": details.get("track_id", ""),
                    "quality": float(details["quality"]) if details.get("quality") else 1.0,
                    "char_probabilities": char_probs,
                }
            )
            rows[-1]["geometric_mean"] = geometric
            index += 1
    return rows


def camera_report(rows: list[dict]) -> dict:
    # Hanoi is the locked test camera. It must not select confidence thresholds.
    cameras = ("danang-ptz", "saigon-drive")
    report = {}
    for score in SCORE_NAMES:
        report[score] = {"risk_coverage": {}}
        for camera in cameras:
            camera_rows = [row for row in rows if row["camera"] == camera]
            report[score]["risk_coverage"][camera] = risk_coverage(camera_rows, score)
        for target in (0.98, 0.99):
            target_report = {}
            for calibration, heldout in ((cameras[0], cameras[1]), (cameras[1], cameras[0])):
                calibration_rows = [row for row in rows if row["camera"] == calibration]
                heldout_rows = [row for row in rows if row["camera"] == heldout]
                threshold = choose_threshold(calibration_rows, score, target)
                target_report[f"{calibration}_to_{heldout}"] = {
                    "calibration_n": len(calibration_rows),
                    "heldout_n": len(heldout_rows),
                    "calibration_threshold": threshold,
                    "heldout": summarize_threshold(heldout_rows, score, threshold),
                }
            report[score][str(target)] = target_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "custom_alpr/outputs/reject-eval-076")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--model",
        action="append",
        help="name=checkpoint.pt; overrides the default E and E-v1 models",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_reports = {}
    models = (
        [(value.split("=", 1)[0], Path(value.split("=", 1)[1])) for value in args.model]
        if args.model
        else [("E", E), ("E-v1", EV1)]
    )
    for label, checkpoint_path in models:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = build_model(checkpoint).eval()
        model.load_state_dict(checkpoint["model_state_dict"])
        for dataset_name, manifest in (("kaggle", KAGGLE), ("confirmed", CONFIRMED)):
            rows = infer(model, manifest, label, args.batch_size)
            output = args.output / f"{label}-{dataset_name}.json"
            output.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            if dataset_name == "confirmed":
                all_reports[label] = camera_report(rows)
            else:
                buckets = {"<16": [], "16-24": [], "24-32": [], ">=32": []}
                for row in rows:
                    key = (
                        "<16"
                        if row["height"] < 16
                        else "16-24"
                        if row["height"] < 24
                        else "24-32"
                        if row["height"] < 32
                        else ">=32"
                    )
                    buckets[key].append(row)
                all_reports[f"{label}-kaggle-height"] = {
                    key: {
                        "n": len(group),
                        "correct": sum(r["correct"] for r in group),
                        "exact": sum(r["correct"] for r in group) / len(group) if group else None,
                        "exact_ci": wilson(sum(r["correct"] for r in group), len(group)),
                    }
                    for key, group in buckets.items()
                }
    (args.output / "report.json").write_text(
        json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
