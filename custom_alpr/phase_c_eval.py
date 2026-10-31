"""Compare E with the Phase C model on Kaggle and original/refined video crops."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from custom_alpr.analyze_custom_ocr import edit_distance
from custom_alpr.confusion_report import substitutions
from custom_alpr.custom_ocr import build_model
from custom_alpr.custom_ocr_data import PlateOCRDataset, plate_ocr_collate
from custom_alpr.reject_eval import E, CONFIRMED, KAGGLE, camera_index, wilson

ROOT = Path(__file__).resolve().parents[1]
REFINE_ROWS = ROOT / "custom_alpr/outputs/refine-confirmed-070/rows.json"


def percentile_interval(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def bootstrap_metrics(base_rows: list[dict], new_rows: list[dict], seed: int = 73) -> dict:
    base = {row["sample_id"]: row for row in base_rows}
    new = {row["sample_id"]: row for row in new_rows}
    ids = sorted(base.keys() & new.keys())
    if not ids:
        return {"accuracy_delta_ci": None, "cer_delta_ci": None}
    generator = np.random.default_rng(seed)
    base_exact = np.array([base[key]["correct"] for key in ids], dtype=np.float64)
    new_exact = np.array([new[key]["correct"] for key in ids], dtype=np.float64)
    base_errors = np.array([base[key]["edits"] for key in ids], dtype=np.float64)
    new_errors = np.array([new[key]["edits"] for key in ids], dtype=np.float64)
    lengths = np.array([len(base[key]["reference"]) for key in ids], dtype=np.float64)
    accuracy_deltas, cer_deltas = [], []
    for _ in range(100_000):
        sample = generator.integers(0, len(ids), size=len(ids))
        accuracy_deltas.append(float((new_exact[sample] - base_exact[sample]).mean()))
        cer_deltas.append(
            float((new_errors[sample].sum() - base_errors[sample].sum()) / lengths[sample].sum())
        )
    return {
        "paired_samples": len(ids),
        "accuracy_delta_ci": percentile_interval(accuracy_deltas),
        "cer_delta_ci": percentile_interval(cer_deltas),
    }


def summarize(rows: list[dict]) -> dict:
    correct = sum(row["correct"] for row in rows)
    edits = sum(row["edits"] for row in rows)
    characters = sum(len(row["reference"]) for row in rows)
    generator = np.random.default_rng(73)
    cer_samples = []
    for _ in range(10_000):
        sample = generator.integers(0, len(rows), size=len(rows)) if rows else []
        chars = sum(len(rows[index]["reference"]) for index in sample)
        cer_samples.append(sum(rows[index]["edits"] for index in sample) / chars if chars else 0.0)
    return {
        "n": len(rows),
        "correct": correct,
        "exact": correct / len(rows) if rows else None,
        "exact_ci_wilson95": wilson(correct, len(rows)),
        "edits": edits,
        "characters": characters,
        "cer": edits / characters if characters else None,
        "cer_ci_bootstrap95": percentile_interval(cer_samples),
    }


def make_refined_manifest(output: Path) -> tuple[Path, dict[str, dict]]:
    rows = json.loads(REFINE_ROWS.read_text(encoding="utf-8"))
    crop_dir = output / "refined-crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    manifest = output / "confirmed-refine.csv"
    joined = {}
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "plate_text", "source"))
        writer.writeheader()
        for row in rows:
            frame_path = (
                ROOT
                / "custom_alpr/data/collected/frames"
                / f"{row['video_id']}-f{row['frame']}.png"
            )
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise FileNotFoundError(frame_path)
            x1, y1, x2, y2 = map(int, row["final_box"])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                raise ValueError(f"Empty refined crop for {row['crop_id']}: {row['final_box']}")
            image_path = crop_dir / f"{row['crop_id']}.png"
            if not cv2.imwrite(str(image_path), crop):
                raise OSError(f"Could not write {image_path}")
            writer.writerow(
                {
                    "image_path": f"refined-crops/{image_path.name}",
                    "plate_text": row["truth"],
                    "source": "youtube",
                }
            )
            joined[image_path.stem] = row
    return manifest, joined


def load_recognizer(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


@torch.inference_mode()
def read_manifest(model, manifest: Path, name: str, batch_size: int) -> list[dict]:
    dataset = PlateOCRDataset(manifest)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=plate_ocr_collate)
    metadata = camera_index()
    rows, index = [], 0
    for images, _, _, references in loader:
        predictions = model.generate(images)
        for reference, prediction in zip(references, predictions, strict=True):
            path, _ = dataset.samples[index]
            match = metadata.get(path.stem, {})
            edits = edit_distance(reference, prediction)
            rows.append(
                {
                    "dataset": name,
                    "sample_id": path.stem,
                    "image": str(path),
                    "camera": match.get("camera", "kaggle"),
                    "height": image_height(path),
                    "reference": reference,
                    "prediction": prediction,
                    "correct": prediction == reference,
                    "edits": edits,
                }
            )
            index += 1
    return rows


def image_height(path: Path) -> int:
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as image:
        return image.height


def confusion_pairs(rows: list[dict]) -> dict[str, int]:
    pairs = Counter()
    for row in rows:
        pairs.update(
            f"{true}->{read}" for true, read in substitutions(row["reference"], row["prediction"])
        )
    return dict(pairs.most_common())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Phase C checkpoint")
    parser.add_argument("--base-checkpoint", type=Path, default=E)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "custom_alpr/outputs/phase-c-076/evaluation"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:  # noqa: PLR0912
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    refine_manifest, refine_metadata = make_refined_manifest(args.output)
    models = {"E": load_recognizer(args.base_checkpoint), "C": load_recognizer(args.checkpoint)}
    manifests = {
        "original": CONFIRMED,
        "refine_box": refine_manifest,
        "kaggle_test": KAGGLE,
    }
    rows_by_model: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for model_name, model in models.items():
        for variant, manifest in manifests.items():
            print(f"READ {model_name} {variant}", flush=True)
            rows = read_manifest(model, manifest, variant, args.batch_size)
            if variant == "refine_box":
                for row in rows:
                    meta = refine_metadata.get(row["sample_id"], {})
                    row["camera"] = meta.get("camera", row["camera"])
                    row["refine_accepted"] = bool(meta.get("refined"))
            rows_by_model[model_name][variant] = rows

    report = {
        "models": {},
        "paired_deltas_C_minus_E": {},
        "confusion_pairs": {},
        "refine_fallbacks": {},
    }
    for model_name, variants in rows_by_model.items():
        report["models"][model_name] = {}
        report["confusion_pairs"][model_name] = {}
        for variant, rows in variants.items():
            groups = {"all": rows}
            if variant != "kaggle_test":
                groups.update(
                    {
                        camera: [row for row in rows if row["camera"] == camera]
                        for camera in sorted({row["camera"] for row in rows})
                    }
                )
            report["models"][model_name][variant] = {
                name: summarize(group) for name, group in groups.items()
            }
            report["confusion_pairs"][model_name][variant] = {
                name: confusion_pairs(group) for name, group in groups.items()
            }
            if variant == "refine_box":
                report["refine_fallbacks"] = {
                    camera: {
                        "n": sum(row["camera"] == camera for row in rows),
                        "fallback": sum(
                            row["camera"] == camera and not row.get("refine_accepted")
                            for row in rows
                        ),
                    }
                    for camera in sorted({row["camera"] for row in rows})
                }
    for variant in manifests:
        base_rows = rows_by_model["E"][variant]
        new_rows = rows_by_model["C"][variant]
        groups = {"all": sorted({row["camera"] for row in base_rows})}
        if variant != "kaggle_test":
            groups = {camera: [camera] for camera in sorted({row["camera"] for row in base_rows})}
        else:
            groups = {"all": ["kaggle"]}
        report["paired_deltas_C_minus_E"][variant] = {}
        for label, cameras in groups.items():
            left = [row for row in base_rows if row["camera"] in cameras]
            right = [row for row in new_rows if row["camera"] in cameras]
            report["paired_deltas_C_minus_E"][variant][label] = bootstrap_metrics(left, right)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for model_name, variants in rows_by_model.items():
        (args.output / f"{model_name}-rows.json").write_text(
            json.dumps(variants, ensure_ascii=False), encoding="utf-8"
        )
    print(f"wrote {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
