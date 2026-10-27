"""CLI: python -m custom_alpr --help (run from the project root)."""

# Keep ML framework imports lazy so preparation and --help remain lightweight.
# ruff: noqa: PLC0415

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Convert labels and create blank OCR CSVs")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=ROOT / "data")
    detector = commands.add_parser("train-detector")
    detector.add_argument("--data", type=Path, required=True)
    detector.add_argument("--epochs", type=int, default=100)
    detector.add_argument("--batch", type=int, default=16)
    detector.add_argument("--device", default="cpu", help="cpu, mps, or CUDA index e.g. 0")
    ocr = commands.add_parser("train-ocr")
    ocr.add_argument("--train", type=Path, required=True)
    ocr.add_argument("--val", type=Path, required=True)
    ocr.add_argument("--epochs", type=int, default=150)
    ocr.add_argument("--batch", type=int, default=64)
    export = commands.add_parser("export-ocr")
    export.add_argument("--model", type=Path, required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--detector", type=Path, required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--confidence", type=float, default=0.4)
    predict.add_argument("--output", type=Path, default=ROOT / "outputs" / "prediction.jpg")
    evaluate = commands.add_parser("evaluate-ocr")
    evaluate.add_argument("--annotations", type=Path, required=True)
    for command in (predict, evaluate):
        command.add_argument("--ocr", type=Path, required=True)
    for command in (ocr, export, predict, evaluate):
        command.add_argument("--plate-config", type=Path, default=ROOT / "configs" / "plate.yaml")
    ocr.add_argument("--model-config", type=Path, default=ROOT / "configs" / "cct_xs_v2.yaml")
    for command in (detector, ocr):
        command.add_argument(
            "--output", type=Path, default=ROOT / "outputs" / command.prog.split()[-1]
        )
    return cli


def main() -> None:
    # Imports stay inside commands so --help works even before dependencies are installed.
    args = parser().parse_args()
    if args.command == "prepare":
        from custom_alpr.data import prepare

        report = prepare(args.source, args.output)
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report["splits"], indent=2))
        print(
            f"Removed {len(report['duplicates_removed'])} duplicate images; "
            f"rejected {len(report['rejected'])}. See {args.output / 'report.json'}"
        )
        print("OCR CSVs contain blank labels: transcribe them before training.")
    elif args.command == "train-detector":
        from ultralytics import YOLO

        YOLO("yolov8n.pt").train(
            data=str(args.data.resolve()),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=640,
            device=args.device,
            project=str(args.output.resolve()),
            name="plates",
            seed=42,
        )
    elif args.command == "train-ocr":
        import yaml

        from custom_alpr.data import read_annotations

        config = yaml.safe_load(args.plate_config.read_text())
        for path in (args.train, args.val):
            read_annotations(path, config["max_plate_slots"])
        run_ocr(
            [
                "train",
                "--model-config-file",
                str(args.model_config.resolve()),
                "--plate-config-file",
                str(args.plate_config.resolve()),
                "--annotations",
                str(args.train.resolve()),
                "--val-annotations",
                str(args.val.resolve()),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch),
                "--output-dir",
                str(args.output.resolve()),
                "--validate-dataset",
                "error",
            ]
        )
    elif args.command == "export-ocr":
        run_ocr(
            [
                "export",
                "--model",
                str(args.model.resolve()),
                "--plate-config-file",
                str(args.plate_config.resolve()),
                "--format",
                "onnx",
            ]
        )
    elif args.command == "predict":
        import cv2

        from custom_alpr.pipeline import PlateReader

        if not 0 <= args.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        image = cv2.imread(str(args.image))
        reader = PlateReader(args.detector, args.ocr, args.plate_config, args.device)
        results = reader.predict(image, args.confidence)
        for result in results:
            x1, y1, x2, y2 = result["box"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                result["text"],
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), image):
            raise OSError(f"Cannot write {args.output}")
        args.output.with_suffix(".json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
    elif args.command == "evaluate-ocr":
        import yaml
        from fast_plate_ocr import LicensePlateRecognizer

        from custom_alpr.data import read_annotations
        from custom_alpr.pipeline import edit_distance

        config = yaml.safe_load(args.plate_config.read_text())
        samples = read_annotations(args.annotations, config["max_plate_slots"])
        recognizer = LicensePlateRecognizer(
            onnx_model_path=str(args.ocr), plate_config_path=str(args.plate_config), device="cpu"
        )
        correct = errors = characters = 0
        for image, expected in samples:
            # Paths let the upstream recognizer handle its own color conversion.
            actual = recognizer.run_one(str(image)).plate
            correct += actual == expected
            errors += edit_distance(expected, actual)
            characters += len(expected)
        print(
            json.dumps(
                {
                    "plates": len(samples),
                    "exact_plate_accuracy": correct / len(samples),
                    "character_error_rate": errors / characters,
                },
                indent=2,
            )
        )


def run_ocr(arguments: list[str]) -> None:
    env = dict(os.environ)
    env.setdefault("KERAS_BACKEND", "tensorflow")
    subprocess.run(["fast-plate-ocr", *arguments], check=True, env=env)


if __name__ == "__main__":
    main()
