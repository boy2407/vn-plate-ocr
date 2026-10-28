"""Clean validation and deterministic stress validation; explicit final test mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from custom_alpr.yolo_vast.corruptions import EFFECTS, apply_effect
from custom_alpr.yolo_vast.data import runtime_yaml, verify_inventory


def build_stress(root, output, effect, strength, count=192):
    inventory = json.loads((root / "inventory.json").read_text())
    records = [r for r in inventory["records"] if r["split"] == "val"]
    records = random.Random(42).sample(records, min(count, len(records)))
    images, labels = output / "images/val", output / "labels/val"
    images.mkdir(parents=True, exist_ok=False)
    labels.mkdir(parents=True)
    for record in records:
        source = root / "data" / record["image"]
        image = cv2.imread(str(source))
        h, w = image.shape[:2]
        boxes = []
        for line in (root / "data" / record["label"]).read_text().splitlines():
            if not line.strip():
                continue
            _, x, y, bw, bh = map(float, line.split())
            boxes.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
        token = f"42:{effect}:{source.name}".encode()
        seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
        corrupted = apply_effect(
            image, effect, np.random.default_rng(seed), strength, np.array(boxes).reshape(-1, 4)
        )
        destination = images / (source.stem + ".png")
        if not cv2.imwrite(str(destination), corrupted):
            raise OSError(str(destination))
        shutil.copy2(root / "data" / record["label"], labels / (source.stem + ".txt"))
    config = output / "dataset.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "path": str(output.resolve()),
                "train": "images/val",
                "val": "images/val",
                "names": {0: "license_plate"},
            }
        )
    )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--stress", choices=("none", "all", *EFFECTS), default="none")
    parser.add_argument("--samples", type=int, default=192)
    args = parser.parse_args()
    if args.split == "test" and args.stress != "none":
        parser.error("Stress screening is validation-only; keep test for the final fixed model")
    if args.samples < 1:
        parser.error("samples must be positive")
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    verify_inventory(root)
    model = YOLO(str(args.weights))
    result = model.val(
        data=str(runtime_yaml(root)),
        split=args.split,
        imgsz=640,
        batch=16,
        device=args.device,
        project=str(output),
        name="clean",
        plots=True,
    )
    report = {"clean": {k: float(v) for k, v in result.results_dict.items()}}
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    if args.stress == "none":
        return
    effects = EFFECTS if args.stress == "all" else (args.stress,)
    for effect in effects:
        for name, strength in (("mild", 0.6), ("moderate", 1.0)):
            key = f"{effect}-{name}"
            config = build_stress(
                root, output / "stress-data" / key, effect, strength, args.samples
            )
            result = model.val(
                data=str(config),
                imgsz=640,
                batch=16,
                device=args.device,
                project=str(output),
                name=key,
                plots=False,
            )
            report[key] = {k: float(v) for k, v in result.results_dict.items()}
            (output / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
