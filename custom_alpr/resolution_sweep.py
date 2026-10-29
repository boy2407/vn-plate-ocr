"""Measure how OCR accuracy falls as a large real plate loses source pixels.

Large real test crops are shrunk to a fixed height (aspect kept), JPEG-compressed
like video, then read through the normal 128x64 path. If accuracy at small
heights matches the measured small-crop accuracy, the bottleneck is the source
resolution, not the fixed model input.

    python -m custom_alpr.resolution_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "custom_alpr/data/ocr/kaggle-v9-prepared/test.csv"
OUTPUT = ROOT / "custom_alpr/outputs/resolution-sweep"
TWO_LINE_ASPECT = 2.5  # Width/height below this is a two-line plate (task 028 rule).


def degrade(image: np.ndarray, height: int, quality: int = 75) -> np.ndarray:
    """Shrink an RGB crop to ``height`` pixels, keeping aspect, then JPEG round-trip."""
    scale = height / image.shape[0]
    width = max(1, round(image.shape[1] * scale))
    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise ValueError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def load_samples(manifest: Path, min_height: int) -> list[dict]:
    """Real (``cropped``) test plates at least ``min_height`` pixels tall."""
    samples = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source"] != "cropped":
                continue
            path = (manifest.parent / row["image_path"]).resolve()
            with Image.open(path) as image_file:
                image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
            if image.shape[0] < min_height:
                continue
            two_line = image.shape[1] / image.shape[0] < TWO_LINE_ASPECT
            samples.append(dict(text=row["plate_text"], image=image, two_line=two_line))
    return samples


def accuracy(rows: list[tuple[bool, bool]]) -> dict:
    """Exact match overall and per layout from (correct, two_line) pairs."""
    report = {}
    for name, keep in (("all", None), ("one_line", False), ("two_line", True)):
        hits = [correct for correct, two_line in rows if keep is None or two_line == keep]
        report[name] = dict(n=len(hits), exact=sum(hits) / len(hits) if hits else None)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--min-height", type=int, default=40, help="Source crops kept")
    parser.add_argument("--heights", type=int, nargs="+", default=[12, 16, 20, 24, 32, 40])
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ocr-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    # Torch is imported here so the pure helpers above load without it.
    from custom_alpr.preprocessing import prepare_ocr_input  # noqa: PLC0415
    from custom_alpr.video_demo import (  # noqa: PLC0415
        DEFAULT_OCR,
        load_ocr,
        ocr_arrays,
        resolve_device,
    )

    args = parse_args()
    device, _ = resolve_device(args.device)
    model = load_ocr(args.ocr_checkpoint or DEFAULT_OCR, device)
    samples = load_samples(args.manifest, args.min_height)
    if not samples:
        raise SystemExit("No real crops at or above --min-height.")

    def read(images: list[np.ndarray]) -> list[str]:
        texts = []
        for start in range(0, len(images), args.batch_size):
            batch = [prepare_ocr_input(image) for image in images[start : start + args.batch_size]]
            texts.extend(text for text, _ in ocr_arrays(batch, model, device))
        return texts

    results = {}
    # "source" is the undegraded crop: the ceiling every height is compared with.
    for height in ["source", *args.heights]:
        images = [
            sample["image"] if height == "source" else degrade(sample["image"], height, args.jpeg_quality)
            for sample in samples
        ]
        texts = read(images)
        rows = [
            (text == sample["text"], sample["two_line"])
            for text, sample in zip(texts, samples, strict=True)
        ]
        results[str(height)] = accuracy(rows)
        line = results[str(height)]
        print(
            f"h={height!s:>6}  all {line['all']['exact']:.3f}  "
            f"1-line {line['one_line']['exact'] or 0:.3f} (n={line['one_line']['n']})  "
            f"2-line {line['two_line']['exact'] or 0:.3f} (n={line['two_line']['n']})"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(
        manifest=str(args.manifest),
        checkpoint=str(args.ocr_checkpoint or DEFAULT_OCR),
        min_source_height=args.min_height,
        jpeg_quality=args.jpeg_quality,
        samples=len(samples),
        results=results,
    )
    path = args.output / "resolution-sweep.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
