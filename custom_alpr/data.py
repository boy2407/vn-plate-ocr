"""Convert YOLO polygons/boxes to detection labels and OCR transcription templates."""

import csv
import hashlib
import math
import shutil
from pathlib import Path

import cv2
import yaml

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_box(line: str) -> tuple[float, float, float, float]:
    """Return normalized corners; merge all source classes into one plate class."""
    fields = line.split()
    class_id = int(fields[0])
    values = [float(value) for value in fields[1:]]
    if class_id < 0 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Invalid label: {line}")
    if len(values) == 4:
        cx, cy, width, height = values
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid box size: {line}")
        corners = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[::2], values[1::2]
        corners = (min(xs), min(ys), max(xs), max(ys))
    else:
        raise ValueError(f"Expected a YOLO box or polygon: {line}")
    # Only tolerate floating point noise, not materially out-of-image annotations.
    if any(value < -1e-6 or value > 1 + 1e-6 for value in corners):
        raise ValueError(f"Coordinates outside image; correct source annotation: {line}")
    x1, y1, x2, y2 = (min(1.0, max(0.0, value)) for value in corners)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Empty box: {line}")
    return x1, y1, x2, y2


def prepare(source: Path, output: Path) -> dict:  # noqa: PLR0912
    """Keep existing splits, deduplicate decoded images, and never invent OCR text."""
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new directory.")
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            if not (source / kind / split).is_dir():
                raise ValueError(f"Missing directory: {source / kind / split}")
    output.mkdir(parents=True)
    seen = set()
    report: dict = {"splits": {}, "duplicates_removed": [], "rejected": []}
    for split in ("train", "val"):
        rows = []
        count = 0
        for kind in ("images", "labels"):
            (output / "detection" / kind / split).mkdir(parents=True)
        crop_dir = output / "ocr" / split / "images"
        crop_dir.mkdir(parents=True)
        images = sorted(
            path
            for path in (source / "images" / split).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        used_stems = set()
        for path in images:
            if path.stem in used_stems:
                raise ValueError(f"Ambiguous image stem: {path.stem}")
            used_stems.add(path.stem)
            try:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError("Unreadable image")
                label = source / "labels" / split / f"{path.stem}.txt"
                boxes = [parse_box(line) for line in label.read_text().splitlines() if line.strip()]
                height, width = image.shape[:2]
                pixels = [
                    (
                        int(x1 * width),
                        int(y1 * height),
                        math.ceil(x2 * width),
                        math.ceil(y2 * height),
                    )
                    for x1, y1, x2, y2 in boxes
                ]
                digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            except (ValueError, OSError, IndexError) as error:
                report["rejected"].append({"image": str(path), "reason": str(error)})
                continue
            if digest in seen:
                report["duplicates_removed"].append(str(path))
                continue
            seen.add(digest)
            shutil.copy2(path, output / "detection" / "images" / split / path.name)
            labels = []
            for index, ((x1, y1, x2, y2), (px1, py1, px2, py2)) in enumerate(
                zip(boxes, pixels, strict=True)
            ):
                labels.append(
                    f"0 {(x1 + x2) / 2:.8f} {(y1 + y2) / 2:.8f} {x2 - x1:.8f} {y2 - y1:.8f}"
                )
                name = f"{path.stem}_{index:03d}.png"
                if not cv2.imwrite(str(crop_dir / name), image[py1:py2, px1:px2]):
                    raise OSError(f"Cannot write crop: {crop_dir / name}")
                rows.append({"image_path": f"images/{name}", "plate_text": ""})
            (output / "detection" / "labels" / split / f"{path.stem}.txt").write_text(
                "\n".join(labels) + ("\n" if labels else "")
            )
            count += 1
        with (crop_dir.parent / "annotations.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["image_path", "plate_text"])
            writer.writeheader()
            writer.writerows(rows)
        report["splits"][split] = {"images": count, "crops": len(rows)}
    with (output / "detection" / "dataset.yaml").open("w") as file:
        yaml.safe_dump(
            {
                "path": str((output / "detection").resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "license_plate"},
            },
            file,
        )
    return report


def read_annotations(path: Path, max_slots: int = 10) -> list[tuple[Path, str]]:
    """Reject missing transcriptions instead of accidentally training blank targets."""
    samples = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not {"image_path", "plate_text"} <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: expected image_path,plate_text columns")
        for line, row in enumerate(reader, 2):
            text = row.get("plate_text") or ""
            image_name = row.get("image_path") or ""
            if not text or len(text) > max_slots or any(char not in ALPHABET for char in text):
                raise ValueError(f"{path}:{line}: label must be 1-{max_slots} uppercase A-Z/0-9")
            image = path.parent / image_name
            if not image_name or not image.is_file() or cv2.imread(str(image)) is None:
                raise ValueError(f"{path}:{line}: missing or unreadable image {image}")
            samples.append((image, text))
    if not samples:
        raise ValueError(f"{path}: empty annotation file")
    return samples
