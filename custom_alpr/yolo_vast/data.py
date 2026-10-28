"""Dataset audit, portable paths and exact transfer integrity."""

from __future__ import annotations

import hashlib
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(entry):
    root, image, split = entry
    label = root / "labels" / split / (image.stem + ".txt")
    if not label.is_file():
        raise ValueError(f"Missing label: {label}")
    count = 0
    for line in label.read_text().splitlines():
        if not line.strip():
            continue
        row = np.array([float(value) for value in line.split()])
        if row.shape != (5,) or not np.isfinite(row).all() or row[0] != 0:
            raise ValueError(f"Invalid class/box: {label}: {line}")
        _, x, y, w, h = row
        if (
            min(w, h) <= 0
            or min(x - w / 2, y - h / 2) < -1e-5
            or max(x + w / 2, y + h / 2) > 1 + 1e-5
        ):
            raise ValueError(f"Out-of-bounds box: {label}: {line}")
        count += 1
    image_bytes = image.read_bytes()
    with Image.open(io.BytesIO(image_bytes)) as opened:
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
        pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        width, height = rgb.size
    return dict(
        split=split,
        image=str(image.relative_to(root)),
        label=str(label.relative_to(root)),
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        label_sha256=sha256(label),
        pixel_sha256=pixel_hash,
        boxes=count,
        width=width,
        height=height,
    )


def audit_dataset(root: Path, workers=8):
    root = root.resolve()
    entries = []
    for split in ("train", "val", "test"):
        images = sorted(
            p for p in (root / "images" / split).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"Empty split: {split}")
        expected = {p.stem for p in images}
        actual = {p.stem for p in (root / "labels" / split).glob("*.txt")}
        if expected != actual or len(expected) != len(images):
            raise ValueError(f"Unmatched/ambiguous image-label files: {split}")
        entries.extend((root, image, split) for image in images)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(inspect_image, entries))
    seen = {}
    for row in records:
        digest = row["pixel_sha256"]
        if digest in seen:
            raise ValueError(f"Duplicate image/leakage: {seen[digest]} and {row['image']}")
        seen[digest] = row["image"]
    manifest = root / "manifest.json"
    if manifest.exists():
        source = json.loads(manifest.read_text())
        expected = {(r["split"], r["filename"]): r["pixel_sha256"] for r in source}
        for row in records:
            if expected.get((row["split"], Path(row["image"]).name)) != row["pixel_sha256"]:
                raise ValueError(f"Source manifest hash mismatch: {row['image']}")
        if len(source) != len(records):
            raise ValueError("Source manifest/image count mismatch")
    counts = {}
    for split in ("train", "val", "test"):
        subset = [r for r in records if r["split"] == split]
        counts[split] = dict(
            images=len(subset),
            boxes=sum(r["boxes"] for r in subset),
            negatives=sum(r["boxes"] == 0 for r in subset),
        )
    return dict(version=1, counts=counts, records=records)


def verify_inventory(root: Path):
    inventory = json.loads((root / "inventory.json").read_text())
    failures = []
    for row in inventory["records"]:
        for kind in ("image", "label"):
            path = root / "data" / row[kind]
            if not path.is_file() or sha256(path) != row[kind + "_sha256"]:
                failures.append(str(path))
    if sha256(root / "weights/initial.pt") != inventory["initial_sha256"]:
        failures.append("weights/initial.pt")
    if failures:
        raise ValueError(f"Bundle integrity failed ({len(failures)}): {failures[:8]}")
    return inventory


def training_policy(inventory, include_empty_labels=False):
    excluded = (
        []
        if include_empty_labels
        else [r["image"] for r in inventory["records"] if r["split"] == "train" and r["boxes"] == 0]
    )
    inventory["train_exclude"] = excluded
    inventory["exclusion_reason"] = (
        "Unreviewed empty-label traffic frames are retained for review, not assumed plate-free."
    )
    excluded = set(excluded)
    counts = {}
    for split in ("train", "val", "test"):
        active = [
            r for r in inventory["records"] if r["split"] == split and r["image"] not in excluded
        ]
        counts[split] = dict(
            images=len(active),
            boxes=sum(r["boxes"] for r in active),
            negatives=sum(r["boxes"] == 0 for r in active),
        )
    inventory["effective_counts"] = counts
    return inventory


def runtime_yaml(root: Path, smoke=False):
    root = root.resolve()
    config = {
        "path": str(root / "data"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "license_plate"},
    }
    inventory = json.loads((root / "inventory.json").read_text())
    excluded = set(inventory.get("train_exclude", []))
    if excluded:
        selected = [
            root / "data" / r["image"]
            for r in inventory["records"]
            if r["split"] == "train" and r["image"] not in excluded
        ]
        train_list = root / "train.runtime.txt"
        train_list.write_text("\n".join(map(str, selected)) + "\n")
        config["train"] = str(train_list)
    if smoke:
        for split, count in (("train", 32), ("val", 16)):
            images = sorted(
                p.resolve()
                for p in (root / "data/images" / split).iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
                and str(p.relative_to(root / "data")) not in excluded
            )
            chosen = random.Random(42).sample(images, min(count, len(images)))
            text = root / f"smoke-{split}.txt"
            text.write_text("\n".join(map(str, chosen)) + "\n")
            config[split] = str(text)
        config.pop("test")
    path = root / ("smoke.runtime.yaml" if smoke else "dataset.runtime.yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def code_fingerprint(root: Path):
    digest = hashlib.sha256()
    for path in sorted((root / "custom_alpr/yolo_vast").rglob("*")):
        if path.suffix in {".py", ".yaml", ".json", ".sh"}:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()
