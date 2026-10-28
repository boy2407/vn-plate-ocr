"""Prepare the Kaggle plate crops as deduplicated OCR split manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

ALPHABET = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZĐ")
DATASET_SLUG = "topkek69/vietnamese-license-plate-ocr"
SOURCES = (
    ("cropped", "labels/crop_labels.csv", "cropped"),
    ("generated", "labels/gen_labels.csv", "generated"),
)
MANIFEST_FIELDS = ("image_path", "plate_text", "source", "type", "sha256")


def normalize_label(label: str) -> str:
    """Remove source separators and normalize a transcription for PlateOCR2D."""
    return "".join(
        character
        for character in label.upper().strip()
        if not character.isspace() and character not in {"-", "."}
    )


def is_supported_image(path: Path) -> bool:
    """Check common Kaggle image signatures without relying on file extensions."""
    with path.open("rb") as image_file:
        signature = image_file.read(8)
    return signature.startswith(b"\xff\xd8\xff") or signature == b"\x89PNG\r\n\x1a\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(dataset_root: Path) -> tuple[list[dict[str, str]], dict]:
    """Load source labels, validate images, deduplicate, and reject label conflicts."""
    by_digest: dict[str, list[dict[str, str]]] = defaultdict(list)
    report = {
        "dataset": DATASET_SLUG,
        "source_rows": {},
        "missing_or_unsupported_images": [],
        "invalid_labels": [],
    }

    for source, label_name, image_dir in SOURCES:
        label_path = dataset_root / label_name
        with label_path.open(encoding="utf-8-sig", newline="") as label_file:
            rows = list(csv.DictReader(label_file))
        report["source_rows"][source] = len(rows)
        for row in rows:
            image_path = dataset_root / image_dir / row["Name"]
            if not image_path.is_file() or not is_supported_image(image_path):
                report["missing_or_unsupported_images"].append(
                    str(image_path.relative_to(dataset_root))
                )
                continue
            text = normalize_label(row["Label"])
            if not text or not set(text) <= ALPHABET:
                report["invalid_labels"].append(
                    {"image": str(image_path.relative_to(dataset_root)), "label": row["Label"]}
                )
                continue
            digest = sha256_file(image_path)
            by_digest[digest].append(
                {
                    "image_path": image_path,
                    "plate_text": text,
                    "source": source,
                    "type": row["Type"],
                    "sha256": digest,
                }
            )

    records = []
    conflicts = []
    duplicate_rows = 0
    for digest, group in by_digest.items():
        labels = {record["plate_text"] for record in group}
        if len(labels) != 1:
            conflicts.append(
                {
                    "sha256": digest,
                    "images": [
                        str(record["image_path"].relative_to(dataset_root)) for record in group
                    ],
                    "labels": sorted(labels),
                }
            )
            continue
        # Stable source priority keeps the original crop when it duplicates a generated sample.
        group.sort(key=lambda record: (record["source"] != "cropped", str(record["image_path"])))
        records.append(group[0])
        duplicate_rows += len(group) - 1

    report["duplicate_rows_removed"] = duplicate_rows
    report["conflicting_duplicate_groups_excluded"] = conflicts
    report["valid_unique_records"] = len(records)
    return records, report


def split_records(records: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    """Make deterministic 80/10/10 splits, stratified by source and Type."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        groups[(record["source"], record["type"])].append(record)

    rng = random.Random(seed)
    splits: dict[str, list[dict[str, str]]] = {name: [] for name in ("train", "val", "test")}
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda record: record["sha256"])
        rng.shuffle(group)
        train_end = int(len(group) * 0.8)
        val_end = train_end + int(len(group) * 0.1)
        splits["train"].extend(group[:train_end])
        splits["val"].extend(group[train_end:val_end])
        splits["test"].extend(group[val_end:])
    for split in splits.values():
        split.sort(key=lambda record: str(record["image_path"]))
    return splits


def write_manifests(
    dataset_root: Path, output_dir: Path, splits: dict[str, list[dict[str, str]]], seed: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    for split_name, rows in splits.items():
        manifest = output_dir / f"{split_name}.csv"
        with manifest.open("w", encoding="utf-8", newline="") as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for record in rows:
                writer.writerow(
                    {
                        **record,
                        "image_path": os.path.relpath(record["image_path"], output_dir),
                    }
                )

    counts = {name: len(rows) for name, rows in splits.items()}
    metadata = {
        "dataset": DATASET_SLUG,
        "license": "Apache-2.0",
        "seed": seed,
        "split_ratio": {"train": 0.8, "val": 0.1, "test": 0.1},
        "stratification": "source x Kaggle Type; Type is not a plate-layout label",
        "image_root": os.path.relpath(dataset_root, output_dir),
        "counts": counts,
        "total": sum(counts.values()),
        "label_normalization": "uppercase; remove spaces, hyphens, and periods; preserve Đ",
        "image_handling": (
            "Original images referenced; formats detected by file signature, not extension"
        ),
        "resize_for_model": (
            "RGB, 128x64 (width x height); resize policy must be applied by the training Dataset"
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("custom_alpr/data/raw/kaggle-vietnamese-license-plate-ocr"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("custom_alpr/data/ocr/kaggle-v9-prepared"),
    )
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_root.is_dir():
        parser.error(f"Dataset directory does not exist: {dataset_root}")
    if output_dir.exists():
        parser.error(f"Output already exists; choose another path: {output_dir}")

    records, report = load_records(dataset_root)
    if report["missing_or_unsupported_images"] or report["invalid_labels"]:
        parser.error(
            "Dataset contains missing/unsupported images or invalid labels; inspect report"
        )
    splits = split_records(records, args.seed)
    write_manifests(dataset_root, output_dir, splits, args.seed)
    report["split_counts"] = {name: len(rows) for name, rows in splits.items()}
    report["seed"] = args.seed
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preparation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
