"""Validate labelled crops from ``queue.csv`` and export OCR training manifests.

Rows with an invalid label, and every label of a track whose crops disagree,
go to ``review.csv`` and are not imported; a fixed-seed sample of good rows is
added there for spot checks. Labels are compared with the OCR output kept in
each video's ``metadata.csv`` to report the model's real error on video.
Splits are assigned per camera (``--camera`` at collection, else the video),
because recordings of one camera share parked cars: one vehicle never sits in
train and test.

    python -m custom_alpr.ingest_labels --test-cameras hanoi-drive
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from custom_alpr.plate_tracks import normalized_text, valid_plate

ROOT = Path(__file__).resolve().parents[1]
COLLECTED = ROOT / "custom_alpr/data/collected"
OUTPUT = ROOT / "custom_alpr/data/ocr/collected-v1"
VERDICTS = {"label", "unreadable", "not_plate", "partial"}
MANIFEST_FIELDS = ["image_path", "plate_text", "source", "type", "sha256"]
REVIEW_FIELDS = [
    "reason", "crop_id", "image_path", "video_id", "track_id",
    "verdict", "plate_text", "lines", "note", "ocr_text",
]  # fmt: skip


def validate_row(row: dict) -> list[str]:
    """Problems with one labelled row; an empty list means it can be imported."""
    verdict = row.get("verdict", "").strip()
    text = row.get("plate_text", "").strip()
    lines = row.get("lines", "").strip()
    if verdict not in VERDICTS:
        return [f"unknown verdict {verdict!r}"]
    errors = []
    if verdict == "label":
        if not text:
            errors.append("label without plate_text")
        elif not valid_plate(normalized_text(text)):
            errors.append(f"plate_text {text!r} does not match PLATE_FORMAT")
        if lines not in {"1", "2"}:
            errors.append(f"lines must be 1 or 2, got {lines!r}")
    elif text:
        errors.append(f"plate_text given with verdict {verdict}")
    if lines and lines not in {"1", "2"}:
        errors.append(f"lines must be 1 or 2, got {lines!r}")
    return sorted(set(errors))


def inconsistent_tracks(rows: list[dict]) -> set[tuple[str, str]]:
    """(video_id, track_id) whose ``label`` rows carry more than one text."""
    texts = defaultdict(set)
    for row in rows:
        if row["verdict"].strip() == "label":
            texts[(row["video_id"], row["track_id"])].add(normalized_text(row["plate_text"]))
    return {key for key, values in texts.items() if len(values) > 1}


def split_by_camera(
    cameras: list[str], test_cameras: list[str], val_fraction: float, seed: int
) -> dict[str, str]:
    """Whole cameras go to one split; held-out test cameras are chosen by name."""
    remaining = sorted(set(cameras) - set(test_cameras))
    random.Random(seed).shuffle(remaining)
    # Keep at least one training camera; val only when there are cameras to spare.
    count = min(len(remaining) - 1, round(len(remaining) * val_fraction)) if remaining else 0
    splits = {camera: "test" for camera in set(cameras) & set(test_cameras)}
    splits.update({camera: "val" for camera in remaining[: max(0, count)]})
    splits.update({camera: "train" for camera in remaining[max(0, count) :]})
    return splits


def review_sample(rows: list[dict], fraction: float, seed: int) -> list[dict]:
    """Fixed-seed spot-check sample; at least one row when there are any."""
    if not rows or fraction <= 0:
        return []
    count = max(1, round(len(rows) * fraction))
    ordered = sorted(rows, key=lambda row: row["crop_id"])
    return random.Random(seed).sample(ordered, min(count, len(ordered)))


def confusions(pairs: list[tuple[str, str]], top: int = 10) -> list[list]:
    """Most frequent (label -> OCR) character swaps, equal-length reads only."""
    counts = Counter(
        (truth_char, read_char)
        for truth, read in pairs
        if len(truth) == len(read)
        for truth_char, read_char in zip(truth, read, strict=True)
        if truth_char != read_char
    )
    return [[f"{truth}->{read}", count] for (truth, read), count in counts.most_common(top)]


def ocr_report(rows: list[dict]) -> dict:
    """Model error on the labelled crops and tracks, overall and per layout."""
    report = {}
    for name, keep in (("all", None), ("one_line", "1"), ("two_line", "2")):
        subset = [row for row in rows if keep is None or row["lines"].strip() == keep]
        crop_hits = [row["ocr_text"] == row["label"] for row in subset]
        tracks = {(row["video_id"], row["track_id"]): row for row in subset}
        track_hits = [row["track_consensus"] == row["label"] for row in tracks.values()]
        report[name] = dict(
            crops=len(crop_hits),
            crop_exact=round(sum(crop_hits) / len(crop_hits), 4) if crop_hits else None,
            tracks=len(track_hits),
            track_exact=round(sum(track_hits) / len(track_hits), 4) if track_hits else None,
        )
    report["confusions"] = confusions([(row["label"], row["ocr_text"]) for row in rows])
    return report


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fields: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collected", type=Path, default=COLLECTED, help="Crops and metadata")
    parser.add_argument("--queue", type=Path, default=None, help="Default: <collected>/queue.csv")
    parser.add_argument("--report-dir", type=Path, default=None, help="Default: --collected")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Manifest folder")
    parser.add_argument("--test-cameras", nargs="*", default=[])
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--review-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261029)
    return parser.parse_args()


def main() -> None:  # noqa: PLR0915
    args = parse_args()
    queue = args.queue or args.collected / "queue.csv"
    report_dir = args.report_dir or args.collected
    labelled = [row for row in read_rows(queue) if row["verdict"].strip()]
    if not labelled:
        raise SystemExit(f"No labelled rows in {queue}.")
    metadata = {}
    for video in sorted({row["video_id"] for row in labelled}):
        for row in read_rows(args.collected / video / "metadata.csv"):
            metadata[row["crop_id"]] = row

    review, accepted, negatives = [], [], []
    for row in labelled:
        meta = metadata.get(row["crop_id"])
        row["ocr_text"] = meta["ocr_text"] if meta else ""
        row["camera"] = (meta or {}).get("camera") or row["video_id"]
        errors = validate_row(row) or ([] if meta else ["crop_id missing from metadata.csv"])
        if errors:
            review.append(dict(row, reason="invalid: " + "; ".join(errors)))
        else:
            accepted.append(row)
    conflicts = inconsistent_tracks(accepted)

    def conflicted(row):
        return row["verdict"].strip() == "label" and (row["video_id"], row["track_id"]) in conflicts

    review.extend(dict(row, reason="track_conflict") for row in accepted if conflicted(row))
    accepted = [row for row in accepted if not conflicted(row)]
    sample = review_sample(accepted, args.review_fraction, args.seed)
    review.extend(dict(row, reason="spot_check") for row in sample)

    labels = []
    for row in accepted:
        verdict = row["verdict"].strip()
        meta = metadata[row["crop_id"]]
        if verdict == "not_plate":
            negatives.append(
                dict(crop_id=row["crop_id"], image_path=row["image_path"], video_id=row["video_id"],
                     frame=meta["frame"], box=meta["box"])
            )  # fmt: skip
        elif verdict == "label":
            labels.append(
                dict(row, label=normalized_text(row["plate_text"]), track_consensus=meta["track_consensus"])
            )

    splits = split_by_camera(
        [row["camera"] for row in labels], args.test_cameras, args.val_fraction, args.seed
    )
    if len(splits) == 1 and not args.test_cameras:
        print("Warning: one camera only; every label goes to train, nothing is held out.")
    manifests = defaultdict(list)
    for row in labels:
        image = args.collected / row["image_path"]
        # Padded and refined (second-pass) crops are other views of the same
        # plate: same label, same split. A refined crop exists only when the
        # second pass did not fall back.
        views = (image, *(image.with_name(f"{image.stem}{tag}{image.suffix}") for tag in ("_pad", "_ref")))
        for path in views:
            if not path.exists():
                continue
            manifests[splits[row["camera"]]].append(
                dict(
                    image_path=Path(os.path.relpath(path.resolve(), args.output.resolve())).as_posix(),
                    plate_text=row["label"],
                    source="youtube",
                    type=int(row["lines"]) - 1,
                    sha256=file_sha256(path),
                )
            )
    for split in ("train", "val", "test"):
        write_rows(args.output / f"{split}.csv", MANIFEST_FIELDS, manifests[split])
    write_rows(report_dir / "review.csv", REVIEW_FIELDS, review)
    write_rows(report_dir / "negatives.csv", ["crop_id", "image_path", "video_id", "frame", "box"], negatives)

    report = dict(
        queue=str(queue),
        labelled_rows=len(labelled),
        verdicts=dict(Counter(row["verdict"].strip() for row in labelled)),
        invalid_rows=sum(row["reason"].startswith("invalid") for row in review),
        conflicting_tracks=sorted("/".join(key) for key in conflicts),
        spot_check_rows=sum(row["reason"] == "spot_check" for row in review),
        imported_labels=len(labels),
        negatives=len(negatives),
        splits=dict(sorted(splits.items())),
        manifest_rows={split: len(manifests[split]) for split in ("train", "val", "test")},
        ocr=ocr_report(labels),
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    (report_dir / "ingest-report.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
