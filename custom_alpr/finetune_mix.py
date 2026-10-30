"""Build fine-tuning manifests: Kaggle train plus repeated, person-labelled video crops.

Video crops are few (about 100), so each is repeated ``--repeat`` times per epoch
to matter next to ~9.5k Kaggle rows. One leave-camera-out fold per camera
measures whether fine-tuning helps on a camera the model never saw:

    <output>/train-<camera>/train.csv     Kaggle train + that camera's crops
    <output>/train-<camera>/holdout.csv   crops of every other camera
    <output>/all/train.csv                Kaggle train + all labelled crops
    <output>/all/holdout.csv              the collected test split
    <output>/base-test.csv                base test rows of --base-source (when given)

    python3 -m custom_alpr.finetune_mix
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OCR_DATA = ROOT / "custom_alpr/data/ocr"
FIELDS = ["image_path", "plate_text", "source", "type", "sha256"]


def read_manifest(path: Path) -> list[dict]:
    """Rows with ``image_path`` made absolute so they can be re-rooted anywhere."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row, image_path=os.path.normpath(path.parent / row["image_path"])) for row in csv.DictReader(handle)]


def video_camera(collected: Path) -> dict[str, str]:
    cameras = {}
    for metadata in collected.glob("*/metadata.csv"):
        with metadata.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                cameras[row["video_id"]] = row.get("camera") or row["video_id"]
    return cameras


def row_camera(row: dict, cameras: dict[str, str]) -> str:
    video_id = Path(row["image_path"]).parts[-3]  # <video_id>/crops/<crop>.png
    return cameras.get(video_id, video_id)


def mix_rows(base: list[dict], video: list[dict], repeat: int) -> list[dict]:
    return base + [row for row in video for _ in range(repeat)]


def write_manifest(folder: Path, name: str, rows: list[dict]):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, image_path=Path(os.path.relpath(row["image_path"], folder)).as_posix()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, default=OCR_DATA / "kaggle-v9-prepared/train.csv")
    parser.add_argument("--video", type=Path, default=OCR_DATA / "collected-v1")
    parser.add_argument("--collected", type=Path, default=ROOT / "custom_alpr/data/collected")
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--base-source", help="keep only base rows of this source, e.g. cropped (real photos)")
    parser.add_argument("--output", type=Path, default=OCR_DATA / "finetune-video-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = read_manifest(args.base)
    if args.base_source:
        base = [row for row in base if row["source"] == args.base_source]
        test = read_manifest(args.base.parent / "test.csv")
        write_manifest(args.output, "base-test.csv", [row for row in test if row["source"] == args.base_source])
    video = read_manifest(args.video / "train.csv")
    cameras = video_camera(args.collected)
    by_camera: dict[str, list[dict]] = {}
    for row in video:
        by_camera.setdefault(row_camera(row, cameras), []).append(row)

    folds = {"all": (video, read_manifest(args.video / "test.csv"))}
    for camera, rows in by_camera.items():
        folds[f"train-{camera}"] = (rows, [row for row in video if row not in rows])
    for name, (train, holdout) in folds.items():
        write_manifest(args.output / name, "train.csv", mix_rows(base, train, args.repeat))
        write_manifest(args.output / name, "holdout.csv", holdout)
        print(f"{name}: {len(base)} base + {len(train)}x{args.repeat} video rows; holdout {len(holdout)}")


if __name__ == "__main__":
    main()
