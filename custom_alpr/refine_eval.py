"""Measure the second-pass box refinement before turning it on.

Part A, labelled detector images: per ground-truth plate, compare the first-pass
box and the refined (or fallen-back) box with the truth box, and compare OCR of
both crops with OCR of the truth crop (there is no text label).

Part B, collected video crops: per camera, how often refinement falls back and
how often it changes the read; a contact sheet of changed reads is written so
a person can judge which read is right.

    python -m custom_alpr.refine_eval --images <detector images/val> --collected custom_alpr/data/collected
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from custom_alpr.refine_box import CONTEXT, SIZE

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "custom_alpr/outputs/refine-eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, default=None, help="Part A: labelled detector images")
    parser.add_argument("--collected", type=Path, default=None, help="Part B: collected crops")
    parser.add_argument("--min-height", type=int, default=16)
    parser.add_argument("--sheet-size", type=int, default=32)
    parser.add_argument("--context", type=float, default=CONTEXT, help="Video crops hold only 0.05")
    parser.add_argument("--size", type=int, default=SIZE, help="Second-pass input long side")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:  # noqa: PLR0915
    import cv2  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    from custom_alpr.box_context_sweep import iou, yolo_boxes  # noqa: PLC0415
    from custom_alpr.plate_tracks import normalized_text  # noqa: PLC0415
    from custom_alpr.preprocessing import prepare_ocr_input  # noqa: PLC0415
    from custom_alpr.refine_box import refine_box, yolo_predictor  # noqa: PLC0415
    from custom_alpr.video_demo import DEFAULT_DETECTOR, DEFAULT_OCR, ocr_arrays, select_boxes  # noqa: PLC0415
    from custom_alpr.web_app import InferenceEngine  # noqa: PLC0415

    args = parse_args()
    engine = InferenceEngine(DEFAULT_DETECTOR, DEFAULT_OCR, args.device)
    predict = yolo_predictor(engine.detector, engine.yolo_device)
    args.output.mkdir(parents=True, exist_ok=True)

    def read(images):
        texts = []
        for start in range(0, len(images), 128):
            batch = [prepare_ocr_input(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images[start : start + 128]]
            texts.extend(normalized_text(text) for text, _ in ocr_arrays(batch, engine.recognizer, engine.device))
        return texts

    def cut(image, box):
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]

    def locate(padded, tight):
        """First-pass box inside a padded crop: the tight crop is an exact sub-image of it."""
        x, y = cv2.minMaxLoc(cv2.matchTemplate(padded, tight, cv2.TM_SQDIFF))[2]
        return [x, y, x + tight.shape[1], y + tight.shape[0]]

    report = {}
    if args.images:
        labels = args.images.parent.parent / "labels" / args.images.name
        rows, crops = [], []
        for path in sorted(args.images.iterdir()):
            label = labels / f"{path.stem}.txt"
            image = cv2.imread(str(path)) if label.exists() else None
            if image is None:
                continue
            height, width = image.shape[:2]
            truth = [box for box in yolo_boxes(label, width, height) if box[3] - box[1] >= args.min_height]
            prediction = engine.detector.predict(image, conf=0.4, imgsz=640, device=engine.yolo_device, verbose=False)[0]
            first_boxes = [list(box[:4]) for box in select_boxes(prediction.boxes, width, height)]
            for gt in truth:
                first = max(first_boxes, key=lambda box: iou(gt, box), default=None)
                if first is None or iou(gt, first) < 0.5:
                    continue
                final, refined = refine_box(predict, image, first, args.context, args.size)
                rows.append(dict(refined=refined, iou_first=iou(gt, first), iou_final=iou(gt, final)))
                crops += [cut(image, first), cut(image, final), cut(image, gt)]
        texts = read(crops)
        for index, row in enumerate(rows):
            row["first"], row["final"], row["truth_crop"] = texts[3 * index : 3 * index + 3]
        refined = [row for row in rows if row["refined"]]
        report["detector_images"] = dict(
            plates=len(rows),
            refined=len(refined),
            fallback_rate=round(1 - len(refined) / len(rows), 4) if rows else None,
            iou_first_median=round(statistics.median(row["iou_first"] for row in rows), 3),
            iou_final_median=round(statistics.median(row["iou_final"] for row in rows), 3),
            iou_dropped_over_0_1=sum(row["iou_final"] < row["iou_first"] - 0.1 for row in rows),
            same_as_truth_crop_first=round(sum(row["first"] == row["truth_crop"] for row in rows) / len(rows), 4),
            same_as_truth_crop_final=round(sum(row["final"] == row["truth_crop"] for row in rows) / len(rows), 4),
            read_changed=sum(row["first"] != row["final"] for row in rows),
        )  # fmt: skip
        print("A", json.dumps(report["detector_images"]))

    if args.collected:
        metadata = [row for path in sorted(args.collected.glob("*/metadata.csv"))
                    for row in csv.DictReader(path.open(encoding="utf-8"))]  # fmt: skip
        rows, crops = [], []
        for meta in metadata:
            folder = args.collected / meta["video_id"] / "crops"
            padded = cv2.imread(str(folder / f"{meta['crop_id']}_pad.png"))
            tight = cv2.imread(str(folder / f"{meta['crop_id']}.png"))
            if padded is None or tight is None:
                continue
            first = locate(padded, tight)
            final, refined = refine_box(predict, padded, first, args.context, args.size)
            rows.append(dict(crop_id=meta["crop_id"], camera=meta["camera"], refined=refined,
                             area=(final[2] - final[0]) * (final[3] - final[1]) / ((first[2] - first[0]) * (first[3] - first[1]))))  # fmt: skip
            crops += [tight, cut(padded, final)]
        texts = read(crops)
        for index, row in enumerate(rows):
            row["first"], row["final"] = texts[2 * index : 2 * index + 2]
        cameras = {}
        for camera in sorted({row["camera"] for row in rows}):
            subset = [row for row in rows if row["camera"] == camera]
            refined = [row for row in subset if row["refined"]]
            cameras[camera] = dict(
                crops=len(subset),
                fallback_rate=round(1 - len(refined) / len(subset), 4),
                refined_area_median=round(statistics.median(row["area"] for row in refined), 3) if refined else None,
                read_changed=sum(row["first"] != row["final"] for row in subset),
            )
            print("B", camera, json.dumps(cameras[camera]))
        report["video_crops"] = cameras

        changed = [row for row in rows if row["first"] != row["final"]]
        random.Random(7).shuffle(changed)
        changed = changed[: args.sheet_size]
        cell_w, cell_h = 560, 150
        sheet = Image.new("RGB", (cell_w * 2, cell_h * ((len(changed) + 1) // 2)), "white")
        draw = ImageDraw.Draw(sheet)
        lookup = {meta["crop_id"]: meta for meta in metadata}
        for index, row in enumerate(changed):
            meta = lookup[row["crop_id"]]
            folder = args.collected / meta["video_id"] / "crops"
            padded = cv2.imread(str(folder / f"{row['crop_id']}_pad.png"))
            tight = cv2.imread(str(folder / f"{row['crop_id']}.png"))
            final, _ = refine_box(predict, padded, locate(padded, tight), args.context, args.size)
            ox, oy = (index % 2) * cell_w, (index // 2) * cell_h
            for column, crop in enumerate((tight, cut(padded, final))):
                picture = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                scale = min(260 / picture.width, 110 / picture.height)
                picture = picture.resize((max(1, int(picture.width * scale)), max(1, int(picture.height * scale))))
                sheet.paste(picture, (ox + 5 + column * 275, oy + 30))
            draw.text((ox + 5, oy + 5), f"#{index} 1st: {row['first']}   2nd: {row['final']}", fill="red")
        sheet.save(args.output / "changed-reads.png")
        (args.output / "changed-reads.json").write_text(json.dumps(changed, indent=1), encoding="utf-8")

    (args.output / "refine-eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
