"""Measure how OCR reacts when the plate crop includes surrounding vehicle body.

Ground-truth plate boxes from the detector's labelled images are cropped at
growing scales around their centre (real context, not padding). There is no
text label, so each read is compared with the tight (scale 1.0) read: if loose
crops change the answer often, a loose detector box is costing accuracy. The
detector's own boxes are matched to ground truth to show how loose they are.

    python -m custom_alpr.box_context_sweep --images custom_alpr/data/detection/images/val
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "custom_alpr/outputs/box-context-sweep"
TWO_LINE_ASPECT = 2.0  # Same rule as collect_crops.


def scaled_box(box, scale: float, width: int, height: int) -> list[int]:
    """Grow (x1, y1, x2, y2) about its centre by ``scale``, clipped to the image."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    return [
        max(0, round(cx - half_w)),
        max(0, round(cy - half_h)),
        min(width, round(cx + half_w)),
        min(height, round(cy + half_h)),
    ]


def yolo_boxes(label: Path, width: int, height: int) -> list[list[int]]:
    """Pixel boxes from a YOLO label file (class cx cy w h, normalised)."""
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cx, cy, w, h = (float(value) for value in parts[1:])
        boxes.append(
            [round((cx - w / 2) * width), round((cy - h / 2) * height),
             round((cx + w / 2) * width), round((cy + h / 2) * height)]
        )  # fmt: skip
    return boxes


def iou(a, b) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None, help="Default: images/../../labels/<split>")
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.15, 1.3, 1.6, 2.0])
    parser.add_argument("--min-height", type=int, default=25, help="Skip tiny plates")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:  # noqa: PLR0915
    # Torch and Ultralytics are imported here so the pure helpers load without them.
    import cv2  # noqa: PLC0415

    from custom_alpr.plate_tracks import normalized_text, valid_plate  # noqa: PLC0415
    from custom_alpr.preprocessing import prepare_ocr_input  # noqa: PLC0415
    from custom_alpr.video_demo import (  # noqa: PLC0415
        DEFAULT_DETECTOR,
        DEFAULT_OCR,
        ocr_arrays,
        select_boxes,
    )
    from custom_alpr.web_app import InferenceEngine  # noqa: PLC0415

    args = parse_args()
    labels = args.labels or args.images.parent.parent / "labels" / args.images.name
    engine = InferenceEngine(DEFAULT_DETECTOR, DEFAULT_OCR, args.device)

    plates, inputs = [], []
    ratios = []
    for image_path in sorted(args.images.iterdir()):
        label = labels / f"{image_path.stem}.txt"
        if not label.exists():
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        truth = [box for box in yolo_boxes(label, width, height) if box[3] - box[1] >= args.min_height]
        if not truth:
            continue
        prediction = engine.detector.predict(frame, conf=0.4, imgsz=640, device=engine.yolo_device, verbose=False)[0]
        predicted = [box[:4] for box in select_boxes(prediction.boxes, width, height)]
        for box in truth:
            match = max(predicted, key=lambda other: iou(box, other), default=None)
            if match is not None and iou(box, match) >= 0.5:
                ratios.append(
                    ((match[2] - match[0]) / (box[2] - box[0]), (match[3] - match[1]) / (box[3] - box[1]))
                )
            two_line = (box[2] - box[0]) / (box[3] - box[1]) < TWO_LINE_ASPECT
            plates.append(dict(image=image_path.name, two_line=two_line))
            for scale in args.scales:
                x1, y1, x2, y2 = scaled_box(box, scale, width, height)
                crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                inputs.append(prepare_ocr_input(crop))

    reads = []
    for start in range(0, len(inputs), 128):
        reads.extend(ocr_arrays(inputs[start : start + 128], engine.recognizer, engine.device))
    per_scale = len(args.scales)
    for index, plate in enumerate(plates):
        plate["reads"] = [normalized_text(text) for text, _ in reads[index * per_scale : (index + 1) * per_scale]]
        plate["conf"] = [float(conf) for _, conf in reads[index * per_scale : (index + 1) * per_scale]]

    results = {}
    for name, keep in (("all", None), ("one_line", False), ("two_line", True)):
        subset = [plate for plate in plates if keep is None or plate["two_line"] == keep]
        # Plates whose tight read is a well-formed plate: the reference we trust most.
        anchored = [plate for plate in subset if valid_plate(plate["reads"][0])]
        rows = {}
        for position, scale in enumerate(args.scales):
            rows[str(scale)] = dict(
                same_as_tight=round(sum(p["reads"][position] == p["reads"][0] for p in anchored) / len(anchored), 4) if anchored else None,
                valid_format=round(sum(valid_plate(p["reads"][position]) for p in subset) / len(subset), 4) if subset else None,
                mean_conf=round(statistics.mean(p["conf"][position] for p in subset), 4) if subset else None,
            )  # fmt: skip
        results[name] = dict(plates=len(subset), tight_valid=len(anchored), scales=rows)
        print(name, json.dumps(results[name]))

    looseness = None
    if ratios:
        looseness = dict(
            matched=len(ratios),
            width_ratio_median=round(statistics.median(r[0] for r in ratios), 3),
            height_ratio_median=round(statistics.median(r[1] for r in ratios), 3),
            width_ratio_p90=round(sorted(r[0] for r in ratios)[int(len(ratios) * 0.9)], 3),
        )
        print("detector/gt", json.dumps(looseness))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "box-context-sweep.json"
    path.write_text(json.dumps(dict(images=str(args.images), results=results, detector_vs_truth=looseness), indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
