"""Compare detector input sizes on the same frames of a YouTube video.

YouTube has no ground truth, so this reports relative counts: boxes found,
how many are small (distant), OCR reads in valid plate format and distinct
valid plates. Every size sees exactly the same frames.

    python -m custom_alpr.compare_detector_sizes https://www.youtube.com/watch?v=...
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from custom_alpr.plate_tracks import normalized_text, valid_plate
from custom_alpr.video_demo import DEFAULT_DETECTOR, DEFAULT_OCR, ROOT
from custom_alpr.web_app import DEFAULT_MIN_HEIGHT, InferenceEngine, resolve_stream

OUTPUT = ROOT / "custom_alpr/outputs/detector-size-comparison"
SMALL_HEIGHT = 30  # Source pixels; below this a plate is "distant".


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--sizes", type=int, nargs="+", default=[320, 640])
    parser.add_argument("--seconds", type=float, default=120, help="Footage to read")
    parser.add_argument("--every", type=int, default=5, help="Use every Nth frame")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--min-height", type=int, default=DEFAULT_MIN_HEIGHT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--ocr-checkpoint", type=Path, default=DEFAULT_OCR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--crops", action="store_true", help="Save the first crop of each text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stream_url, title = resolve_stream(args.url)
    capture = cv2.VideoCapture(stream_url)
    if not capture.isOpened():
        raise SystemExit("Could not open the YouTube stream.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    engine = InferenceEngine(args.detector, args.ocr_checkpoint, args.device)
    for size in args.sizes:
        engine.warm_detector(size)
    stats = {
        size: dict(boxes=0, small_boxes=0, too_small=0, ocr_reads=0, valid_reads=0, seconds=0.0)
        for size in args.sizes
    }
    plates = {size: Counter() for size in args.sizes}
    heights = {size: [] for size in args.sizes}
    video_id = args.url.rsplit("=", 1)[-1].rsplit("/", 1)[-1]
    crops = args.output / f"{video_id}-crops"
    index = frames = 0
    width = height = 0
    while index < args.seconds * fps:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        if (index - 1) % args.every:
            continue
        frames += 1
        height, width = frame.shape[:2]
        for size in args.sizes:
            started = time.perf_counter()
            results, too_small = engine.recognize(
                frame, args.confidence, padding=True, image_size=size, min_height=args.min_height
            )
            row = stats[size]
            row["seconds"] += time.perf_counter() - started
            row["too_small"] += too_small
            row["boxes"] += len(results) + too_small
            row["ocr_reads"] += len(results)
            for item in results:
                box_height = item["box"][3] - item["box"][1]
                heights[size].append(box_height)
                row["small_boxes"] += box_height < SMALL_HEIGHT
                text = normalized_text(item["text"])
                if args.crops and text not in plates[size]:
                    folder = crops / str(size)
                    folder.mkdir(parents=True, exist_ok=True)
                    x1, y1, x2, y2 = item["box"]
                    name = f"{text or 'empty'}-f{index - 1}-x{x1}-y{y1}.png"
                    cv2.imwrite(str(folder / name), frame[y1:y2, x1:x2])
                if valid_plate(text):
                    row["valid_reads"] += 1
                    plates[size][text] += 1
            # Boxes dropped by min_height are small by definition.
            row["small_boxes"] += too_small
    capture.release()
    if not frames:
        raise SystemExit("No frames were read from the stream.")

    report = dict(
        url=args.url,
        title=title,
        source_size=[width, height],
        source_fps=fps,
        frames_compared=frames,
        every=args.every,
        confidence=args.confidence,
        min_height=args.min_height,
        small_height=SMALL_HEIGHT,
        detector=str(args.detector),
        results={},
    )
    for size in args.sizes:
        row = stats[size]
        # A plate read in two or more frames is unlikely to be an OCR accident.
        repeated = {text: count for text, count in plates[size].items() if count >= 2}
        report["results"][str(size)] = dict(
            row,
            seconds=round(row["seconds"], 2),
            frames_per_second=round(frames / max(row["seconds"], 1e-9), 1),
            distinct_valid_plates=len(plates[size]),
            plates_seen_twice=len(repeated),
            median_box_height=sorted(heights[size])[len(heights[size]) // 2]
            if heights[size]
            else None,
            plates=dict(plates[size].most_common()),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{video_id}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{title} · {width}×{height} · {frames} frames compared")
    print(f"{'imgsz':>6} {'boxes':>6} {'small':>6} {'valid':>6} {'plates':>7} {'≥2x':>5} {'fps':>6}")
    for size in args.sizes:
        r = report["results"][str(size)]
        print(
            f"{size:>6} {r['boxes']:>6} {r['small_boxes']:>6} {r['valid_reads']:>6} "
            f"{r['distinct_valid_plates']:>7} {r['plates_seen_twice']:>5} {r['frames_per_second']:>6}"
        )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
