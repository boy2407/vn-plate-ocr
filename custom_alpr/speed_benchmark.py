"""Time the demo pipeline per stage with the .pt and OpenVINO YOLO models on one clip.

Also checks the exports change nothing: boxes are matched to the .pt run by IoU and
the OCR text of matched boxes is compared.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2

from custom_alpr.video_demo import DEFAULT_DETECTOR, DEFAULT_OCR, ROOT
from custom_alpr.web_app import DEFAULT_POSE, InferenceEngine
from custom_alpr.yolo_export import openvino_dir

DEFAULT_CLIP = ROOT / "custom_alpr/outputs/video-demo/highway-test-sample.mp4"


def read_frames(clip: Path, limit: int) -> list:
    capture = cv2.VideoCapture(str(clip))
    frames = []
    while len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise SystemExit(f"No frames read from {clip}")
    return frames


def run(engine: InferenceEngine, frames: list, confidence: float, image_size: int) -> dict:
    stages = {"detector": [], "pose": [], "ocr": []}
    per_frame = []
    for frame in frames:
        results, _ = engine.recognize(
            frame, confidence, pose_warp=True, padding=True, image_size=image_size
        )
        for name, ms in engine.stage_ms.items():
            stages[name].append(ms)
        per_frame.append([(item["box"], item["text"]) for item in results])
    return {
        "median_ms": {name: round(statistics.median(v), 1) for name, v in stages.items()},
        "mean_total_ms": round(sum(sum(v) for v in stages.values()) / len(frames), 1),
        "per_frame": per_frame,
    }


def iou(a, b) -> float:
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def compare(reference: list, candidate: list) -> dict:
    matched = same_text = 0
    for ref_frame, cand_frame in zip(reference, candidate, strict=True):
        unused = list(cand_frame)
        for box, text in ref_frame:
            best = max(unused, key=lambda item: iou(box, item[0]), default=None)
            if best is not None and iou(box, best[0]) >= 0.5:
                unused.remove(best)
                matched += 1
                same_text += best[1] == text
    return {
        "reference_boxes": sum(map(len, reference)),
        "candidate_boxes": sum(map(len, candidate)),
        "matched_boxes": matched,
        "same_text": same_text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--ocr", type=Path, default=DEFAULT_OCR.with_suffix(".onnx"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frames = read_frames(args.clip, args.frames)
    backends = {
        "pt": (DEFAULT_DETECTOR, DEFAULT_POSE),
        "openvino": (openvino_dir(DEFAULT_DETECTOR), openvino_dir(DEFAULT_POSE)),
    }
    report = {"clip": str(args.clip), "frames": len(frames)}
    runs = {}
    for name, (detector, pose) in backends.items():
        engine = InferenceEngine(detector, args.ocr, "cpu", pose=pose)
        runs[name] = run(engine, frames, args.confidence, args.image_size)
        report[name] = {key: runs[name][key] for key in ("median_ms", "mean_total_ms")}
        print(name, json.dumps(report[name]))
    report["openvino_vs_pt"] = compare(runs["pt"]["per_frame"], runs["openvino"]["per_frame"])
    print(json.dumps(report["openvino_vs_pt"]))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
