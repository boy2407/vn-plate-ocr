"""Run the trained YOLO detector and OCR model E over a video clip."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from custom_alpr.custom_ocr import build_model
from custom_alpr.preprocessing import prepare_ocr_input

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OCR = ROOT / "custom_alpr/outputs/train-custom-ocr/run-20261028-08-parseq-e-all/best.pt"
DEFAULT_DETECTOR = (
    ROOT / "custom_alpr/outputs/vast-yolo-20261028/remote-results/moderate-s42/weights/best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Input video clip")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--ocr-checkpoint", type=Path, default=DEFAULT_OCR)
    parser.add_argument("--output", type=Path, required=True, help="Annotated MP4 output")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda:0, or 0")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence")
    parser.add_argument("--ocr-batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, choices=(320, 640, 960, 1280), default=640)
    parser.add_argument("--denoise", action="store_true", help="Light OCR crop denoising")
    parser.add_argument("--sharpen", action="store_true", help="Light OCR crop sharpening")
    args = parser.parse_args()
    for path in (args.video, args.detector, args.ocr_checkpoint):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.ocr_batch_size < 1:
        parser.error("--ocr-batch-size must be positive")
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must have an .mp4 extension")
    return args


def resolve_device(requested: str) -> tuple[torch.device, int | str]:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0"), 0
        if torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"
    if requested == "0":
        requested = "cuda:0"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    yolo_device: int | str = device.index or 0 if device.type == "cuda" else str(device)
    return device, yolo_device


def load_ocr(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def font_for_video(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:  # noqa: PERF203 -- each candidate font may be unavailable
            continue
    return ImageFont.load_default(size=size)


def ocr_crops(
    crops: list[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    *,
    denoise: bool = False,
    sharpen: bool = False,
    rectify: bool = False,
) -> list[tuple[str, float]]:
    tensors = []
    for crop in crops:
        array = prepare_ocr_input(crop, denoise=denoise, sharpen=sharpen, rectify=rectify)
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0))
    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        texts, confidences = model.generate_with_confidence(batch)
    return list(zip(texts, confidences, strict=True))


def select_best_box(boxes, width: int, height: int) -> tuple | None:
    """Choose the highest-confidence valid crop from thresholded YOLO detections."""
    best = None
    if boxes is None:
        return None
    for box in boxes:
        x1, y1, x2, y2 = map(int, np.rint(box.xyxy[0].cpu().tolist()))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        confidence = float(box.conf.item())
        if best is None or confidence > best[4]:
            best = (x1, y1, x2, y2, confidence)
    return best


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[dict],
    font: ImageFont.ImageFont,
) -> np.ndarray:
    canvas = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    for result in detections:
        x1, y1, x2, y2 = result["box"]
        draw.rectangle((x1, y1, x2, y2), outline=(0, 225, 0), width=3)
        label = (
            f"{result['text']}  det {result['detection_confidence']:.2f}  "
            f"ocr {result['ocr_confidence']:.2f}"
        )
        text_box = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = text_box[2] - text_box[0], text_box[3] - text_box[1]
        label_x = min(max(x1, 0), max(canvas.width - text_w - 8, 0))
        label_y = max(y1 - text_h - 8, 0)
        draw.rectangle(
            (label_x, label_y, label_x + text_w + 8, label_y + text_h + 6), fill=(0, 180, 0)
        )
        draw.text((label_x + 4, label_y + 2), label, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def run_video(args: argparse.Namespace) -> dict:  # noqa: PLR0915
    from ultralytics import YOLO  # noqa: PLC0415 -- optional runtime dependency

    device, yolo_device = resolve_device(args.device)
    detector = YOLO(str(args.detector), task="detect")
    recognizer = load_ocr(args.ocr_checkpoint, device)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 1 or height < 1:
        capture.release()
        raise ValueError("Video has invalid dimensions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_video = args.output.with_name(args.output.stem + ".raw.mp4")
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create video output: {raw_video}")
    font = font_for_video(max(16, round(min(width, height) / 35)))
    csv_path = args.output.with_suffix(".csv")
    frame_index = detection_count = 0
    started = time.perf_counter()

    try:
        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            rows = csv.writer(csv_file)
            rows.writerow(
                (
                    "frame",
                    "time_seconds",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "detection_confidence",
                    "ocr_text",
                    "ocr_confidence",
                )
            )
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                prediction = detector.predict(
                    source=frame,
                    conf=args.confidence,
                    device=yolo_device,
                    verbose=False,
                    imgsz=args.imgsz,
                )[0]
                crops, metadata = [], []
                best = select_best_box(prediction.boxes, width, height)
                if best is not None:
                    x1, y1, x2, y2, _ = best
                    crops.append(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                    metadata.append(best)

                frame_results = []
                for start in range(0, len(crops), args.ocr_batch_size):
                    end = start + args.ocr_batch_size
                    recognized = ocr_crops(
                        crops[start:end],
                        recognizer,
                        device,
                        denoise=args.denoise,
                        sharpen=args.sharpen,
                    )
                    for values, (text, ocr_confidence) in zip(
                        metadata[start:end], recognized, strict=True
                    ):
                        x1, y1, x2, y2, det_confidence = values
                        item = {
                            "box": (x1, y1, x2, y2),
                            "text": text,
                            "detection_confidence": det_confidence,
                            "ocr_confidence": ocr_confidence,
                        }
                        frame_results.append(item)
                        rows.writerow(
                            (
                                frame_index,
                                f"{frame_index / fps:.3f}",
                                x1,
                                y1,
                                x2,
                                y2,
                                f"{det_confidence:.6f}",
                                text,
                                f"{ocr_confidence:.6f}",
                            )
                        )

                writer.write(draw_detections(frame, frame_results, font))
                detection_count += len(frame_results)
                frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        raw_video.unlink(missing_ok=True)
        raise ValueError("Video contains no readable frames")
    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw_video),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(args.output),
            ],
            check=True,
        )
        raw_video.unlink(missing_ok=True)
    else:
        raw_video.replace(args.output)

    report = {
        "input_video": str(args.video),
        "output_video": str(args.output),
        "detections_csv": str(csv_path),
        "frames": frame_index,
        "fps": fps,
        "detections": detection_count,
        "average_detections_per_frame": detection_count / frame_index,
        "processing_seconds": time.perf_counter() - started,
        "device": str(device),
        "detector_confidence": args.confidence,
        "detector_image_size": args.imgsz,
        "preprocessing": {"denoise": args.denoise, "sharpen": args.sharpen},
        "ocr_checkpoint_epoch": torch.load(
            args.ocr_checkpoint, map_location="cpu", weights_only=True
        ).get("epoch"),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    print(json.dumps(run_video(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
