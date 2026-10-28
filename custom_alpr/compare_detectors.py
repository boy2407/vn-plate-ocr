"""Compare frozen plate detectors with common ONNX CPU preprocessing and metrics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils.metrics import ap_per_class


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match_boxes(boxes: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Confidence-ordered, one-to-one greedy matches at IoU .50 through .95."""
    correct = np.zeros((len(boxes), 10), dtype=bool)
    if not len(boxes) or not len(truth):
        return correct
    intersection = np.maximum(
        0,
        np.minimum(truth[:, None, 2:], boxes[None, :, 2:])
        - np.maximum(truth[:, None, :2], boxes[None, :, :2]),
    ).prod(axis=2)
    areas_t = np.maximum(0, truth[:, 2:] - truth[:, :2]).prod(axis=1)
    areas_p = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    overlaps = intersection / np.maximum(areas_t[:, None] + areas_p - intersection, 1e-12)
    for column, threshold in enumerate(np.linspace(0.5, 0.95, 10)):
        used = np.zeros(len(truth), dtype=bool)
        for index in range(len(boxes)):
            available = np.where(used, -1, overlaps[:, index])
            target = available.argmax()
            if available[target] >= threshold:
                correct[index, column] = True
                used[target] = True
    return correct


def summarize(rows: list[dict]) -> dict:
    true_positives, scores, targets = [], [], []
    counts = {str(c): dict(tp=0, fp=0, fn=0) for c in (0.25, 0.4, 0.5, 0.6, 0.7, 0.8)}
    for row in rows:
        boxes = np.asarray(row["boxes"], float).reshape(-1, 4)
        confidence = np.asarray(row["scores"], float)
        truth = np.asarray(row["truth"], float).reshape(-1, 4)
        order = np.argsort(-confidence, kind="stable")
        boxes, confidence = boxes[order], confidence[order]
        matched = match_boxes(boxes, truth)
        true_positives.append(matched)
        scores.append(confidence)
        targets.append(np.zeros(len(truth)))
        for threshold, count in counts.items():
            selected = confidence >= float(threshold)
            tp = int(matched[selected, 0].sum())
            count["tp"] += tp
            count["fp"] += int(selected.sum()) - tp
            count["fn"] += len(truth) - tp
    correct, confidence, truth_classes = map(np.concatenate, (true_positives, scores, targets))
    metrics = ap_per_class(correct, confidence, np.zeros(len(confidence)), truth_classes)
    for count in counts.values():
        tp, fp, fn = count["tp"], count["fp"], count["fn"]
        count.update(
            precision=tp / max(tp + fp, 1),
            recall=tp / max(tp + fn, 1),
            false_detection_fraction=fp / max(tp + fp, 1),
            miss_rate=fn / max(tp + fn, 1),
        )
    return dict(
        images=len(rows),
        boxes=len(truth_classes),
        mAP50=float(metrics[5][:, 0].mean()),
        mAP50_95=float(metrics[5].mean()),
        operating_points=counts,
    )


class OnnxDetector:
    """Same ORT provider, thread limits, letterbox and float coordinate restoration."""

    def __init__(self, path: Path, kind: str, threads: int):
        import onnxruntime as ort  # noqa: PLC0415 -- optional evaluation dependency

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.size = self.input.shape[-1]
        self.kind = kind
        self.path = path

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from open_image_models.detection.core.yolo_v9.preprocess import preprocess  # noqa: PLC0415

        pixels, ratio, padding = preprocess(image, self.size)
        raw = self.session.run(None, {self.input.name: pixels})[0]
        if self.kind == "project":
            data = raw.reshape(-1, 6)
            boxes, confidence = data[:, :4].copy(), data[:, 4]
        else:
            data = raw.reshape(-1, 7)
            boxes, confidence = data[:, 1:5].copy(), data[:, 6]
        selected = np.isfinite(confidence) & (confidence >= 0.001)
        boxes, confidence = boxes[selected], confidence[selected]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padding[0]) / ratio[0]
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padding[1]) / ratio[1]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image.shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image.shape[0])
        valid = (
            np.isfinite(boxes).all(axis=1)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        return boxes[valid], confidence[valid]


def prepare_models(root: Path, output: Path) -> dict:
    from open_image_models.detection.core.hub import download_model  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    folder = output / "models"
    folder.mkdir(exist_ok=True)
    checkpoint = root / "runs/moderate-s42/weights/best.pt"
    copied = folder / "project.pt"
    shutil.copy2(checkpoint, copied)
    exported = folder / "project.onnx"
    if not exported.is_file():
        YOLO(str(copied)).export(
            format="onnx",
            imgsz=640,
            batch=1,
            device="cpu",
            opset=17,
            dynamic=False,
            simplify=False,
            nms=True,
            conf=0.001,
            iou=0.45,
            max_det=100,
        )
    paths = {"project-yolov8n-640": (exported, "project")}
    for size in (384, 640):
        name = f"yolo-v9-t-{size}-license-plate-end2end"
        paths[f"fastalpr-yolov9t-{size}"] = (download_model(name), "fastalpr")
    return paths


def tune_threads(paths: dict, images: list[np.ndarray], output: Path) -> tuple[dict, dict]:
    detectors, timings = {}, {}
    for name, (path, kind) in paths.items():
        timings[name] = {}
        for threads in (4, 8, 16):
            detector = OnnxDetector(path, kind, threads)
            detector.predict(images[0])
            elapsed = []
            for image in images:
                started = time.perf_counter()
                detector.predict(image)
                elapsed.append(time.perf_counter() - started)
            timings[name][str(threads)] = float(np.median(elapsed) * 1000)
        chosen = min(timings[name], key=timings[name].get)
        detectors[name] = OnnxDetector(path, kind, int(chosen))
        print(f"{name}: threads={chosen}, benchmark={timings[name]}", flush=True)
        (output / "thread-benchmark.json").write_text(json.dumps(timings, indent=2))
    return detectors, timings


def load_truth(root: Path, record: dict, image: np.ndarray) -> list:
    label = root / "data" / record["label"]
    if sha256(label) != record["label_sha256"]:
        raise ValueError(f"Label hash mismatch: {label}")
    boxes = []
    height, width = image.shape[:2]
    for line in label.read_text().splitlines():
        if line.strip():
            cls, x, y, bw, bh = map(float, line.split())
            if cls != 0:
                raise ValueError("Expected single plate class")
            boxes.append(
                [
                    (x - bw / 2) * width,
                    (y - bh / 2) * height,
                    (x + bw / 2) * width,
                    (y + bh / 2) * height,
                ]
            )
    return boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    paths = prepare_models(root, output)
    inventory = json.loads((root / "inventory.json").read_text())
    records = [row for row in inventory["records"] if row["split"] == "test"]
    if len(records) != 1228:
        raise ValueError(f"Expected fixed 1228-image test, got {len(records)}")
    warmup = [cv2.imread(str(root / "data" / row["image"])) for row in records[:8]]
    detectors, timings = tune_threads(paths, warmup, output)
    predictions = {name: [] for name in paths}
    durations = {name: [] for name in paths}
    start = time.perf_counter()
    for index, record in enumerate(records):
        path = root / "data" / record["image"]
        if sha256(path) != record["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        truth = load_truth(root, record, image)
        # Interleave to reduce systematic bias from other CPU work on the host.
        names = list(detectors)
        names = names[index % len(names) :] + names[: index % len(names)]
        for name in names:
            started = time.perf_counter()
            boxes, confidence = detectors[name].predict(image)
            durations[name].append((time.perf_counter() - started) * 1000)
            predictions[name].append(
                dict(
                    image=record["image"],
                    truth=truth,
                    boxes=boxes.tolist(),
                    scores=confidence.tolist(),
                )
            )
        if (index + 1) % 100 == 0:
            progress = dict(
                images=index + 1, total=len(records), seconds=time.perf_counter() - start
            )
            print(progress, flush=True)
            (output / "progress.json").write_text(json.dumps(progress))
    report = dict(
        protocol=dict(
            split="test",
            images=len(records),
            confidence_floor=0.001,
            match_iou="0.50:0.05:0.95",
            nms_iou=0.45,
            max_det=100,
            fixed_confidence=0.4,
            runtime="ONNX Runtime CPU FP32 batch 1",
            latency="decode/hash/metrics excluded; preprocess + inference + postprocess included",
            boxes="float XYXY, same clipping/restoration for all models",
            preprocessing="shared FastALPR letterbox RGB /255; square input; scaleup=True",
            inventory_sha256=sha256(root / "inventory.json"),
        ),
        environment=dict(
            platform=platform.platform(),
            cpu=platform.processor(),
            packages={
                n: importlib.metadata.version(n)
                for n in ("onnxruntime", "onnx", "open-image-models", "ultralytics", "torch")
            },
        ),
        thread_benchmark=timings,
        models={},
    )
    for name, rows in predictions.items():
        (output / f"{name}-predictions.json").write_text(json.dumps(rows))
        metrics = summarize(rows)
        if metrics["boxes"] != 2926:
            raise ValueError("Unexpected test ground-truth count")
        values = durations[name]
        metrics.update(
            latency_ms_mean=float(np.mean(values)),
            latency_ms_median=float(np.median(values)),
            latency_ms_p95=float(np.percentile(values, 95)),
            fps=1000 / float(np.mean(values)),
            model=str(paths[name][0]),
            model_sha256=sha256(paths[name][0]),
        )
        report["models"][name] = metrics
    (output / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
