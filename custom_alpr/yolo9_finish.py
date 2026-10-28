"""Export fine-tuned native YOLOv9-t and score the frozen detection test split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from custom_alpr.compare_detectors import OnnxDetector, load_truth, summarize
from custom_alpr.yolo9_vast import digest, native_imports
from custom_alpr.yolo9_export import FastPlateExport


def main():  # noqa: PLR0915 -- explicit staged job with evidence checks
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", default="moderate-384-s42")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    root, dataset = args.root.resolve(), args.dataset.resolve()
    native = native_imports(root)
    import onnx  # noqa: PLC0415 -- isolated export dependency

    run = root / "runs" / args.run
    result = json.loads((run / "run_result.json").read_text())
    if result["status"] != "complete":
        raise ValueError("Training must finish before final export/test")
    weights = run / "weights/best.pt"
    saved = torch.load(weights, map_location="cpu")
    model = (saved.get("ema") or saved["model"]).float().eval()
    model.fuse().eval()
    model.model[-1].export = True
    wrapper = FastPlateExport(model).eval()
    exported = run / "weights/best-end2end.onnx"
    torch.set_num_threads(4)
    torch.onnx.export(
        wrapper,
        torch.zeros((1, 3, 384, 384)),
        str(exported),
        input_names=["images"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"output": {0: "detections"}},
        dynamo=False,
    )
    graph = onnx.load(exported)
    onnx.checker.check_model(graph)
    detector = OnnxDetector(exported, "fastalpr", 16)
    # Validate actual export predictions against the trained PT on clean validation.
    from open_image_models.detection.core.yolo_v9.preprocess import preprocess  # noqa: PLC0415

    records = json.loads((dataset / "inventory.json").read_text())["records"]
    parity = []
    for record in [r for r in records if r["split"] == "val"][:8]:
        image = cv2.imread(str(dataset / "data" / record["image"]))
        pixels, ratio, padding = preprocess(image, 384)
        model.model[-1].export = False
        with torch.inference_mode():
            raw = model(torch.from_numpy(pixels))[0]
        predicted = native.validate.non_max_suppression(raw, 0.4, 0.45, max_det=100)[0].numpy()
        boxes, scores = detector.predict(image)
        boxes = boxes[scores >= 0.4]
        reference = predicted[:, :4].copy()
        reference[:, [0, 2]] = (reference[:, [0, 2]] - padding[0]) / ratio[0]
        reference[:, [1, 3]] = (reference[:, [1, 3]] - padding[1]) / ratio[1]
        reference[:, [0, 2]] = reference[:, [0, 2]].clip(0, image.shape[1])
        reference[:, [1, 3]] = reference[:, [1, 3]].clip(0, image.shape[0])
        if boxes.shape != reference.shape:
            raise ValueError("Export parity: detection counts differ")
        error = float(np.abs(boxes - reference).max()) if len(boxes) else 0
        if error > 0.25:
            raise ValueError(f"Export parity: coordinate error {error}")
        parity.append(dict(image=record["image"], max_error_px=error))
    (run / "export-parity.json").write_text(json.dumps(parity, indent=2))
    if args.export_only:
        return
    rows, durations = [], []
    for index, record in enumerate(r for r in records if r["split"] == "test"):
        image_path = dataset / "data" / record["image"]
        if digest(image_path) != record["image_sha256"]:
            raise ValueError("Test image hash mismatch")
        image = cv2.imread(str(image_path))
        truth = load_truth(dataset, record, image)
        started = time.perf_counter()
        boxes, scores = detector.predict(image)
        durations.append((time.perf_counter() - started) * 1000)
        rows.append(
            dict(image=record["image"], truth=truth, boxes=boxes.tolist(), scores=scores.tolist())
        )
        if (index + 1) % 200 == 0:
            print(f"Test {index + 1}/1228", flush=True)
    if len(rows) != 1228:
        raise ValueError("Unexpected test count")
    metrics = summarize(rows)
    if metrics["boxes"] != 2926:
        raise ValueError("Unexpected target count")
    metrics.update(latency_ms_mean=float(np.mean(durations)), fps=1000 / float(np.mean(durations)))
    baseline = json.loads(
        Path("/workspace/fastalpr-detector-eval/results/comparison.json").read_text()
    )
    report = dict(
        new=metrics,
        baselines=baseline["models"],
        protocol=baseline["protocol"],
        weights_sha256=digest(weights),
        onnx_sha256=digest(exported),
        export_parity=parity,
        note="Same frozen test/protocol; no threshold tuning on test",
    )
    (run / "test-predictions.json").write_text(json.dumps(rows))
    (run / "test-metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
