"""Evaluate task073 pose boxes, corner error and locked-margin OCR on cached frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from custom_alpr.box_context_sweep import iou
from custom_alpr.plate_pose import warp_plate
from custom_alpr.refine_box import refine_box, yolo_predictor
from custom_alpr.refine_fallback_report import E, FRAMES, ROOT, cut, read
from custom_alpr.refine_holdout_eval import metrics, wilson
from custom_alpr.video_demo import DEFAULT_DETECTOR, load_ocr, select_boxes

OUT = ROOT / "custom_alpr/outputs/pose-073"


class PoseBoxValidator(DetectionValidator):
    """Read detection labels but keep pose extras out of class-score NMS."""

    def postprocess(self, preds):
        previous_task = self.args.task
        self.args.task = "pose"  # Base implementation now passes nc=self.nc to NMS.
        try:
            return super().postprocess(preds)
        finally:
            self.args.task = previous_task


def compare(truth, predictions, base):
    result = metrics(truth, predictions)
    delta = np.array(
        [int(p == t) - int(b == t) for t, p, b in zip(truth, predictions, base, strict=True)]
    )
    rng = np.random.default_rng(73)
    ci = np.quantile(
        delta[rng.integers(0, len(truth), (100000, len(truth)))].mean(1), [0.025, 0.975]
    )
    result.update(
        fixed=wilson(int((delta == 1).sum()), len(truth)),
        broken=wilson(int((delta == -1).sum()), len(truth)),
        difference=dict(
            n=len(truth),
            rate=float(delta.mean()),
            ci95=ci.tolist(),
            method="paired bootstrap 100000 resamples seed73",
        ),
    )
    return result


def find_target(prediction, original):
    boxes = prediction.boxes.xyxy.cpu().numpy().tolist()
    matches = [(iou(original, b), i) for i, b in enumerate(boxes)]
    score, index = max(matches, default=(0, None))
    return index if score > 0 else None


def map_and_corners(pose):
    config = yaml.safe_load((ROOT / "custom_alpr/data/detection/dataset.yaml").read_text())
    config["path"] = str((ROOT / "custom_alpr/data/detection").resolve())
    yaml_path = OUT / "detection-eval.yaml"
    yaml_path.write_text(yaml.safe_dump(config))
    report = {}
    for name, model in [
        ("current_detector", YOLO(str(DEFAULT_DETECTOR), task="detect")),
        ("pose", pose),
    ]:
        cached = OUT / "default-detector-map.json"
        if name == "current_detector" and cached.exists():
            report[name] = json.loads(cached.read_text())
            if (
                report[name]["checkpoint_sha256"]
                != hashlib.sha256(DEFAULT_DETECTOR.read_bytes()).hexdigest()
            ):
                raise ValueError("Cached detector metric checkpoint mismatch")
            continue
        result = model.val(
            validator=PoseBoxValidator,
            data=str(yaml_path),
            device="cpu",
            imgsz=640,
            batch=8,
            workers=0,
            plots=False,
            project=str(OUT),
            name=f"map-{name}",
            verbose=False,
        )
        report[name] = dict(
            map50=float(result.box.map50), map50_95=float(result.box.map), val_images=1127
        )
    errors = []
    normalized = []
    normalized_quad_height = []
    matched = 0
    total = 0
    details = []
    for label in sorted((ROOT / "custom_alpr/data/pose-073-v2/labels/val").glob("*.txt")):
        path = ROOT / "custom_alpr/data/pose-073-v2/images/val" / f"{label.stem}.png"
        frame = cv2.imread(str(path))
        height, width = frame.shape[:2]
        gt = []
        for line in label.read_text().splitlines():
            v = list(map(float, line.split()))
            cx, cy, w, h = v[1:5]
            box = [
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ]
            points = np.array([[v[5 + 3 * i] * width, v[6 + 3 * i] * height] for i in range(4)])
            gt.append((box, points, h * height))
        p = pose.predict(frame, device="cpu", imgsz=640, conf=0.25, verbose=False)[0]
        edges = sorted(
            [
                (iou(b, pb), i, j)
                for i, (b, _, _) in enumerate(gt)
                for j, pb in enumerate(p.boxes.xyxy.cpu().numpy().tolist())
            ],
            reverse=True,
        )
        used_gt = set()
        used_pred = set()
        total += len(gt)
        for overlap, i, j in edges:
            if overlap < 0.5 or i in used_gt or j in used_pred:
                continue
            used_gt.add(i)
            used_pred.add(j)
            matched += 1
            points = p.keypoints.xy[j].cpu().numpy()
            distance = np.linalg.norm(points - gt[i][1], axis=1)
            errors.extend(distance.tolist())
            normalized.extend((distance / gt[i][2] * 100).tolist())
            quad = gt[i][1]
            quad_height = (
                np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])
            ) / 2
            normalized_quad_height.extend((distance / quad_height * 100).tolist())
            details.append(
                dict(
                    image=path.name,
                    gt_index=i,
                    iou=overlap,
                    errors_px=distance.tolist(),
                    height_px=gt[i][2],
                )
            )
    report["corner_error"] = dict(
        median_px=float(np.median(errors)),
        median_percent_gt_height=float(np.median(normalized)),
        height_definition="GT axis-aligned box height",
        median_percent_quad_edge_height=float(np.median(normalized_quad_height)),
        matched=wilson(matched, total),
        corners=len(errors),
        matching="unique greedy IoU>=.5, conf>=.25; error over matched four-corner GT only",
    )
    (OUT / "corner-rows.json").write_text(json.dumps(details, indent=2))
    (OUT / "detection-report.json").write_text(json.dumps(report, indent=2))
    return report


def camera_eval(camera, variants, pose, model, current, second):  # noqa: PLR0912, PLR0915
    source = json.loads((ROOT / "custom_alpr/outputs/refine-holdout-071/rows.json").read_text())[
        f"E--heldout-{camera}"
    ]
    arrays = []
    slots = []
    records = []
    images = {}
    times = []
    for r in source:
        frame = cv2.imread(str(FRAMES / f"{r['video_id']}-f{r['frame']}.png"))
        # A full-frame pipeline benchmark: refine every detected plate >=16 px.
        start = time.perf_counter()
        det = current.predict(frame, device="cpu", imgsz=640, conf=0.4, verbose=False)[0]
        detected = select_boxes(det.boxes, frame.shape[1], frame.shape[0])
        n = 0
        for box in detected:
            if box[3] - box[1] >= 16:
                refine_box(second, frame, list(box[:4]), 0.5, 320)
                n += 1
        old_ms = 1000 * (time.perf_counter() - start)
        start = time.perf_counter()
        p = pose.predict(frame, device="cpu", imgsz=640, conf=0.25, verbose=False)[0]
        pose_ms = 1000 * (time.perf_counter() - start)
        index = find_target(p, r["box"])
        crops = {}
        reason = None
        points = None
        if index is None:
            reason = "no_pose_box_intersects_original"
        else:
            box = list(map(int, np.rint(p.boxes.xyxy[index].cpu().numpy()).tolist()))
            box = [
                max(0, box[0]),
                max(0, box[1]),
                min(frame.shape[1], box[2]),
                min(frame.shape[0], box[3]),
            ]
            crops["pose_box"] = cut(frame, box)
            points = p.keypoints.xy[index].cpu().numpy()
            if p.keypoints.conf is None or float(p.keypoints.conf[index].min()) < 0.25:
                reason = "keypoint_confidence_below_025"
            else:
                for margin in variants:
                    try:
                        crops[f"warp_{margin}"] = warp_plate(frame, points, margin)
                    except ValueError:  # noqa: PERF203
                        reason = "invalid_quad"
        # Warp timing covers every pose detection, not only the annotated target.
        warp_times = {}
        for margin in variants:
            start = time.perf_counter()
            valid = 0
            for j in range(len(p.boxes)):
                if p.keypoints.conf is not None and float(p.keypoints.conf[j].min()) >= 0.25:
                    try:
                        warp_plate(frame, p.keypoints.xy[j].cpu().numpy(), margin)
                        valid += 1
                    except ValueError:
                        pass
            warp_times[str(margin)] = dict(
                ms=1000 * (time.perf_counter() - start), valid_warps=valid
            )
        item = dict(
            crop_id=r["crop_id"],
            truth=r["truth"],
            original_box=r["box"],
            pose_index=index,
            pose_crop_box=box if index is not None else None,
            pose_detection_confidence=float(p.boxes.conf[index]) if index is not None else None,
            pose_keypoint_confidence=(
                p.keypoints.conf[index].cpu().tolist()
                if index is not None and p.keypoints.conf is not None
                else None
            ),
            pose_corners=points.tolist() if points is not None else None,
            missing_reason=reason,
            texts=dict(tight=r["predictions"]["tight"], refine=r["predictions"]["refine"]),
        )
        for name, crop in crops.items():
            if crop.size:
                slots.append((len(records), name))
                arrays.append(crop)
        records.append(item)
        images[r["crop_id"]] = dict(
            crops, tight=cut(frame, r["box"]), refine=cut(frame, r["final_box"])
        )
        times.append(
            dict(
                crop_id=r["crop_id"],
                current_plus_refine_ms=old_ms,
                refined_detections=n,
                pose_detector_ms=pose_ms,
                warp=warp_times,
                pose_detections=len(p.boxes),
            )
        )
    texts = read(model, arrays)
    for (i, name), text in zip(slots, texts, strict=True):
        records[i]["texts"][name] = text
    names = ["tight", "refine", "pose_box", *[f"warp_{m}" for m in variants]]
    truth = [r["truth"] for r in records]
    base = [r["texts"]["tight"] for r in records]
    result = {}
    for name in names:
        predictions = [r["texts"].get(name, "") for r in records]
        result[name] = compare(truth, predictions, base)
        result[name]["vs_refine"] = compare(
            truth, predictions, [r["texts"]["refine"] for r in records]
        )
        result[name]["missing"] = wilson(sum(name not in r["texts"] for r in records), len(records))
    result["timing"] = dict(
        n=len(times),
        unit="full frame, every detected plate; excludes OCR",
        current_plus_refine_ms=dict(
            mean=float(np.mean([t["current_plus_refine_ms"] for t in times])),
            median=float(np.median([t["current_plus_refine_ms"] for t in times])),
        ),
        pose_plus_warp_ms={
            str(m): dict(
                mean=float(
                    np.mean([t["pose_detector_ms"] + t["warp"][str(m)]["ms"] for t in times])
                ),
                median=float(
                    np.median([t["pose_detector_ms"] + t["warp"][str(m)]["ms"] for t in times])
                ),
            )
            for m in variants
        },
    )
    (OUT / f"{camera}-rows.json").write_text(json.dumps(records, indent=2))
    (OUT / f"{camera}-timings.json").write_text(json.dumps(times, indent=2))
    (OUT / f"{camera}-report.json").write_text(json.dumps(result, indent=2))
    changed = [r for r in records if len(set(r["texts"].values())) > 1 or r["missing_reason"]]
    for page in range(0, len(changed), 8):
        subset = changed[page : page + 8]
        sheet = Image.new("RGB", (300 * len(names), 190 * len(subset)), "white")
        draw = ImageDraw.Draw(sheet)
        for y, r in enumerate(subset):
            draw.text((0, y * 190), f"{r['crop_id']} truth={r['truth']}", fill="black")
            for x, name in enumerate(names):
                crop = images[r["crop_id"]].get(name)
                draw.text(
                    (x * 300, y * 190 + 18),
                    f"{name}: {r['texts'].get(name, 'MISSING')}",
                    fill="red",
                )
                if crop is not None:
                    im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    im.thumbnail((290, 145))
                    sheet.paste(im, (x * 300, y * 190 + 40))
        sheet.save(OUT / f"{camera}-changed-{page // 8}.png")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", choices=["detection", "danang", "saigon"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    pose = YOLO(str(args.checkpoint), task="pose")
    if args.stage == "detection":
        map_and_corners(pose)
        return
    lock = OUT / "selected-margin.json"
    if args.stage == "saigon" and (OUT / "saigon-run-started.json").exists():
        raise RuntimeError("Sài Gòn already started; single-run policy forbids rerun")
    model = load_ocr(E, torch.device("cpu"))
    current = YOLO(str(DEFAULT_DETECTOR), task="detect")
    second = yolo_predictor(YOLO(str(DEFAULT_DETECTOR), task="detect"), "cpu")
    blank = np.zeros((1080, 1920, 3), np.uint8)
    for _ in range(3):
        current.predict(blank, device="cpu", imgsz=640, verbose=False)
        pose.predict(blank, device="cpu", imgsz=640, verbose=False)
        second(np.zeros((160, 320, 3), np.uint8))
    if args.stage == "danang":
        if lock.exists():
            raise RuntimeError("Margin already selected; do not tune again")
        results = camera_eval("danang-ptz", [0.0, 0.05, 0.1], pose, model, current, second)
        chosen = min(
            [0.0, 0.05, 0.1],
            key=lambda m: (
                -results[f"warp_{m}"]["exact"]["k"],
                results[f"warp_{m}"]["cer"]["rate"],
                m,
            ),
        )
        lock.write_text(
            json.dumps(
                dict(
                    margin=chosen,
                    criterion="maximize exact on Danang; ties minimize CER, then margin",
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    danang_results=results,
                ),
                indent=2,
            )
        )
        print("LOCKED MARGIN", chosen, flush=True)
    else:
        selected = json.loads(lock.read_text())
        assert (
            selected["checkpoint_sha256"]
            == hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        )
        (OUT / "saigon-run-started.json").write_text(
            json.dumps(
                dict(
                    margin=selected["margin"],
                    checkpoint_sha256=selected["checkpoint_sha256"],
                    single_run=True,
                )
            )
        )
        camera_eval("saigon-drive", [selected["margin"]], pose, model, current, second)


if __name__ == "__main__":
    main()
