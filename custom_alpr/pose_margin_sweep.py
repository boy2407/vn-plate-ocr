"""Exploratory OCR scoring of configurable margins using task073's saved pose corners."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import torch

from custom_alpr.plate_pose import warp_plate
from custom_alpr.pose_eval import compare
from custom_alpr.refine_fallback_report import E, FRAMES, ROOT, read
from custom_alpr.refine_holdout_eval import wilson
from custom_alpr.video_demo import load_ocr

SOURCE = ROOT / "custom_alpr/outputs/pose-073"
OUT = ROOT / "custom_alpr/outputs/pose-margin-074"
PERCENTS = (1, 2, 3, 4, 5)


def summarize(records, percents=PERCENTS):
    truth = [r["truth"] for r in records]
    base5 = [r["sweep_texts"]["5"] for r in records]
    refine = [r["texts"]["refine"] for r in records]
    tight = [r["texts"]["tight"] for r in records]
    results = {}
    for percent in percents:
        key = str(percent)
        predictions = [r["sweep_texts"][key] for r in records]
        results[key] = dict(
            compare(truth, predictions, tight),
            vs_refine=compare(truth, predictions, refine),
            vs_5percent=compare(truth, predictions, base5),
            missing=wilson(sum(not r["valid"][key] for r in records), len(records)),
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--percents", type=int, nargs="+", default=list(PERCENTS))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    percents = sorted(set(args.percents))
    if 5 not in percents or any(p < 0 or p > 100 for p in percents):
        parser.error("Include the 5% control; percentages must be within 0-100")
    output = args.output
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    model = load_ocr(E, torch.device("cpu"))
    report = dict(
        exploratory=True,
        margins_percent=percents,
        note="Both cameras already inspected; this sweep is not a fresh held-out test.",
        inference="E OCR rerun on saved task073 corners and frames; no detector inference.",
        checkpoint_ocr=str(E),
        checkpoint_ocr_sha256=hashlib.sha256(E.read_bytes()).hexdigest(),
        cameras={},
        source_sha256={},
    )
    combined = []
    for camera in ["danang-ptz", "saigon-drive"]:
        path = SOURCE / f"{camera}-rows.json"
        report["source_sha256"][camera] = hashlib.sha256(path.read_bytes()).hexdigest()
        records = json.loads(path.read_text())
        crops, slots = [], []
        for i, row in enumerate(records):
            row["camera"] = camera
            row["sweep_texts"] = {str(p): "" for p in percents}
            row["valid"] = {str(p): False for p in percents}
            row["dimensions"] = {}
            if row["missing_reason"] is not None:
                continue
            cid = row["crop_id"]
            video, index = cid.rsplit("-f", 1)
            frame_number = index.split("-x", 1)[0]
            frame = cv2.imread(str(FRAMES / f"{video}-f{frame_number}.png"))
            if frame is None:
                raise ValueError(f"Missing cached frame: {cid}")
            for percent in percents:
                key = str(percent)
                crop = warp_plate(frame, row["pose_corners"], percent / 100)
                row["valid"][key] = True
                row["dimensions"][key] = [crop.shape[1], crop.shape[0]]
                slots.append((i, key))
                crops.append(crop)
        for (index, key), text in zip(slots, read(model, crops), strict=True):
            records[index]["sweep_texts"][key] = text
        result = summarize(records, percents)
        result["five_percent_reproduction"] = dict(
            changed_read_count=sum(
                r["sweep_texts"]["5"] != r["texts"].get("warp_0.05", "") for r in records
            ),
            original_correct=sum(r["texts"].get("warp_0.05", "") == r["truth"] for r in records),
            current_correct=result["5"]["exact"]["k"],
        )
        report["cameras"][camera] = result
        combined.extend(records)
        (output / f"{camera}-rows.json").write_text(json.dumps(records, indent=2))
        print(camera, {str(p): result[str(p)]["exact"]["k"] for p in percents}, flush=True)
    report["pooled"] = summarize(combined, percents)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print("pooled", {str(p): report["pooled"][str(p)]["exact"]["k"] for p in percents}, flush=True)


if __name__ == "__main__":
    main()
