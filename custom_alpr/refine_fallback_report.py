"""Diagnose refinement rejection without modifying production refinement (task 072)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

from custom_alpr.preprocessing import padded_box, prepare_ocr_input
from custom_alpr.refine_box import accept_refined, refine_box, yolo_predictor
from custom_alpr.video_demo import DEFAULT_DETECTOR, load_ocr, ocr_arrays

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "custom_alpr/outputs/refine-fallback-072"
FRAMES = ROOT / "custom_alpr/data/collected/frames"
E = ROOT / "custom_alpr/outputs/train-custom-ocr/run-20261029-09-parseq-f-compress/best.pt"
REASONS = ("no_candidate", "low_confidence", "aspect", "area", "outside_slack", "accepted")


def classify(first, candidate):
    """Assign exactly one reason in accept_refined order; fail on degenerate boxes."""
    if candidate is None:
        return "no_candidate"
    box, confidence = candidate["box"], candidate["confidence"]
    w, h = box[2] - box[0], box[3] - box[1]
    if w <= 0 or h <= 0:
        raise ValueError("Degenerate candidate: cannot assign requested aspect/area categories")
    if confidence < 0.4:
        return "low_confidence"
    if not 1 <= w / h <= 6:
        return "aspect"
    area = (first[2] - first[0]) * (first[3] - first[1])
    if not area or not 0.2 <= w * h / area <= 0.95:
        return "area"
    if not accept_refined(first, box, confidence, 0.5):
        return "outside_slack"
    return "accepted"


def candidates(predict, frame, box):
    """Use the identical context, resize, rounding and clipping as refine_box."""
    height, width = frame.shape[:2]
    x, y, a, b = padded_box(box, width, height, 0.5)
    region = frame[y:b, x:a]
    scale = 320 / max(region.shape[:2])
    enlarged = cv2.resize(
        region,
        (max(1, round(region.shape[1] * scale)), max(1, round(region.shape[0] * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    found = []
    for cx, cy, ca, cb, conf in predict(enlarged):
        mapped = [
            max(0, round(x + cx / scale)),
            max(0, round(y + cy / scale)),
            min(width, round(x + ca / scale)),
            min(height, round(y + cb / scale)),
        ]
        w, h = mapped[2] - mapped[0], mapped[3] - mapped[1]
        found.append(
            dict(
                box=mapped,
                confidence=conf,
                aspect=w / h if h else None,
                area_ratio=w * h / ((box[2] - box[0]) * (box[3] - box[1])),
            )
        )
    return sorted(found, key=lambda c: c["confidence"], reverse=True)


def read(model, crops):
    texts = []
    inputs = [prepare_ocr_input(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
    for i in range(0, len(inputs), 16):
        texts.extend(t for t, _ in ocr_arrays(inputs[i : i + 16], model, torch.device("cpu")))
    return texts


def cut(frame, box):
    x, y, a, b = box
    return frame[y:b, x:a]


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads((ROOT / "custom_alpr/outputs/refine-holdout-071/rows.json").read_text())
    records = []
    for camera in ["saigon-drive", "danang-ptz"]:
        records.extend(dict(r, camera=camera) for r in source[f"E--heldout-{camera}"])
    predict = yolo_predictor(YOLO(str(DEFAULT_DETECTOR), task="detect"), "cpu")
    predict(np.zeros((160, 320, 3), np.uint8))
    model = load_ocr(E, torch.device("cpu"))
    crops = []
    for r in records:
        frame = cv2.imread(str(FRAMES / f"{r['video_id']}-f{r['frame']}.png"))
        found = candidates(predict, frame, r["box"])
        best = found[0] if found else None
        reason = classify(r["box"], best)
        # Independently check the diagnostic reproduces the production decision.
        final, accepted = refine_box(predict, frame, r["box"], 0.5, 320)
        assert accepted == (reason == "accepted") and final == (
            best["box"] if accepted else r["box"]
        )
        assert accepted == r["refined"] and final == r["final_box"]
        r.update(
            reason=reason,
            candidates=found,
            candidate_count=len(found),
            second_candidate_ge_04=len(found) > 1 and found[1]["confidence"] >= 0.4,
            original_height=r["box"][3] - r["box"][1],
            confidence=best["confidence"] if best else None,
            aspect=best["aspect"] if best else None,
            area_ratio=best["area_ratio"] if best else None,
        )
        crops.extend(
            [cut(frame, r["box"]), cut(frame, best["box"]) if best else cut(frame, r["box"])]
        )
    reads = read(model, crops)
    for i, r in enumerate(records):
        r["E_tight"] = reads[2 * i]
        r["E_candidate"] = reads[2 * i + 1] if r["candidates"] else None
        assert r["E_tight"] == r["predictions"]["tight"]
    counts = {
        c: {
            reason: sum(r["camera"] == c and r["reason"] == reason for r in records)
            for reason in REASONS
        }
        for c in ["saigon-drive", "danang-ptz"]
    }
    (OUT / "rows.json").write_text(json.dumps(records, indent=2))
    (OUT / "counts.json").write_text(json.dumps(counts, indent=2))
    for camera in counts:
        fallback = [r for r in records if r["camera"] == camera and r["reason"] != "accepted"]
        sheet = Image.new("RGB", (1100, 240 * len(fallback)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(fallback):
            y = i * 240
            frame = cv2.imread(str(FRAMES / f"{r['video_id']}-f{r['frame']}.png"))
            box = r["box"]
            best = r["candidates"][0] if r["candidates"] else None
            context = padded_box(box, frame.shape[1], frame.shape[0], 0.5)
            x0, y0, _, _ = context
            view = Image.fromarray(cv2.cvtColor(cut(frame, context), cv2.COLOR_BGR2RGB))
            vd = ImageDraw.Draw(view)
            for b, color in [(box, "lime")] + ([(best["box"], "red")] if best else []):
                vd.rectangle(
                    (b[0] - x0, b[1] - y0, b[2] - x0 - 1, b[3] - y0 - 1), outline=color, width=2
                )
            view.thumbnail((460, 190))
            sheet.paste(view, (5, y + 42))
            draw.text((5, y), f"{r['crop_id']} {r['reason']} truth={r['truth']}", fill="black")
            draw.text(
                (5, y + 18),
                f"conf={r['confidence']} aspect={r['aspect']} "
                f"area={r['area_ratio']} candidates={r['candidate_count']}",
                fill="black",
            )
            for j, (b, text) in enumerate(
                [(box, r["E_tight"])] + ([(best["box"], r["E_candidate"])] if best else [])
            ):
                im = Image.fromarray(cv2.cvtColor(cut(frame, b), cv2.COLOR_BGR2RGB))
                im.thumbnail((280, 180))
                sheet.paste(im, (480 + j * 310, y + 65))
                draw.text(
                    (480 + j * 310, y + 45), f"{['tight', 'candidate'][j]}: {text}", fill="red"
                )
        sheet.save(OUT / f"{camera}-fallback.png")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
