"""Recover source frames and measure refinement on the confirmed video manifest."""

from __future__ import annotations

# Imports inside main keep manifest tooling usable without optional inference dependencies.
# ruff: noqa: PLC0415

import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "custom_alpr/data/collected"
OUT = ROOT / "custom_alpr/outputs/refine-confirmed-070"


def main():  # noqa: PLR0912, PLR0915
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from ultralytics import YOLO

    from custom_alpr.plate_tracks import normalized_text
    from custom_alpr.preprocessing import padded_box, prepare_ocr_input
    from custom_alpr.refine_box import refine_box, yolo_predictor
    from custom_alpr.video_demo import DEFAULT_DETECTOR, DEFAULT_OCR, load_ocr, ocr_arrays
    from custom_alpr.web_app import resolve_stream

    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    frames = DATA / "frames"
    frames.mkdir(exist_ok=True)
    manifest = ROOT / "custom_alpr/data/ocr/collected-v1/confirmed-tight.csv"
    meta = {r["crop_id"]: r for p in DATA.glob("*/metadata.csv") for r in csv.DictReader(p.open())}
    rows = []
    for label in csv.DictReader(manifest.open()):
        path = (manifest.parent / label["image_path"]).resolve()
        rows.append(
            dict(meta[path.stem], truth=normalized_text(label["plate_text"]), path=str(path))
        )
    detector = YOLO(str(DEFAULT_DETECTOR), task="detect")
    predict = yolo_predictor(detector, "cpu")
    predict(np.zeros((160, 320, 3), np.uint8))
    models = {}
    for name, checkpoint in [
        (
            "E",
            ROOT / "custom_alpr/outputs/train-custom-ocr/run-20261029-09-parseq-f-compress/best.pt",
        ),
        ("E-v1", DEFAULT_OCR),
    ]:
        models[name] = load_ocr(checkpoint, torch.device("cpu"))
        print(
            "SMOKE",
            name,
            ocr_arrays([np.zeros((64, 128, 3), np.uint8)], models[name], torch.device("cpu")),
            flush=True,
        )
    errors = {}
    for video in sorted({r["video_id"] for r in rows}):
        needed = [r for r in rows if r["video_id"] == video]
        try:
            if all((frames / f"{video}-f{r['frame']}.png").exists() for r in needed):
                continue
            url, _ = resolve_stream("https://www.youtube.com/watch?v=" + video)
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                raise RuntimeError("Cannot open H.264 preferred stream")
            for row in needed:
                target = int(row["frame"])
                dest = frames / f"{video}-f{target}.png"
                if dest.exists():
                    continue
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, target):
                    raise RuntimeError(f"Seek rejected at {target}")
                landed = round(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ok, frame = cap.read()
                after = round(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if not ok or landed != target or after != target + 1:
                    raise RuntimeError(
                        f"Seek mismatch requested={target} landed={landed} after={after}"
                    )
                cv2.imwrite(str(dest), frame)
                print("FRAME", video, target, flush=True)
            cap.release()
        except Exception as exc:
            errors[video] = str(exc)
            print("VIDEO ERROR", video, str(exc), flush=True)
        (OUT / "frame-errors.json").write_text(json.dumps(errors, indent=2))
    crops, valid = [], []
    for row in rows:
        frame = cv2.imread(str(frames / f"{row['video_id']}-f{row['frame']}.png"))
        if frame is None:
            continue
        box = json.loads(row["box"])
        x1, y1, x2, y2 = box
        tight = cv2.imread(row["path"])
        recovered = frame[y1:y2, x1:x2]
        if recovered.shape != tight.shape:
            raise RuntimeError(f"Frame dimensions mismatch: {row['crop_id']}")
        mae = float(np.abs(recovered.astype(float) - tight.astype(float)).mean())
        row["frame_crop_mae"] = mae
        # Re-resolving YouTube can select a different H.264 rendition. Verify
        # alignment and correlation as well as pixel error before comparing crops.
        sx, sy = max(0, x1 - 10), max(0, y1 - 10)
        match = cv2.matchTemplate(frame[sy : y2 + 10, sx : x2 + 10], tight, cv2.TM_CCOEFF_NORMED)
        _, correlation, _, location = cv2.minMaxLoc(match)
        row["frame_crop_correlation"] = correlation
        if correlation < 0.9 or location != (x1 - sx, y1 - sy):
            raise RuntimeError(
                f"Unverified source frame {row['crop_id']}: {correlation}, {location}"
            )
        start = time.perf_counter()
        final, accepted = refine_box(predict, frame, box)
        row.update(
            refined=accepted, refine_ms=1000 * (time.perf_counter() - start), final_box=final
        )
        px1, py1, px2, py2 = padded_box(box, frame.shape[1], frame.shape[0])
        rx1, ry1, rx2, ry2 = final
        crops.append([recovered, frame[py1:py2, px1:px2], frame[ry1:ry2, rx1:rx2]])
        valid.append(row)
    report = dict(requested=len(rows), measured=len(valid), errors=errors)
    for name, model in models.items():
        reads = []
        inputs = [
            prepare_ocr_input(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for group in crops for c in group
        ]
        for start in range(0, len(inputs), 16):
            reads.extend(
                normalized_text(t)
                for t, _ in ocr_arrays(inputs[start : start + 16], model, torch.device("cpu"))
            )
        for i, row in enumerate(valid):
            row[name] = reads[3 * i : 3 * i + 3]
        result = {}
        for j, variant in enumerate(["tight", "padding", "refine"]):
            result[variant] = dict(
                correct=sum(r[name][j] == r["truth"] for r in valid),
                fixed=sum(r[name][0] != r["truth"] == r[name][j] for r in valid),
                broken=sum(r[name][0] == r["truth"] != r[name][j] for r in valid),
            )
        report[name] = result
        changed = [i for i, r in enumerate(valid) if len(set(r[name])) > 1]
        for page in range(0, len(changed), 12):
            subset = changed[page : page + 12]
            sheet = Image.new("RGB", (960, 180 * len(subset)), "white")
            draw = ImageDraw.Draw(sheet)
            for y, i in enumerate(subset):
                row = valid[i]
                draw.text((5, y * 180), f"{row['crop_id']} truth={row['truth']}", fill="black")
                for j, crop in enumerate(crops[i]):
                    im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    im.thumbnail((300, 140))
                    sheet.paste(im, (j * 320, y * 180 + 35))
                    draw.text((j * 320, y * 180 + 18), row[name][j], fill="red")
            sheet.save(OUT / f"{name}-changed-{page // 12}.png")
    if valid:
        report.update(
            fallback=sum(not r["refined"] for r in valid),
            refine_ms_mean=float(np.mean([r["refine_ms"] for r in valid])),
        )
    (OUT / "rows.json").write_text(json.dumps(valid, indent=2))
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
