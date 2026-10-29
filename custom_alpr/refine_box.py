"""Tighten a loose plate box with a second detector pass on the enlarged crop.

On video the detector's box often includes bumper and bracket, and OCR trained
on tight crops then misreads plates it would read on a tight crop. The second
pass runs the same detector on the upscaled region around the box. Its box is
used only when it passes sanity checks; otherwise the first-pass box stands,
so a failed refinement can never make the OCR input worse than before.
"""

from __future__ import annotations

MIN_CONFIDENCE = 0.4
MIN_AREA, MAX_AREA = 0.2, 0.95  # Refined area relative to the first-pass box.
MIN_ASPECT, MAX_ASPECT = 1.0, 6.0  # Width/height of VN plates, two-line to one-line.
# Margin around the first-pass box and long side of the second-pass input. On
# 429 labelled plates, 0.5/320 tightened boxes (median IoU 0.892 -> 0.925, none
# worse by >0.1); 0.05/640 shrank good boxes into the plate (148 worse, OCR
# agreement with the truth crop 92% -> 51%): the detector expects small plates.
CONTEXT = 0.5
SIZE = 320


def accept_refined(first: list, refined: list, confidence: float, context: float = CONTEXT) -> bool:
    """Whether a second-pass box may replace the first-pass box."""
    x1, y1, x2, y2 = refined
    width, height = x2 - x1, y2 - y1
    if confidence < MIN_CONFIDENCE or width <= 0 or height <= 0:
        return False
    if not MIN_ASPECT <= width / height <= MAX_ASPECT:
        return False
    area = (first[2] - first[0]) * (first[3] - first[1])
    if not area or not MIN_AREA <= width * height / area <= MAX_AREA:
        return False
    # Some slack: the second pass sees ``context`` beyond the first box.
    slack = 0.5 * context * max(first[2] - first[0], first[3] - first[1])
    return (
        x1 >= first[0] - slack
        and y1 >= first[1] - slack
        and x2 <= first[2] + slack
        and y2 <= first[3] + slack
    )


def refine_box(predict, image, box: list, context: float = CONTEXT, size: int = SIZE) -> tuple[list, bool]:
    """(box, refined): the tightened box in image coordinates, or ``box`` unchanged.

    ``predict(region)`` runs the detector on an image and returns
    ``[(x1, y1, x2, y2, confidence), ...]`` in that image's pixels.
    """
    import cv2  # noqa: PLC0415 -- keep the checks importable without OpenCV

    from custom_alpr.preprocessing import padded_box  # noqa: PLC0415

    height, width = image.shape[:2]
    rx1, ry1, rx2, ry2 = padded_box(box, width, height, context)
    region = image[ry1:ry2, rx1:rx2]
    if region.size == 0:
        return box, False
    scale = size / max(region.shape[:2])
    enlarged = cv2.resize(
        region,
        (max(1, round(region.shape[1] * scale)), max(1, round(region.shape[0] * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    candidates = predict(enlarged)
    if not candidates:
        return box, False
    x1, y1, x2, y2, confidence = max(candidates, key=lambda item: item[4])
    refined = [
        max(0, round(rx1 + x1 / scale)),
        max(0, round(ry1 + y1 / scale)),
        min(width, round(rx1 + x2 / scale)),
        min(height, round(ry1 + y2 / scale)),
    ]
    if not accept_refined(box, refined, confidence, context):
        return box, False
    return refined, True


def yolo_predictor(detector, device, confidence: float = 0.25):
    """``predict`` callable for :func:`refine_box` from an Ultralytics model."""

    def predict(region):
        imgsz = max(32, -(-max(region.shape[:2]) // 32) * 32)  # stride multiple covering the region
        boxes = detector.predict(region, conf=confidence, imgsz=imgsz, device=device, verbose=False)[0].boxes
        return [(*row[:4], row[4]) for row in boxes.data[:, :5].tolist()]

    return predict
