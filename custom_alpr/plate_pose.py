"""Plate corners from the task073 YOLOv8-pose model and the perspective crop they give.

The pose model runs once per frame next to the detector. A detector box takes
the corners of the pose box it overlaps most; without a confident match it keeps
the detector crop, because on video the pose model misses plates the detector
finds (task073: 8/37 and 11/58 held-out crops).
"""

from __future__ import annotations

import cv2
import numpy as np

from custom_alpr.pose_data import order_corners

# Locked in task073 before the held-out Sài Gòn run: margin picked on Đà Nẵng only.
POSE_CONFIDENCE = 0.25
KEYPOINT_CONFIDENCE = 0.25
WARP_MARGIN = 0.05


def iou(a, b) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def warp_plate(frame, corners, margin):
    """Map corner order TL/TR/BR/BL to a rectangle; margin includes real context."""
    points = np.ascontiguousarray(corners, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Invalid corners")
    # Validate convexity without changing semantic keypoint identities.
    order_corners(points.tolist())
    cross = [
        float(np.cross(points[(i + 1) % 4] - points[i], points[(i + 2) % 4] - points[(i + 1) % 4]))
        for i in range(4)
    ]
    if min(cross) <= 0:
        raise ValueError("Invalid corner winding")
    w = round(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))
    h = round(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1])))
    if min(w, h) < 2 or max(w, h) > 4096:
        raise ValueError("Degenerate or oversized warp")
    mx, my = round(w * margin), round(h * margin)
    destination = np.array(
        [[mx, my], [mx + w - 1, my], [mx + w - 1, my + h - 1], [mx, my + h - 1]], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(
        frame,
        transform,
        (w + 2 * mx, h + 2 * my),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def match_corners(box, pose_boxes, corners, corner_confidences):
    """Corners (TL/TR/BR/BL) of the pose box overlapping ``box`` most, or None.

    None when no pose box overlaps or the weakest corner is below
    ``KEYPOINT_CONFIDENCE``; the caller then keeps the detector crop.
    """
    score, index = max(
        ((iou(box, other), i) for i, other in enumerate(pose_boxes)), default=(0, None)
    )
    if (
        not score
        or corner_confidences is None
        or min(corner_confidences[index]) < KEYPOINT_CONFIDENCE
    ):
        return None
    return [[float(x), float(y)] for x, y in corners[index]]


def corner_box(corners, width, height):
    """Axis-aligned box around the corners, clipped to the frame (for the saved crop)."""
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    return [
        max(0, int(min(xs))),
        max(0, int(min(ys))),
        min(width, int(np.ceil(max(xs)))),
        min(height, int(np.ceil(max(ys)))),
    ]
