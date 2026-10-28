"""Image-only corruption budget, independent from YOLO's geometry RNG."""

from __future__ import annotations

import json
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

GROUPS = {
    "noise": ("gaussian", "poisson", "salt_pepper"),
    "blur": ("motion", "defocus"),
    "light": ("glare", "low_light", "color_cast"),
    "codec": ("jpeg", "low_resolution"),
    "weather": ("fog", "rain", "occlusion"),
}
GROUP_WEIGHTS = np.array([0.25, 0.25, 0.20, 0.20, 0.10])
CHOICE_WEIGHTS = {
    "noise": [0.65, 0.25, 0.10],
    "blur": [0.8, 0.2],
    "light": [0.45, 0.45, 0.10],
    "codec": [0.5, 0.5],
    "weather": [0.4, 0.3, 0.3],
}
EFFECTS = tuple(effect for effects in GROUPS.values() for effect in effects)


def target_box(image, boxes, rng):
    h, w = image.shape[:2]
    if boxes is not None and len(boxes):
        x1, y1, x2, y2 = boxes[int(rng.integers(len(boxes)))].astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2
    return w // 4, h // 4, 3 * w // 4, 3 * h // 4


def apply_effect(image, effect, rng, strength=1.0, boxes=None):  # noqa: PLR0912, PLR0915
    """BGR uint8 -> BGR uint8; no geometric or bounding-box changes."""
    if effect not in EFFECTS:
        raise ValueError(f"Unknown corruption: {effect}")
    image = image.copy()
    h, w = image.shape[:2]
    if effect == "gaussian":
        result = image.astype(np.float32) + rng.normal(
            0, rng.uniform(2, 12) * strength, image.shape
        )
    elif effect == "poisson":
        levels = rng.uniform(400, 1200)
        noisy = rng.poisson(image.astype(float) / 255 * levels) / levels * 255
        result = image + (noisy - image) * strength
    elif effect == "salt_pepper":
        mask = rng.random((h, w))
        amount = rng.uniform(0.001, 0.005) * strength
        image[mask < amount / 2] = 0
        image[mask > 1 - amount / 2] = 255
        result = image
    elif effect == "motion":
        size = int(rng.choice([3, 5]))
        angle = rng.uniform(0, np.pi)
        dx, dy = np.cos(angle) * (size // 2), np.sin(angle) * (size // 2)
        center = size // 2
        kernel = np.zeros((size, size), np.float32)
        cv2.line(
            kernel,
            (round(center - dx), round(center - dy)),
            (round(center + dx), round(center + dy)),
            1,
            1,
        )
        blurred = cv2.filter2D(image, -1, kernel / max(kernel.sum(), 1))
        result = cv2.addWeighted(image, 1 - strength, blurred, strength, 0)
    elif effect == "defocus":
        result = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.3, 0.8) * strength)
    elif effect == "glare":
        x1, y1, x2, y2 = target_box(image, boxes, rng)
        cx, cy = rng.uniform(x1, x2), rng.uniform(y1, y2)
        sx, sy = max(2, (x2 - x1) * 0.2), max(2, (y2 - y1) * 0.25)
        yy, xx = np.ogrid[:h, :w]
        alpha = np.exp(-((xx - cx) ** 2 / (2 * sx**2) + (yy - cy) ** 2 / (2 * sy**2)))
        alpha = alpha[..., None] * rng.uniform(0.1, 0.3) * strength
        result = image * (1 - alpha) + 255 * alpha
    elif effect == "low_light":
        gamma = 1 + (rng.uniform(0.7, 1.5) - 1) * strength
        gain = 1 + (rng.uniform(0.7, 1.2) - 1) * strength
        lut = np.clip((np.arange(256) / 255) ** gamma * gain * 255, 0, 255).astype(np.uint8)
        result = cv2.LUT(image, lut)
    elif effect == "color_cast":
        result = image * (1 + rng.uniform(-0.1, 0.1, 3) * strength)
    elif effect == "jpeg":
        quality = round(rng.uniform(55, 90) + (1 - strength) * 10)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("JPEG encoding failed")
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    elif effect == "low_resolution":
        scale = 1 - rng.uniform(0.1, 0.4) * strength
        small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
        result = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif effect == "fog":
        alpha = rng.uniform(0.03, 0.1) * strength
        result = image * (1 - alpha) + 220 * alpha
    elif effect == "rain":
        overlay = image.copy()
        for _ in range(int(rng.integers(15, 36))):
            x, y = int(rng.integers(w)), int(rng.integers(h))
            cv2.line(overlay, (x, y), (x + 3, y + int(rng.integers(8, 19))), (210, 210, 210), 1)
        result = cv2.addWeighted(image, 1 - 0.2 * strength, overlay, 0.2 * strength, 0)
    else:  # mild translucent dirt/shadow, at most ~12% of one box
        x1, y1, x2, y2 = target_box(image, boxes, rng)
        pw = max(1, round((x2 - x1) * rng.uniform(0.15, 0.3)))
        ph = max(1, round((y2 - y1) * rng.uniform(0.2, 0.4)))
        x = int(rng.integers(x1, max(x1 + 1, x2 - pw + 1)))
        y = int(rng.integers(y1, max(y1 + 1, y2 - ph + 1)))
        alpha = rng.uniform(0.25, 0.5) * strength
        image[y : y + ph, x : x + pw] = (
            image[y : y + ph, x : x + pw] * (1 - alpha) + rng.uniform(30, 150) * alpha
        )
        result = image
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_budget(image, rng, boxes=None):
    count = int(rng.choice([0, 1, 2], p=[0.40, 0.45, 0.15]))
    groups = rng.choice(list(GROUPS), size=count, replace=False, p=GROUP_WEIGHTS)
    result, effects = image, []
    for group in groups:
        choices = GROUPS[str(group)]
        weights = CHOICE_WEIGHTS[str(group)]
        effect = str(rng.choice(choices, p=weights))
        result = apply_effect(result, effect, rng, 0.7 if count == 2 else 1.0, boxes)
        effects.append(effect)
    return result, effects


class QualityTransform:
    """Worker-local RNG and counters; leave all labels/instances untouched."""

    def __init__(self, arm="moderate", log_dir=None):
        self.arm, self.log_dir = arm, log_dir
        self.rng = None
        self.worker_seed = None
        self.counts = Counter()

    def __call__(self, labels):
        worker = torch.utils.data.get_worker_info()
        seed = (worker.seed if worker else torch.initial_seed()) % (2**32)
        if self.rng is None or self.worker_seed != seed:
            self.rng = np.random.default_rng(seed + 4852)
            self.worker_seed = seed
            self.counts = Counter()
        effects = []
        if self.arm == "moderate":
            image = labels["img"]
            instances = deepcopy(labels["instances"])
            instances.convert_bbox("xyxy")
            instances.denormalize(image.shape[1], image.shape[0])
            labels["img"], effects = apply_budget(image, self.rng, instances.bboxes)
        self.counts["images"] += 1
        self.counts[f"effects_{len(effects)}"] += 1
        self.counts.update(effects)
        if self.log_dir and (self.counts["images"] == 1 or self.counts["images"] % 128 == 0):
            folder = Path(self.log_dir)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{self.arm}-{os.getpid()}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(dict(self.counts), indent=2), encoding="utf-8")
            temporary.replace(path)
        return labels
