"""Training-only image augmentations for custom plate OCR."""

from __future__ import annotations

import math

import cv2
import numpy as np


def motion_blur(image: np.ndarray, kernel_size: int, angle_degrees: float) -> np.ndarray:
    """Apply a directional linear blur while keeping image size and label unchanged."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel image shaped [height, width, 3]")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer of at least 3")

    center = (kernel_size - 1) / 2
    radius = center
    angle = math.radians(angle_degrees)
    offset_x = math.cos(angle) * radius
    offset_y = math.sin(angle) * radius
    start = (round(center - offset_x), round(center - offset_y))
    end = (round(center + offset_x), round(center + offset_y))
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cv2.line(kernel, start, end, color=1.0, thickness=1, lineType=cv2.LINE_AA)
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0:
        raise ValueError("Could not create a nonempty motion blur kernel")
    kernel /= kernel_sum
    return cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)


def random_motion_blur(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.2,
) -> np.ndarray:
    """Sometimes apply mild directional blur to a training crop.

    The starting distribution applies blur to 20% of crops and samples 3/5/7-pixel
    kernels with probabilities 0.5/0.35/0.15. These are tunable starting values,
    not measurements from the project dataset. Do not use this for validation or
    inference images.
    """
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if rng.random() >= probability:
        return image
    kernel_size = int(rng.choice((3, 5, 7), p=(0.5, 0.35, 0.15)))
    angle_degrees = float(rng.uniform(0, 180))
    return motion_blur(image, kernel_size, angle_degrees)
