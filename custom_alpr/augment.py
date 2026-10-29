"""Training-only image augmentations for custom plate OCR."""

from __future__ import annotations

import io
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


def random_photometric(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.5,
) -> np.ndarray:
    """Sometimes jitter hue, saturation, brightness, and contrast of an RGB crop.

    Ranges (hue +/-10 OpenCV units, saturation x0.6-1.4, brightness x0.5-1.5,
    contrast x0.7-1.3) are starting values, not measured from the dataset.
    """
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if rng.random() >= probability:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-10, 10)) % 180
    hsv[..., 1] *= rng.uniform(0.6, 1.4)
    hsv[..., 2] *= rng.uniform(0.5, 1.5)
    adjusted = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    mean = adjusted.mean()
    contrast = rng.uniform(0.7, 1.3)
    return np.clip((adjusted - mean) * contrast + mean, 0, 255).astype(np.uint8)


def random_glare(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.15,
) -> np.ndarray:
    """Sometimes add a soft white elliptical highlight that imitates sun glare."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if rng.random() >= probability:
        return image
    height, width = image.shape[:2]
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    center_y, center_x = rng.uniform(0, height), rng.uniform(0, width)
    radius_y = rng.uniform(0.2, 0.6) * height
    radius_x = rng.uniform(0.15, 0.5) * width
    distance = ((rows - center_y) / radius_y) ** 2 + ((columns - center_x) / radius_x) ** 2
    alpha = rng.uniform(0.5, 0.95) * np.exp(-distance)[..., None]
    return np.clip(image * (1 - alpha) + 255 * alpha, 0, 255).astype(np.uint8)


def random_downscale(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.3,
    min_height: int = 12,
    max_height: int = 32,
) -> np.ndarray:
    """Sometimes shrink then re-enlarge a crop to imitate tiny, low-resolution plates."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if rng.random() >= probability:
        return image
    height, width = image.shape[:2]
    small_height = int(rng.integers(min_height, max_height + 1))
    small_width = max(1, round(width * small_height / height))
    small = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def random_gaussian_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.3,
    min_sigma: float = 5.0,
    max_sigma: float = 30.0,
) -> np.ndarray:
    """Sometimes add per-pixel Gaussian noise (sigma on the 0-255 scale), like high-ISO grain."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if rng.random() >= probability:
        return image
    noise = rng.normal(0.0, rng.uniform(min_sigma, max_sigma), size=image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def random_salt_pepper(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.2,
    min_amount: float = 0.005,
    max_amount: float = 0.05,
) -> np.ndarray:
    """Sometimes set a random fraction of pixels to pure black or pure white."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if rng.random() >= probability:
        return image
    amount = rng.uniform(min_amount, max_amount)
    draw = rng.random(image.shape[:2])
    noisy = image.copy()
    noisy[draw < amount / 2] = 0
    noisy[(draw >= amount / 2) & (draw < amount)] = 255
    return noisy


def _at_source_size(
    image: np.ndarray,
    rng: np.random.Generator,
    min_height: int,
    max_height: int,
    transform,
) -> np.ndarray:
    """Shrink to a random plate height, transform there, then enlarge back.

    Video is compressed at its own resolution, while the plate is still small.
    """
    height, width = image.shape[:2]
    small_height = int(rng.integers(min_height, max_height + 1))
    small_width = max(1, round(width * small_height / height))
    small = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA)
    return cv2.resize(transform(small), (width, height), interpolation=cv2.INTER_LINEAR)


def random_jpeg(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.3,
    min_height: int = 14,
    max_height: int = 48,
    min_quality: int = 25,
    max_quality: int = 60,
) -> np.ndarray:
    """Sometimes JPEG-compress a small copy: 8x8 blocking and ringing around strokes."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if rng.random() >= probability:
        return image
    quality = int(rng.integers(min_quality, max_quality + 1))

    def compress(small: np.ndarray) -> np.ndarray:
        ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("JPEG encoding failed")
        return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED).reshape(small.shape)

    return _at_source_size(image, rng, min_height, max_height, compress)


def h264_round_trip(
    image: np.ndarray, crf: int, shift: tuple[float, float] = (0.0, 0.0), frames: int = 4
) -> np.ndarray:
    """Encode a short clip of ``image`` drifting by ``shift`` px; return the last decoded frame.

    With B-frames disabled the last frame is a P-frame, so it carries the motion
    prediction artefacts of real H.264 video, not only intra-frame blocking.
    """
    import av  # noqa: PLC0415 -- optional training dependency

    height, width = image.shape[:2]
    # yuv420p needs even sides; x264 rejects frames smaller than a macroblock.
    padded_height, padded_width = max(16, height + height % 2), max(16, width + width % 2)
    padded = cv2.copyMakeBorder(
        image, 0, padded_height - height, 0, padded_width - width, cv2.BORDER_REPLICATE
    )
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="h264") as output:
        stream = output.add_stream("libx264", rate=25)
        stream.width, stream.height = padded_width, padded_height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "ultrafast", "tune": "zerolatency"}
        for index in range(frames):
            fraction = index / max(1, frames - 1)
            matrix = np.float32([[1, 0, shift[0] * fraction], [0, 1, shift[1] * fraction]])
            frame = cv2.warpAffine(
                padded, matrix, (padded_width, padded_height), borderMode=cv2.BORDER_REPLICATE
            )
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    buffer.seek(0)
    with av.open(buffer, "r", format="h264") as source:
        decoded = [frame.to_ndarray(format="rgb24") for frame in source.decode(video=0)]
    return np.ascontiguousarray(decoded[-1][:height, :width])


def random_h264(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.15,
    min_height: int = 14,
    max_height: int = 48,
    min_crf: int = 26,
    max_crf: int = 38,
    max_shift: float = 2.0,
) -> np.ndarray:
    """Sometimes pass a small copy through real H.264 (libx264) with slight motion."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if rng.random() >= probability:
        return image
    crf = int(rng.integers(min_crf, max_crf + 1))
    shift = tuple(float(value) for value in rng.uniform(-max_shift, max_shift, size=2))
    return _at_source_size(
        image, rng, min_height, max_height, lambda small: h264_round_trip(small, crf, shift)
    )


def random_compression(
    image: np.ndarray,
    rng: np.random.Generator,
    h264_probability: float = 0.15,
    jpeg_probability: float = 0.3,
) -> np.ndarray:
    """At most one compression per crop: H.264 first, otherwise maybe JPEG."""
    compressed = random_h264(image, rng, probability=h264_probability)
    if compressed is not image:
        return compressed
    return random_jpeg(image, rng, probability=jpeg_probability)


def random_blue_plate(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.08,
) -> np.ndarray:
    """Sometimes recolor a dark-on-light plate as white text on a blue background.

    Brightness is normalized between its 1st and 99th percentiles, then light
    pixels (background) map to a random blue and dark pixels (text) to near
    white, imitating Vietnamese blue plates. Crops whose plate background (judged
    in the central half, away from dark car bodies) is darker than the text, i.e.
    blue plates, are left unchanged. The default 8% raises
    blue plates seen in training from 9% of all train crops to about 16%, their
    share among real photos (847 of 5,236).
    """
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if rng.random() >= probability:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    low, high = np.percentile(gray, (1, 99))
    height, width = gray.shape
    center = gray[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    if center.mean() > np.median(center):  # few light pixels on a dark majority: light text
        return image
    lightness = np.clip((gray - low) / max(high - low, 1.0), 0, 1)[..., None]
    blue = np.array(
        [rng.uniform(10, 50), rng.uniform(50, 100), rng.uniform(140, 210)], dtype=np.float32
    )
    white = np.full(3, rng.uniform(215, 250), dtype=np.float32)
    return (blue * lightness + white * (1 - lightness)).astype(np.uint8)


def random_rotate_perspective(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.3,
    max_degrees: float = 8.0,
    max_shift: float = 0.05,
) -> np.ndarray:
    """Sometimes rotate the crop and move its corners to imitate tilted, oblique plates.

    Corners move by up to ``max_shift`` of the width/height. The warped plate is
    then shrunk toward the center just enough to stay inside the frame, so no
    character is cut off; uncovered borders repeat the edge pixels.
    """
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if rng.random() >= probability:
        return image
    height, width = image.shape[:2]
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    shifts = rng.uniform(-max_shift, max_shift, size=(4, 2)) * [width, height]
    angle = np.deg2rad(rng.uniform(-max_degrees, max_degrees))
    center = np.float32([width / 2, height / 2])
    rotation = np.float32([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    offsets = (corners - center) @ rotation.T + shifts.astype(np.float32)
    fit = min(
        1.0, width / 2 / np.abs(offsets[:, 0]).max(), height / 2 / np.abs(offsets[:, 1]).max()
    )
    moved = offsets * fit + center
    matrix = cv2.getPerspectiveTransform(corners, moved.astype(np.float32))
    return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def warp_view(
    image: np.ndarray, degrees: float, side: float, vertical: float, rng: np.random.Generator
) -> np.ndarray:
    """Rotate by ``degrees`` and show the plate as seen at an angle.

    ``side``/``vertical`` shrink the far edge (a random one of left/right, top/bottom) by
    that fraction and squeeze the far half toward the center by half of it. The result
    is shrunk to stay inside the frame, so no character is cut off.
    """
    height, width = image.shape[:2]
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    center = np.float32([width / 2, height / 2])
    offsets = corners - center
    if side:
        far = [0, 3] if rng.random() < 0.5 else [1, 2]  # left or right edge
        offsets[far, 1] *= 1 - side
        offsets[far, 0] *= 1 - side / 2
    if vertical:
        far = [0, 1] if rng.random() < 0.5 else [2, 3]  # top or bottom edge
        offsets[far, 0] *= 1 - vertical
        offsets[far, 1] *= 1 - vertical / 2
    angle = np.deg2rad(degrees)
    rotation = np.float32([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    offsets = offsets @ rotation.T
    fit = min(
        1.0, width / 2 / np.abs(offsets[:, 0]).max(), height / 2 / np.abs(offsets[:, 1]).max()
    )
    moved = offsets * fit + center
    matrix = cv2.getPerspectiveTransform(corners, moved.astype(np.float32))
    return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def random_view_tilt(
    image: np.ndarray, rng: np.random.Generator, probability: float = 0.3
) -> np.ndarray:
    """Sometimes rotate or tilt a straight plate; the three modes are equally likely.

    - rotation only: 10-40 degrees either way
    - view tilt only: 10-40% (side, vertical or both)
    - both: up to 12 degrees plus a 10-25% tilt
    Ranges were chosen by the user from preview sheets.
    """
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if rng.random() >= probability:
        return image

    def signed(low: float, high: float) -> float:
        return rng.uniform(low, high) * rng.choice((-1, 1))

    def tilts(low: float, high: float) -> tuple[float, float]:
        side, vertical = [(True, False), (False, True), (True, True)][rng.integers(3)]
        side_tilt = rng.uniform(low, high) if side else 0.0
        return side_tilt, rng.uniform(low, high) if vertical else 0.0

    mode = rng.integers(3)
    if mode == 0:
        return warp_view(image, signed(10, 40), 0.0, 0.0, rng)
    if mode == 1:
        return warp_view(image, 0.0, *tilts(0.10, 0.40), rng)
    return warp_view(image, rng.uniform(-12, 12), *tilts(0.10, 0.25), rng)


def looks_straight(image: np.ndarray, max_degrees: float = 3.0) -> bool:
    """True when the plate's long edges/text lines are level and parallel.

    Lines come from a Hough transform on the 128x64 crop. Without enough lines the plate
    is treated as not straight, so an already tilted plate is never tilted further.
    """
    gray = cv2.cvtColor(cv2.resize(image, (128, 64)), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=40, maxLineGap=4)
    if lines is None:
        return False
    found = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 90) % 180 - 90
        if abs(angle) < 35:
            found.append((angle, np.hypot(x2 - x1, y2 - y1), (y1 + y2) / 2))
    if not found:
        return False
    angles, lengths, rows = np.array(found).T
    if abs(np.average(angles, weights=lengths)) >= max_degrees:
        return False
    top, bottom = rows < 32, rows >= 32
    if top.any() and bottom.any():  # top and bottom lines converge when seen at an angle
        spread = np.average(angles[top], weights=lengths[top]) - np.average(
            angles[bottom], weights=lengths[bottom]
        )
        return bool(abs(spread) < max_degrees)
    return True
