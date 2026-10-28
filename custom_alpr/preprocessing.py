"""Light, deterministic OCR preprocessing; RGB uint8 throughout."""

import cv2
import numpy as np
from PIL import Image

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 64


def crop_quality(crop: np.ndarray) -> dict:
    """Bounded review score; this is image quality, not a recognition probability."""
    gray = cv2.cvtColor(cv2.resize(crop, (128, 64)), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.6)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    exposure = float(np.mean((gray <= 8) | (gray >= 247)))
    resolution = min(1.0, crop.shape[0] / 32, crop.shape[1] / 64)
    score = (sharpness / (sharpness + 100)) * min(1, contrast / 80)
    score *= (1 - 0.5 * exposure) * resolution
    return dict(score=float(score), sharpness=sharpness, contrast=contrast, exposure=exposure)


def padded_box(box, width: int, height: int, margin: float = 0.05) -> list[int]:
    """Expand in original image coordinates; never fabricate missing border pixels."""
    x1, y1, x2, y2 = box
    dx, dy = max(1, round((x2 - x1) * margin)), max(1, round((y2 - y1) * margin))
    return [max(0, x1 - dx), max(0, y1 - dy), min(width, x2 + dx), min(height, y2 + dy)]


def boundary_supported(corners: np.ndarray, edges: np.ndarray) -> bool:
    """Reject polygon shortcuts across rounded borders or characters."""
    supported = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    for start, end in zip(corners, np.roll(corners, -1, axis=0), strict=True):
        samples = start + np.linspace(0.1, 0.9, 64)[:, None] * (end - start)
        xs = np.clip(np.rint(samples[:, 0]).astype(int), 0, edges.shape[1] - 1)
        ys = np.clip(np.rint(samples[:, 1]).astype(int), 0, edges.shape[0] - 1)
        if np.mean(supported[ys, xs] > 0) < 0.8:
            return False
    return True


def rectify_plate(crop: np.ndarray) -> tuple[np.ndarray, list | None]:
    """Warp only a large, convex, visible plate boundary; otherwise preserve pixels.

    This contour heuristic does not infer invisible corners. Work is bounded to
    512px, while the accepted transform is applied to the original resolution.
    A small inner border in the destination avoids clipping characters at edges.
    """
    height, width = crop.shape[:2]
    if min(height, width) < 16:
        return crop, None
    scale = min(1.0, 512 / max(height, width))
    small = cv2.resize(crop, (max(1, round(width * scale)), max(1, round(height * scale))))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(polygon) / (small.shape[0] * small.shape[1])
        if len(polygon) != 4 or not cv2.isContourConvex(polygon) or not 0.55 < area < 0.97:
            continue
        points = polygon.reshape(4, 2).astype(np.float32) / scale
        # Sum/difference ordering is safe for the bounded tilts accepted below.
        sums, differences = points.sum(axis=1), np.diff(points, axis=1).ravel()
        order = [sums.argmin(), differences.argmin(), sums.argmax(), differences.argmax()]
        if len(set(order)) != 4:
            continue
        corners = points[order]
        if not boundary_supported(corners * scale, edges):
            continue
        tl, tr, br, bl = corners
        sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        out_w, out_h = max(sides[0], sides[2]), max(sides[1], sides[3])
        if not 1.0 < out_w / out_h < 6.5 or min(sides) < 12:
            continue
        if max(abs(tr[1] - tl[1]), abs(br[1] - bl[1])) > out_w * 0.4:
            continue
        if (
            points[:, 0].min() > width * 0.2
            or points[:, 0].max() < width * 0.8
            or points[:, 1].min() > height * 0.2
            or points[:, 1].max() < height * 0.8
        ):
            continue
        tilt = max(abs(tr[1] - tl[1]), abs(br[1] - bl[1])) / out_w
        distortion = max(
            sides[0] / sides[2], sides[2] / sides[0], sides[1] / sides[3], sides[3] / sides[1]
        )
        if tilt < 0.025 and distortion < 1.08:
            return crop, None
        ow, oh = round(out_w), round(out_h)
        border = max(1, round(min(ow, oh) * 0.03))
        destination = np.float32(
            [
                [border, border],
                [ow + border - 1, border],
                [ow + border - 1, oh + border - 1],
                [border, oh + border - 1],
            ]
        )
        transform = cv2.getPerspectiveTransform(corners, destination)
        output = cv2.warpPerspective(
            crop,
            transform,
            (ow + 2 * border, oh + 2 * border),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return output, corners.tolist()
    return crop, None


def prepare_ocr_input(
    crop: np.ndarray, *, denoise: bool = False, sharpen: bool = False, rectify: bool = False
) -> np.ndarray:
    """Optionally rectify, resize as in training, then denoise/sharpen.

    Filters operate at the fixed model resolution to bound runtime and avoid
    expensive full-frame filtering. With all options off, pixels match training resize.
    """
    if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3 or crop.size == 0:
        raise ValueError("OCR crop must be nonempty RGB uint8 with three channels")
    if rectify:
        crop, _ = rectify_plate(crop)
    resized = Image.fromarray(crop).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    array = np.array(resized, dtype=np.uint8, copy=True)
    if denoise:
        array = cv2.bilateralFilter(array, d=5, sigmaColor=20, sigmaSpace=2)
    if sharpen:
        blurred = cv2.GaussianBlur(array, (0, 0), sigmaX=0.7)
        array = cv2.addWeighted(array, 1.4, blurred, -0.4, 0)
    return array
