"""Generate synthetic plate crops by swapping characters inside the Kaggle `_lp` renders.

Each render is a clean plate of one layout (type1..type7) whose label is known. Characters
are found as connected components, so a new plate is made by taking a render of the same
layout, painting each character box with the plate background, and drawing a glyph of the
new character cut from another render of that layout. Plate colour, border, separators and
spacing therefore stay exactly as in the original renders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from custom_alpr.augment import random_downscale, random_h264, random_motion_blur, random_view_tilt

RAW = Path("custom_alpr/data/raw/kaggle-vietnamese-license-plate-ocr")
TWO_ROW_ASPECT = 0.4  # height / width: 1-row renders are ~0.21, 2-row ones 0.5-0.7
# Synthetic plates per layout: few real samples (type5, type6) or many dev errors (type2..type7).
DEFAULT_COUNTS = {
    "type1": 200,
    "type2": 500,
    "type3": 300,
    "type4": 300,
    "type5": 300,
    "type6": 500,
    "type7": 300,
}
# Letters are drawn to fill the gap up to LETTER_TARGET plates per letter (train real + renders).
LETTER_TARGET = 400
RARE_LAYOUT_TARGET = 40  # I and J: only seen at the end of the 99999A(A) layout
RARE_LAYOUT_LETTERS = set("IJ")
# Digits are never scarce; the ones run D misreads most on dev are drawn more often.
DIGIT_WEIGHTS = dict.fromkeys("869320", 2.0)
CONFUSABLE_GROUPS = (frozenset("32589"), frozenset("17"), frozenset("086"), frozenset("NHM"))


@dataclass
class Render:
    layout: str
    two_rows: bool
    image: np.ndarray  # RGB uint8
    background: np.ndarray  # RGB float
    ink: np.ndarray  # RGB float
    groups: list[str]  # label split by row/group, e.g. ["29A1", "00274"]
    boxes: list[list[tuple[int, int, int, int]]]  # per group, per character: x0, y0, x1, y1


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path)
    if image.mode == "RGBA":
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        canvas.paste(image, mask=image.getchannel("A"))
        image = canvas
    return np.asarray(image.convert("RGB"))


def ink_alpha(image: np.ndarray, background: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """0 on background, 1 on text, anti-aliased edges in between."""
    span = max(float(np.linalg.norm(ink - background)), 1.0)
    alpha = np.linalg.norm(image.astype(np.float32) - background, axis=2) / span
    # Renders have faint shading; values below 0.25 are background, not anti-aliased ink.
    return np.clip((alpha - 0.25) / 0.6, 0, 1)


def segment(image: np.ndarray, groups: list[str], two_rows: bool):
    """Return background, ink colour and character boxes, or None if they do not match the label."""
    height, width = image.shape[:2]
    inner = image[height // 5 : -height // 5, width // 10 : -width // 10].reshape(-1, 3)
    background = np.median(inner, axis=0).astype(np.float32)
    for _ in range(2):  # narrow fonts cover most of the middle; re-estimate from non-text pixels
        distance = np.linalg.norm(image.astype(np.float32) - background, axis=2)
        mask = (distance > 0.5 * distance.max()).astype(np.uint8)
        inner_text = mask[height // 5 : -height // 5, width // 10 : -width // 10].reshape(-1)
        background = np.median(inner[inner_text == 0], axis=0).astype(np.float32)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    parts = []
    for x, y, w, h, area in stats[1:count]:
        touches_edge = x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2
        if touches_edge or w > width * 0.5 or area < 4:
            continue  # plate border or speck
        parts.append((x, y, x + w, y + h))
    if not parts:
        return None
    ink = np.median(image[mask.astype(bool)].reshape(-1, 3), axis=0).astype(np.float32)

    rows = [parts]
    if two_rows:
        middle = height / 2
        top = [p for p in parts if (p[1] + p[3]) / 2 < middle]
        rows = [top, [p for p in parts if p not in top]]
    if len(rows) != len(groups) and not two_rows:
        rows = [parts]
    boxes = []
    texts = groups if two_rows else ["".join(groups)]
    for row, text in zip(rows, texts, strict=False):
        if not row:
            return None
        tallest = max(p[3] - p[1] for p in row)
        characters = sorted((p for p in row if p[3] - p[1] >= 0.6 * tallest), key=lambda p: p[0])
        if len(characters) != len(text):
            return None
        boxes.append(characters)
    if len(boxes) != len(texts):
        return None
    return background, ink, texts, boxes


def load_renders(raw: Path) -> dict[str, list[Render]]:
    renders: dict[str, list[Render]] = {}
    with (raw / "labels" / "gen_labels.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["Name"]
            layout = name.split("_")[0]
            groups = row["Label"].split()
            image = load_rgb(raw / "generated" / name)
            two_rows = image.shape[0] / image.shape[1] > TWO_ROW_ASPECT
            found = segment(image, groups, two_rows)
            if found is None:
                continue
            background, ink, texts, boxes = found
            render = Render(layout, two_rows, image, background, ink, texts, boxes)
            renders.setdefault(layout, []).append(render)
    return renders


def glyph_library(renders: list[Render]) -> dict[str, list[np.ndarray]]:
    """Alpha masks of every character seen in one layout."""
    library: dict[str, list[np.ndarray]] = {}
    for render in renders:
        alpha = ink_alpha(render.image, render.background, render.ink)
        for text, boxes in zip(render.groups, render.boxes, strict=True):
            for character, (x0, y0, x1, y1) in zip(text, boxes, strict=True):
                crop = alpha[y0:y1, x0:x1]
                # Shading near the text lifts the whole box; its 30th percentile is background.
                floor = min(float(np.percentile(crop, 30)), 0.5)
                glyph = np.clip((crop - floor) / (1 - floor), 0, 1)
                library.setdefault(character, []).append(glyph)
    return library


def letter_gap(letter: str, plates_with: dict[str, int]) -> float:
    target = RARE_LAYOUT_TARGET if letter in RARE_LAYOUT_LETTERS else LETTER_TARGET
    return max(target - plates_with.get(letter, 0), 0)


def new_text(
    template: str,
    library: dict[str, list[np.ndarray]],
    plates_with: dict[str, int],
    provinces: list[str],
    rng: np.random.Generator,
    confusable_weights: dict[str, float] | None = None,
    confusable_generated: dict[str, int] | None = None,
) -> str:
    """Keep the template's letter/digit pattern (and Đ); letters fill the largest gaps first.

    `plates_with` counts plates containing each letter and is updated with the new plate.
    A leading 2-digit province code is replaced by a code seen in the training labels.
    """
    digits = [c for c in library if c.isdigit()]
    rare_layout = template[:5].isdigit() and not template[5:].isdigit()
    letters = [
        c
        for c in library
        if not c.isdigit() and c != "Đ" and (rare_layout or c not in RARE_LAYOUT_LETTERS)
    ]

    def draw(pool: list[str], weights: np.ndarray) -> str:
        if confusable_weights:
            weights = weights * np.array([confusable_weights.get(c, 1.0) for c in pool])
        if weights.sum() == 0:
            weights = np.ones(len(pool))
        return pool[rng.choice(len(pool), p=weights / weights.sum())]

    out = []
    for character in template:
        if character == "Đ" or (not character.isdigit() and not letters):
            out.append(character)
        elif character.isdigit():
            out.append(draw(digits, np.array([DIGIT_WEIGHTS.get(c, 1.0) for c in digits])))
        else:
            letter_weights = []
            for candidate in letters:
                gap = letter_gap(candidate, plates_with)
                if confusable_weights and candidate in confusable_weights:
                    remaining = LETTER_TARGET - (confusable_generated or {}).get(candidate, 0)
                    gap = max(gap, remaining, 0)
                letter_weights.append(gap)
            out.append(draw(letters, np.array(letter_weights)))
    if template[:2].isdigit() and not template[2].isdigit():
        out[:2] = provinces[rng.integers(len(provinces))]
    for letter in set(out) - set(digits):
        plates_with[letter] = plates_with.get(letter, 0) + 1
        if confusable_generated is not None and letter in confusable_generated:
            confusable_generated[letter] += 1
    return "".join(out)


def load_confusable_weights(report_path: Path) -> dict[str, float]:
    """Weight requested confusion groups by observed video pair counts."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pairs: dict[tuple[str, str], int] = {}
    for model_report in report.values():
        if not isinstance(model_report, dict):
            continue
        for pair, count in model_report.get("free", {}).get("top_confusions", []):
            source, target = pair.split("->", 1)
            pairs[(source, target)] = pairs.get((source, target), 0) + int(count)
    members = set().union(*CONFUSABLE_GROUPS)
    weights = {character: 1.0 for character in members}
    for (source, target), count in pairs.items():
        if source in members:
            weights[source] += count
        if target in members:
            weights[target] += count
    return weights


def make_vehicle_context(
    plate: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Place a synthetic plate in a textured car-body crop with a visible tail light."""
    plate_height, plate_width = plate.shape[:2]
    fractions = tuple(float(rng.uniform(0.10, 0.60)) for _ in range(4))  # left, top, right, bottom
    left, top, right, bottom = [
        round(value * size)
        for value, size in zip(
            fractions, (plate_width, plate_height, plate_width, plate_height), strict=True
        )
    ]
    width, height = plate_width + left + right, plate_height + top + bottom
    paints = np.array(
        [(42, 45, 49), (86, 91, 95), (167, 171, 174), (127, 36, 30), (32, 54, 74)],
        dtype=np.float32,
    )
    base = paints[int(rng.integers(len(paints)))]
    rows = np.linspace(1.16, 0.78, height, dtype=np.float32)[:, None, None]
    texture = rng.normal(0.0, 5.0, size=(height, width, 1)).astype(np.float32)
    canvas = np.clip(base[None, None, :] * rows + texture, 0, 255).astype(np.uint8)

    # Broad panel seam and bumper cues make context read as a vehicle rear.
    seam_y = min(height - 1, top + plate_height + max(2, round(0.18 * plate_height)))
    cv2.line(canvas, (0, seam_y), (width - 1, seam_y), (20, 22, 24), max(1, plate_height // 45))
    bumper_y = min(height - 1, seam_y + max(2, round(0.12 * plate_height)))
    cv2.line(
        canvas, (0, bumper_y), (width - 1, bumper_y), (111, 113, 113), max(1, plate_height // 30)
    )

    side_left = bool(rng.integers(2))
    side_space = left if side_left else right
    light_width = max(4, min(round(0.34 * plate_width), side_space - 2))
    light_height = max(4, round(0.42 * plate_height))
    if side_space >= 6:
        light_x = (
            max(0, (left - light_width) // 2)
            if side_left
            else min(width - light_width, left + plate_width + (right - light_width) // 2)
        )
        light_y = max(0, min(height - light_height, top + round(0.18 * plate_height)))
        cv2.rectangle(
            canvas,
            (light_x, light_y),
            (light_x + light_width, light_y + light_height),
            (28, 20, 19),
            -1,
        )
        cv2.rectangle(
            canvas,
            (light_x + 1, light_y + 1),
            (light_x + light_width - 1, light_y + light_height - 1),
            (160, 27, 23),
            -1,
        )
        cv2.line(
            canvas,
            (light_x + 2, light_y + light_height // 2),
            (light_x + light_width - 2, light_y + light_height // 2),
            (242, 117, 49),
            max(1, light_height // 8),
        )
        cv2.line(
            canvas,
            (light_x + 2, light_y + light_height // 3),
            (light_x + light_width - 2, light_y + light_height // 3),
            (230, 75, 47),
            1,
        )

    # A slight shadow separates the plate from the car paint, then the plate is pasted.
    cv2.rectangle(
        canvas,
        (max(0, left - 2), max(0, top - 2)),
        (min(width - 1, left + plate_width + 1), min(height - 1, top + plate_height + 1)),
        (15, 16, 17),
        -1,
    )
    resized = cv2.resize(plate, (plate_width, plate_height), interpolation=cv2.INTER_AREA)
    canvas[top : top + plate_height, left : left + plate_width] = resized
    return canvas, fractions


def apply_video_degradation(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    motion_blur_probability: float = 0.5,
    h264_probability: float = 0.4,
    view_tilt_probability: float = 0.3,
) -> tuple[np.ndarray, dict]:
    """Apply the video-like geometry, resolution, blur, and H.264 transforms."""
    flags = {"downscale": True, "motion_blur": False, "h264": False, "view_tilt": False}
    if rng.random() < view_tilt_probability:
        image = random_view_tilt(image, rng, probability=1.0)
        flags["view_tilt"] = True
    image = random_downscale(image, rng, probability=1.0, min_height=12, max_height=24)
    if rng.random() < motion_blur_probability:
        image = random_motion_blur(image, rng, probability=1.0)
        flags["motion_blur"] = True
    if rng.random() < h264_probability:
        image = random_h264(image, rng, probability=1.0, min_height=12, max_height=24)
        flags["h264"] = True
    return image, flags


def write_preview_grid(
    rows: list[dict], image_root: Path, preview_path: Path, count: int = 64
) -> None:
    """Save a labeled contact sheet from generated training crops."""
    selected = rows[: min(count, len(rows))]
    columns, cell_width, cell_height = 8, 192, 118
    grid = Image.new(
        "RGB",
        (columns * cell_width, ((len(selected) + columns - 1) // columns) * cell_height),
        (245, 245, 245),
    )
    from PIL import ImageDraw  # noqa: PLC0415

    draw = ImageDraw.Draw(grid)
    for index, row in enumerate(selected):
        image = Image.open(image_root / row["image_path"]).convert("RGB")
        image.thumbnail((cell_width - 8, 88), Image.Resampling.BILINEAR)
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        grid.paste(image, (x + (cell_width - image.width) // 2, y + 2))
        draw.text((x + 4, y + 92), row["plate_text"], fill=(10, 10, 10))
        margins = row.get("context_margins", "")
        draw.text((x + 4, y + 104), f"margins={margins}", fill=(50, 50, 50))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(preview_path)


def count_plates_with(manifest: Path) -> tuple[dict[str, int], list[str]]:
    """Plates containing each letter, and the province codes that start real plates."""
    plates_with: dict[str, int] = {}
    provinces = set()
    with manifest.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = row["plate_text"]
            for character in set(text):
                if not character.isdigit():
                    plates_with[character] = plates_with.get(character, 0) + 1
            if text[:2].isdigit() and not text[2:3].isdigit():
                provinces.add(text[:2])
    return plates_with, sorted(provinces)


def ink_share(glyphs: list[np.ndarray]) -> float:
    return float(np.median([glyph.mean() for glyph in glyphs]))


def boldness(local: dict[str, list[np.ndarray]], shared: dict[str, list[np.ndarray]]) -> float:
    """How much more ink this layout's font puts in a character than the average font."""
    return float(np.median([ink_share(local[c]) / ink_share(shared[c]) for c in local]))


def background_patch(image, alpha, background, box) -> np.ndarray:
    """Plate background for a box, blended column by column from the rows just above and below.

    Renders have a vertical shading gradient; a flat fill would leave a visible rectangle.
    Sample pixels that are ink (border, the other text row) fall back to the median colour.
    """
    x0, y0, x1, y1 = box
    height = image.shape[0]

    def edge(row: int) -> np.ndarray:
        row = min(max(row, 0), height - 1)
        pixels = image[row, x0:x1].copy()
        pixels[alpha[row, x0:x1] > 0] = background
        return pixels

    top, bottom = edge(y0 - 1), edge(y1)
    weight = np.linspace(0, 1, y1 - y0, dtype=np.float32)[:, None, None]
    return top[None] * (1 - weight) + bottom[None] * weight


def compose(
    render: Render,
    groups: list[str],
    library,
    rng: np.random.Generator,
    target_ink: dict[str, float],
) -> np.ndarray:
    """Erase the render's characters, then draw `groups` in the same boxes.

    `target_ink` gives the ink share a glyph should have in this font; thinner glyphs borrowed
    from another layout are dilated towards it.
    """
    image = render.image.astype(np.float32)
    alpha = ink_alpha(render.image, render.background, render.ink)
    for boxes in render.boxes:
        for x0, y0, x1, y1 in boxes:
            image[y0 - 2 : y1 + 2, x0 - 2 : x1 + 2] = background_patch(
                image, alpha, render.background, (x0 - 2, y0 - 2, x1 + 2, y1 + 2)
            )
    for text, boxes in zip(groups, render.boxes, strict=True):
        for character, (x0, y0, x1, y1) in zip(text, boxes, strict=True):
            glyph = library[character][rng.integers(len(library[character]))]
            height = y1 - y0
            width = max(1, round(glyph.shape[1] * height / glyph.shape[0]))
            glyph = cv2.resize(glyph, (width, height), interpolation=cv2.INTER_AREA)
            for _ in range(3):
                if glyph.mean() >= 0.9 * target_ink[character]:
                    break
                glyph = cv2.dilate(glyph, np.ones((2, 2), np.float32))
            left = (x0 + x1) // 2 - width // 2
            left = max(0, min(left, image.shape[1] - width))
            alpha = glyph[..., None]
            region = image[y0:y1, left : left + width]
            image[y0:y1, left : left + width] = region * (1 - alpha) + render.ink * alpha
    return np.clip(image, 0, 255).astype(np.uint8)


def add_finish(image: np.ndarray, render: Render, rng: np.random.Generator) -> np.ndarray:
    """Make a clean composed plate look like a real metal plate.

    - embossed characters: a light edge on the upper left and a dark edge on the lower right
    - gloss: brightness changes smoothly along a random direction (x0.85-1.15)
    - cast shadow (half the plates): one side darkened x0.55-0.85 behind a soft edge
    """
    out = image.astype(np.float32)
    height, width = out.shape[:2]
    alpha = ink_alpha(image, render.background, render.ink)
    raised = np.roll(alpha, (-1, -1), axis=(0, 1)) - np.roll(alpha, (1, 1), axis=(0, 1))
    out += rng.uniform(15, 35) * raised[..., None]

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(angle) * xs / width + np.sin(angle) * ys / height) % 1.0
    low, high = sorted(rng.uniform(0.85, 1.15, size=2))
    out *= (low + (high - low) * ramp)[..., None]

    if rng.random() < 0.5:
        angle = rng.uniform(0, 2 * np.pi)
        distance = np.cos(angle) * (xs - width / 2) + np.sin(angle) * (ys - height / 2)
        edge = rng.uniform(-0.3, 0.3) * max(width, height)
        softness = rng.uniform(2, 8)
        shade = 1 / (1 + np.exp(-(distance - edge) / softness))
        out *= (1 - (1 - rng.uniform(0.55, 0.85)) * shade)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def generate(  # noqa: PLR0913
    counts: dict[str, int],
    output: Path,
    seed: int,
    train: Path,
    raw: Path = RAW,
    *,
    confusable_weights: dict[str, float] | None = None,
    video_like: bool = False,
    motion_blur_probability: float = 0.5,
    h264_probability: float = 0.4,
    view_tilt_probability: float = 0.3,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    renders = load_renders(raw)
    plates_with, provinces = count_plates_with(train)
    (output / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    confusable_generated = (
        {character: 0 for group in CONFUSABLE_GROUPS for character in group if character.isalpha()}
        if confusable_weights
        else None
    )
    # Glyphs are colour-free alpha masks, so a letter missing from one layout (blue plates only
    # use a few series letters) is borrowed from the others and painted in this layout's ink.
    shared = glyph_library([render for pool in renders.values() for render in pool])
    for layout, count in counts.items():
        pool = renders[layout]
        # Prefer glyphs from the same layout and row count: 1-row renders use a narrower font.
        libraries, target_inks = {}, {}
        for two_rows in (False, True):
            same_rows = [r for r in pool if r.two_rows == two_rows]
            local = glyph_library(same_rows) or glyph_library(pool)
            libraries[two_rows] = {**shared, **glyph_library(pool), **local}
            bold = boldness(local, shared)
            target_inks[two_rows] = {
                c: ink_share(local[c]) if c in local else bold * ink_share(shared[c])
                for c in libraries[two_rows]
            }
        for index in range(count):
            render = pool[rng.integers(len(pool))]
            library = libraries[render.two_rows]
            text = new_text(
                "".join(render.groups),
                library,
                plates_with,
                provinces,
                rng,
                confusable_weights,
                confusable_generated,
            )
            groups, start = [], 0
            for group in render.groups:
                groups.append(text[start : start + len(group)])
                start += len(group)
            image = compose(render, groups, library, rng, target_inks[render.two_rows])
            image = add_finish(image, render, rng)
            context_margins = None
            degradation = {
                "downscale": False,
                "motion_blur": False,
                "h264": False,
                "view_tilt": False,
            }
            if video_like:
                image, margins = make_vehicle_context(image, rng)
                context_margins = ",".join(f"{value:.3f}" for value in margins)
                image, degradation = apply_video_degradation(
                    image,
                    rng,
                    motion_blur_probability=motion_blur_probability,
                    h264_probability=h264_probability,
                    view_tilt_probability=view_tilt_probability,
                )
            name = f"{layout}_syn{index:04d}.png"
            Image.fromarray(image).save(output / "images" / name)
            row = {
                "image_path": f"images/{name}",
                "plate_text": text,
                "source": "synthetic",
                "type": layout,
                "sha256": hashlib.sha256((output / "images" / name).read_bytes()).hexdigest(),
            }
            if video_like:
                row["context_margins"] = context_margins
                row.update(degradation)
            rows.append(row)
    write_manifest(output / "synthetic.csv", rows)
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def combine_with_train(train: Path, synthetic_rows: list[dict], output: Path) -> Path:
    """Write train + synthetic as one manifest in `output`, paths relative to that folder."""
    with train.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["image_path"] = os.path.relpath(train.parent / row["image_path"], output)
    path = output / "train-plus-synthetic.csv"
    write_manifest(path, rows + synthetic_rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("custom_alpr/data/ocr/synthetic-v1"))
    parser.add_argument("--per-layout", type=int, help="override every layout count (for previews)")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument(
        "--train", type=Path, default=Path("custom_alpr/data/ocr/kaggle-v9-prepared/train.csv")
    )
    parser.add_argument("--confusable-boost", action="store_true")
    parser.add_argument(
        "--confusion-report",
        type=Path,
        default=Path("custom_alpr/outputs/phase-c-076/confusion-report/report.json"),
    )
    parser.add_argument(
        "--video-like", action="store_true", help="Add loose vehicle context and video degradation"
    )
    parser.add_argument("--motion-blur-probability", type=float, default=0.5)
    parser.add_argument("--h264-probability", type=float, default=0.4)
    parser.add_argument("--view-tilt-probability", type=float, default=0.3)
    parser.add_argument("--preview-grid", type=Path)
    args = parser.parse_args()
    counts = DEFAULT_COUNTS
    if args.per_layout is not None:
        counts = dict.fromkeys(DEFAULT_COUNTS, args.per_layout)
    weights = load_confusable_weights(args.confusion_report) if args.confusable_boost else None
    rows = generate(
        counts,
        args.output,
        args.seed,
        args.train,
        confusable_weights=weights,
        video_like=args.video_like,
        motion_blur_probability=args.motion_blur_probability,
        h264_probability=args.h264_probability,
        view_tilt_probability=args.view_tilt_probability,
    )
    combined = combine_with_train(args.train, rows, args.output)
    if args.video_like:
        preview_path = args.preview_grid or args.output / "preview-grid-64.png"
        write_preview_grid(rows, args.output, preview_path, 64)
        print(f"Preview grid: {preview_path}")
    print(f"Wrote {len(rows)} synthetic plates to {args.output}; train + synthetic: {combined}")


if __name__ == "__main__":
    main()
