"""Generate reproducible image/box previews and audit the corruption budget."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from custom_alpr.yolo_vast.corruptions import EFFECTS, apply_budget, apply_effect
from custom_alpr.yolo_vast.data import IMAGE_EXTENSIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "previews"
    out.mkdir(exist_ok=True)
    images = sorted(
        p for p in (root / "data/images/train").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    # Show several training images, not a hand-picked successful corruption.
    rng = np.random.default_rng(42)
    counts = Counter()
    for _ in range(5000):
        _, effects = apply_budget(np.full((32, 32, 3), 127, np.uint8), rng)
        counts[f"effects_{len(effects)}"] += 1
        counts.update(effects)
    (out / "budget-5000.json").write_text(json.dumps(dict(counts), indent=2))
    font = ImageFont.load_default(size=14)
    for sample, path in enumerate(rng.choice(images, size=min(4, len(images)), replace=False)):
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        boxes = []
        label = root / "data/labels/train" / (path.stem + ".txt")
        for line in label.read_text().splitlines():
            _, x, y, bw, bh = map(float, line.split())
            boxes.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
        boxes = np.asarray(boxes).reshape(-1, 4)
        canvas = Image.new("RGB", (4 * 320, 4 * 250), "#edf3ef")
        draw = ImageDraw.Draw(canvas)
        for i, effect in enumerate(("original", *EFFECTS)):
            pixels = (
                image
                if effect == "original"
                else apply_effect(image, effect, np.random.default_rng(42), boxes=boxes)
            )
            tile = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
            pen = ImageDraw.Draw(tile)
            for box in boxes:
                pen.rectangle(tuple(box), outline="red", width=3)
            tile = ImageOps.contain(tile, (308, 214))
            x, y = i % 4 * 320, i // 4 * 250
            canvas.paste(tile, (x + 6, y + 28))
            draw.text((x + 8, y + 5), effect, font=font, fill="black")
        canvas.save(out / f"corruptions-{sample}.jpg")
    print(out)


if __name__ == "__main__":
    main()
