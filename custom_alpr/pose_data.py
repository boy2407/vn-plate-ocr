"""Convert source plate quadrilaterals to ordered YOLO pose labels."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

FLIP_IDX = [1, 0, 3, 2]


def order_corners(points):
    """Clockwise from TL: start of the edge with the lowest midpoint y."""
    if len(points) != 4 or len(set(map(tuple, points))) != 4:
        raise ValueError("Expected four distinct corners")
    if not all(math.isfinite(v) for p in points for v in p):
        raise ValueError("Nonfinite corner")
    cx = sum(p[0] for p in points) / 4
    cy = sum(p[1] for p in points) / 4
    ordered = sorted(map(tuple, points), key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    # The upper edge is invariant under horizontal reflection. A min(x+y)
    # vertex alone can switch diagonals on strongly skewed plates after flipping.
    start = min(
        range(4),
        key=lambda i: (
            ordered[i][1] + ordered[(i + 1) % 4][1],
            -abs(ordered[(i + 1) % 4][0] - ordered[i][0]),
        ),
    )
    ordered = ordered[start:] + ordered[:start]
    for i in range(4):
        a, b, c = ordered[i], ordered[(i + 1) % 4], ordered[(i + 2) % 4]
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) <= 1e-12:
            raise ValueError("Nonconvex or degenerate quadrilateral")
    return ordered


def pose_line(line):
    fields = line.split()
    if len(fields) != 9:
        return None
    values = list(map(float, fields[1:]))
    points = order_corners(list(zip(values[::2], values[1::2], strict=True)))
    if any(v < -1e-6 or v > 1 + 1e-6 for p in points for v in p):
        raise ValueError("Corner outside image")
    xs, ys = zip(*points, strict=True)
    x, y, a, b = min(xs), min(ys), max(xs), max(ys)
    result = [0, (x + a) / 2, (y + b) / 2, a - x, b - y]
    for px, py in points:
        result.extend([px, py, 2])
    return " ".join(f"{v:.8f}" for v in result)


def main():  # noqa: PLR0912, PLR0915
    import yaml  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("custom_alpr/data/raw/kaggle-licenseplates/licenseplates.zip"),
    )
    parser.add_argument("--detection", type=Path, default=Path("custom_alpr/data/detection"))
    parser.add_argument("--output", type=Path, default=Path("custom_alpr/data/pose-073"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists: choose a new directory to avoid overwriting")
    report = dict(
        raw=Counter(),
        splits={},
        skipped_polygons=[],
        excluded_no_pose_images=[],
        duplicates_removed=json.loads(Path("custom_alpr/data/report.json").read_text())[
            "duplicates_removed"
        ],
    )
    previews = []
    with zipfile.ZipFile(args.archive) as archive:
        for name in archive.namelist():
            if name.startswith("labels/") and name.endswith(".txt"):
                for line in archive.read(name).decode().splitlines():
                    report["raw"][
                        "four_points" if len(line.split()) == 9 else "non_four_points"
                    ] += 1
                    if len(line.split()) == 9:
                        values = list(map(float, line.split()[1:]))
                        rectangle = len(set(values[::2])) == 2 and len(set(values[1::2])) == 2
                        report["raw"][
                            "axis_rectangles" if rectangle else "true_quadrilaterals"
                        ] += 1
        for split in ["train", "val"]:
            images = args.output / "images" / split
            labels = args.output / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            count = Counter()
            for path in sorted((args.detection / "images" / split).iterdir()):
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                lines = archive.read(f"labels/{split}/{path.stem}.txt").decode().splitlines()
                converted = []
                for index, line in enumerate(lines):
                    value = pose_line(line)
                    if value is None:
                        count["skipped_polygons"] += 1
                        report["skipped_polygons"].append(
                            dict(
                                split=split,
                                image=path.name,
                                index=index,
                                vertices=(len(line.split()) - 1) // 2,
                            )
                        )
                        continue
                    converted.append(value)
                    count["poses"] += 1
                    original = list(map(float, line.split()[1:]))
                    xs, ys = original[::2], original[1::2]
                    count[
                        "axis_rectangles"
                        if len(set(xs)) == 2 and len(set(ys)) == 2
                        else "true_quadrilaterals"
                    ] += 1
                if not converted:
                    report["excluded_no_pose_images"].append(f"{split}/{path.name}")
                    continue
                shutil.copy2(path, images / path.name)
                (labels / f"{path.stem}.txt").write_text("\n".join(converted) + "\n")
                count["images"] += 1
                previews.append((images / path.name, converted))
            report["splits"][split] = dict(count)
    (args.output / "dataset.yaml").write_text(
        yaml.safe_dump(
            dict(
                path=str(args.output.resolve()),
                train="images/train",
                val="images/val",
                names={0: "license_plate"},
                kpt_shape=[4, 3],
                flip_idx=FLIP_IDX,
            )
        )
    )
    output = Path("custom_alpr/outputs/pose-073")
    output.mkdir(parents=True, exist_ok=True)
    report["ordering"] = "atan2 clockwise; first endpoint of upper edge (minimum midpoint y)"
    (output / "conversion-report.json").write_text(json.dumps(report, indent=2))
    chosen = random.Random(73).sample(previews, 30)
    sheet = Image.new("RGB", (1200, 250 * 10), "white")
    draw = ImageDraw.Draw(sheet)
    colors = ["red", "lime", "blue", "yellow"]
    for i, (path, lines) in enumerate(chosen):
        im = Image.open(path).convert("RGB")
        width, height = im.size
        d = ImageDraw.Draw(im)
        for line in lines:
            v = list(map(float, line.split()))
            points = [(v[5 + 3 * j] * width, v[6 + 3 * j] * height) for j in range(4)]
            d.line([*points, points[0]], fill="cyan", width=2)
            for j, (x, y) in enumerate(points):
                d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[j])
                d.text(
                    (x + 5, y - 12),
                    ["0 TL", "1 TR", "2 BR", "3 BL"][j],
                    fill=colors[j],
                    stroke_width=1,
                    stroke_fill="black",
                )
        im.thumbnail((395, 225))
        ox = (i % 3) * 400
        oy = (i // 3) * 250
        sheet.paste(im, (ox, oy + 22))
        draw.text((ox, oy), path.name, fill="black")
    sheet.save(output / "labels-contact.png")
    print(json.dumps(report["splits"], indent=2))


if __name__ == "__main__":
    main()
