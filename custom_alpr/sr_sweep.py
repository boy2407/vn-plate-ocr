"""Does super-resolving a plate crop before OCR help, and at which crop heights?

Each crop is read three ways through the normal 128x64 path: as is, after a
plain 4x bicubic upscale (control: upscaling without a learned prior), and
after Real-ESRGAN. Two sets:

- natural: every real (``cropped``) Kaggle test crop, bucketed by source height;
- degraded: real crops >= 40 px shrunk to 12-24 px and JPEG-compressed like
  video (``resolution_sweep.degrade``), so the true text is known at small sizes.

A learned upscaler invents strokes; "breaks" counts crops read right as is and
wrong after the upscale, so a gain that also breaks plates is visible.

    python -m custom_alpr.sr_sweep --weights RealESRGAN_x4plus.pth --ocr-checkpoint <E.pt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from custom_alpr.resolution_sweep import MANIFEST, degrade, load_samples

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "custom_alpr/outputs/sr-sweep"
HEIGHT_BUCKETS = ((0, 16), (16, 24), (24, 32), (32, 10_000))  # as confusion_report.py


class ResidualDenseBlock(nn.Module):
    def __init__(self, features: int = 64, growth: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(features, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(features + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(features + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(features + 3 * growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(features + 4 * growth, features, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, features: int, growth: int):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(features, growth)
        self.rdb2 = ResidualDenseBlock(features, growth)
        self.rdb3 = ResidualDenseBlock(features, growth)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN generator; parameter names match the released ``.pth`` files."""

    def __init__(self, scale: int = 4, features: int = 64, blocks: int = 23, growth: int = 32):
        super().__init__()
        self.scale = scale
        channels = 3 * 4 if scale == 2 else 3  # x2 models pixel-unshuffle the input
        self.conv_first = nn.Conv2d(channels, features, 3, 1, 1)
        self.body = nn.Sequential(*(RRDB(features, growth) for _ in range(blocks)))
        self.conv_body = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_hr = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_last = nn.Conv2d(features, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = F.pixel_unshuffle(x, 2) if self.scale == 2 else x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_esrgan(weights: Path, scale: int, device: torch.device) -> RRDBNet:
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model = RRDBNet(scale)
    model.load_state_dict(state.get("params_ema", state.get("params", state)))
    return model.to(device).eval()


@torch.inference_mode()
def esrgan_upscale(model: RRDBNet, image: np.ndarray, device: torch.device) -> np.ndarray:
    """RGB uint8 crop -> upscaled RGB uint8 crop."""
    height, width = image.shape[:2]
    pad_h, pad_w = (-height) % 2, (-width) % 2  # pixel-unshuffle needs even sides
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    output = model(tensor)[..., : height * model.scale, : width * model.scale]
    return output.clamp(0, 1).mul(255).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()


def bicubic_upscale(image: np.ndarray, scale: int = 4) -> np.ndarray:
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def bucket(height: int) -> str:
    low, high = next((low, high) for low, high in HEIGHT_BUCKETS if low <= height < high)
    return f"{low}-{high}px" if high < 10_000 else f">={low}px"


def compare(texts: dict[str, list[str]], truths: list[str], groups: list[str]) -> dict:
    """Exact match per group and method, and fixes/breaks of each method against ``none``."""
    report = {}
    for group in sorted(set(groups), key=lambda name: (len(name), name)):
        index = [i for i, name in enumerate(groups) if name == group]
        base = [texts["none"][i] == truths[i] for i in index]
        report[group] = {"n": len(index)}
        for method, read in texts.items():
            hits = [read[i] == truths[i] for i in index]
            report[group][method] = dict(
                correct=sum(hits),
                exact=round(sum(hits) / len(index), 4),
                fixes=sum(hit and not was for hit, was in zip(hits, base, strict=True)),
                breaks=sum(was and not hit for hit, was in zip(hits, base, strict=True)),
            )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True, help="Real-ESRGAN .pth")
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--heights", type=int, nargs="+", default=[12, 16, 20, 24])
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--ocr-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sheet", type=int, default=40, help="Changed reads drawn per set")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def contact_sheet(rows: list[dict], path: Path) -> None:
    """Crop as is | ESRGAN output, both at 256x128, with truth and the two reads."""
    tiles = []
    for row in rows:
        left = cv2.resize(row["none_image"], (256, 128), interpolation=cv2.INTER_NEAREST)
        right = cv2.resize(row["esrgan_image"], (256, 128), interpolation=cv2.INTER_AREA)
        caption = np.full((24, 512, 3), 255, np.uint8)
        text = f"{row['truth']}  none={row['none']}  sr={row['esrgan']}  h={row['height']}"
        cv2.putText(caption, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(np.vstack([np.hstack([left, right]), caption]))
    if tiles:
        cv2.imwrite(str(path), cv2.cvtColor(np.vstack(tiles), cv2.COLOR_RGB2BGR))


def main() -> None:
    from custom_alpr.preprocessing import prepare_ocr_input  # noqa: PLC0415
    from custom_alpr.video_demo import DEFAULT_OCR, load_ocr, ocr_arrays, resolve_device  # noqa: PLC0415

    args = parse_args()
    device, _ = resolve_device(args.device)
    ocr = load_ocr(args.ocr_checkpoint or DEFAULT_OCR, device)
    esrgan = load_esrgan(args.weights, args.scale, device)
    args.output.mkdir(parents=True, exist_ok=True)

    def read(images: list[np.ndarray]) -> list[str]:
        texts = []
        for start in range(0, len(images), args.batch_size):
            batch = [prepare_ocr_input(image) for image in images[start : start + args.batch_size]]
            texts.extend(text for text, _ in ocr_arrays(batch, ocr, device))
        return texts

    natural = load_samples(args.manifest, 0)
    large = [sample for sample in natural if sample["image"].shape[0] >= 40]
    sets = {
        "natural": [(s["image"], s["text"], bucket(s["image"].shape[0])) for s in natural],
        "degraded": [
            (degrade(s["image"], height, args.jpeg_quality), s["text"], f"h={height}")
            for height in args.heights
            for s in large
        ],
    }
    report = dict(
        checkpoint=str(args.ocr_checkpoint or DEFAULT_OCR),
        weights=str(args.weights),
        scale=args.scale,
        manifest=str(args.manifest),
        degraded_sources=len(large),
    )
    for name, items in sets.items():
        images, truths, groups = (list(column) for column in zip(*items, strict=True))
        upscaled = [esrgan_upscale(esrgan, image, device) for image in images]
        texts = {
            "none": read(images),
            "bicubic": read([bicubic_upscale(image, args.scale) for image in images]),
            "esrgan": read(upscaled),
        }
        report[name] = compare(texts, truths, groups)
        for group, line in report[name].items():
            print(
                f"{name:8s} {group:9s} n={line['n']:4d}  "
                + "  ".join(
                    f"{m}={line[m]['exact']:.3f} (+{line[m]['fixes']}/-{line[m]['breaks']})"
                    for m in texts
                )
            )
        changed = [
            dict(truth=truths[i], none=texts["none"][i], esrgan=texts["esrgan"][i],
                 height=images[i].shape[0], none_image=images[i], esrgan_image=upscaled[i])
            for i in range(len(images))
            if texts["none"][i] != texts["esrgan"][i]
        ]  # fmt: skip
        contact_sheet(changed[: args.sheet], args.output / f"changed-{name}.png")
    path = args.output / "sr-sweep.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
