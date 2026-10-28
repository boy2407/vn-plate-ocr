"""Measure PlateOCR exact accuracy when test crops get Gaussian or salt-and-pepper noise.

Noise is added to the resized 64x128 model input (pixel range 0-1) with a fixed
seed, so every checkpoint sees exactly the same corrupted images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from custom_alpr.analyze_custom_ocr import choose_device, edit_distance, greedy_with_confidence
from custom_alpr.custom_ocr import build_model
from custom_alpr.custom_ocr_data import PlateOCRDataset, plate_ocr_collate

DEFAULT_DATA = Path("custom_alpr/data/ocr/kaggle-v9-prepared")
GAUSSIAN_SIGMAS = (5, 10, 20, 30)  # standard deviation on the 0-255 scale
SALT_PEPPER_AMOUNTS = (0.01, 0.03, 0.05, 0.10)  # fraction of pixels set to black or white


def add_gaussian(images: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(images.shape, generator=generator) * (sigma / 255)
    return (images + noise).clamp(0, 1)


def add_salt_pepper(
    images: torch.Tensor, amount: float, generator: torch.Generator
) -> torch.Tensor:
    batch, _, height, width = images.shape
    draw = torch.rand((batch, 1, height, width), generator=generator)
    noisy = images.clone()
    noisy = noisy.masked_fill(draw < amount / 2, 0.0)
    return noisy.masked_fill((draw >= amount / 2) & (draw < amount), 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA / "test.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda[:index]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    dataset = PlateOCRDataset(args.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=plate_ocr_collate)
    conditions = [("clean", 0.0)]
    conditions += [("gaussian", sigma) for sigma in GAUSSIAN_SIGMAS]
    conditions += [("salt_pepper", amount) for amount in SALT_PEPPER_AMOUNTS]

    models = {}
    for path in args.checkpoint:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        models[path.parent.name] = model.to(device).eval()

    results = {name: [] for name in models}
    for kind, level in conditions:
        generator = torch.Generator().manual_seed(args.seed)
        counts = {name: [0, 0, 0] for name in models}  # exact, character errors, characters
        for clean_images, _, _, references in loader:
            images = clean_images
            if kind == "gaussian":
                images = add_gaussian(clean_images, level, generator)
            elif kind == "salt_pepper":
                images = add_salt_pepper(clean_images, level, generator)
            images = images.to(device)
            for name, model in models.items():
                predictions, _ = greedy_with_confidence(model, images)
                for prediction, reference in zip(predictions, references, strict=True):
                    counts[name][0] += prediction == reference
                    counts[name][1] += edit_distance(reference, prediction)
                    counts[name][2] += len(reference)
        for name, (exact, errors, characters) in counts.items():
            row = {
                "noise": kind,
                "level": level,
                "exact_accuracy": exact / len(dataset),
                "cer": errors / characters,
            }
            results[name].append(row)
            print(
                f"{name:28s} {kind:11s} {level:<5g} exact={row['exact_accuracy']:.4f} "
                f"cer={row['cer']:.4f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"manifest": str(args.manifest), "seed": args.seed, "results": results}, indent=2
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
