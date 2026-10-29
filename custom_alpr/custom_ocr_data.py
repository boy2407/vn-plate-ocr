"""Dataset and batching helpers for the project-owned PlateOCR2D model."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from custom_alpr.augment import (
    random_downscale,
    random_gaussian_noise,
    random_blue_plate,
    random_compression,
    random_glare,
    random_motion_blur,
    random_photometric,
    random_rotate_perspective,
    looks_straight,
    random_view_tilt,
    random_salt_pepper,
)
from custom_alpr.custom_ocr import PAD_INDEX, encode_target

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 64


class PlateOCRDataset(Dataset):
    """Read one prepared manifest; apply training augmentations only when enabled."""

    def __init__(
        self,
        manifest: Path,
        *,
        augment: bool = False,
        noise: bool = False,
        blue_plate: bool = False,
        rotate: bool = False,
        strong_downscale: bool = False,
        view_tilt: bool = False,
        compression: bool = False,
        seed: int = 0,
    ):
        self.manifest = manifest
        self.augment = augment
        self.noise = noise
        self.blue_plate = blue_plate
        self.rotate = rotate
        self.strong_downscale = strong_downscale
        self.view_tilt = view_tilt
        self.compression = compression
        self.rng = np.random.default_rng(seed)
        self.samples: list[tuple[Path, str]] = []
        # Renders and synthetic plates are drawn straight; real crops are measured once.
        self.straight: list[bool | None] = []
        with manifest.open(encoding="utf-8", newline="") as manifest_file:
            reader = csv.DictReader(manifest_file)
            if not {"image_path", "plate_text"} <= set(reader.fieldnames or []):
                raise ValueError(f"{manifest}: expected image_path,plate_text columns")
            for line, row in enumerate(reader, 2):
                image_path = manifest.parent / (row.get("image_path") or "")
                text = row.get("plate_text") or ""
                if not image_path.is_file():
                    raise ValueError(f"{manifest}:{line}: missing image {image_path}")
                try:
                    encode_target(text)
                except ValueError as error:
                    raise ValueError(f"{manifest}:{line}: invalid plate_text {text!r}") from error
                self.samples.append((image_path, text))
                clean = row.get("source") in {"generated", "synthetic"}
                self.straight.append(True if clean else None)
        if not self.samples:
            raise ValueError(f"{manifest}: no labeled samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        image_path, text = self.samples[index]
        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB").resize(
                    (IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR
                )
                pixels = np.asarray(image, dtype=np.uint8).copy()
        except OSError as error:
            raise ValueError(f"Cannot decode image {image_path}") from error
        if self.augment:
            if self.view_tilt:  # replaces the milder rotation of run D; straight plates only
                if self.straight[index] is None:
                    self.straight[index] = looks_straight(pixels)
                if self.straight[index]:
                    pixels = random_view_tilt(pixels, self.rng)
            elif self.rotate:
                pixels = random_rotate_perspective(pixels, self.rng)
            if self.strong_downscale:  # match real small crops: 8-24 px high, half the time
                pixels = random_downscale(
                    pixels, self.rng, probability=0.5, min_height=8, max_height=24
                )
            else:
                pixels = random_downscale(pixels, self.rng)
            pixels = random_photometric(pixels, self.rng)
            if self.blue_plate:
                pixels = random_blue_plate(pixels, self.rng)
            pixels = random_glare(pixels, self.rng)
            pixels = random_motion_blur(pixels, self.rng)
            if self.noise:
                pixels = random_gaussian_noise(pixels, self.rng)
                pixels = random_salt_pepper(pixels, self.rng)
            if self.compression:  # last: video is compressed after the sensor adds noise
                pixels = random_compression(pixels, self.rng)
        image_tensor = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0)
        target = torch.tensor(encode_target(text), dtype=torch.long)
        return image_tensor, target, text


def rare_letter_repeat_factors(
    texts: list[str], rare_threshold: int = 200
) -> tuple[list[float], set[str]]:
    """Repeat Factor Sampling (Gupta et al., LVIS 2019) over plate letters.

    For each letter, ``count`` is the number of plates containing it. A letter is
    rare when ``count < rare_threshold`` and gets ``sqrt(rare_threshold / count)``;
    a plate's factor is the largest factor of its letters, at least 1.
    """
    counts: dict[str, int] = {}
    for text in texts:
        for character in set(text):
            if not character.isdigit():
                counts[character] = counts.get(character, 0) + 1
    rare = {c for c, n in counts.items() if n < rare_threshold}
    factors = [
        max([math.sqrt(rare_threshold / counts[c]) for c in set(text) if c in rare], default=1.0)
        for text in texts
    ]
    return factors, rare


class RepeatFactorSampler(Sampler[int]):
    """Each epoch, show every plate at least once and rare-letter plates more often.

    A plate with factor 2.3 appears 2 times plus a third time with probability 0.3,
    so the epoch grows to about ``sum(factors)`` crops; order is reshuffled.
    """

    def __init__(self, factors: list[float], generator: torch.Generator | None = None):
        if any(f < 1 for f in factors):
            raise ValueError("repeat factors must be at least 1")
        self.factors = torch.tensor(factors, dtype=torch.float64)
        self.generator = generator

    def __len__(self) -> int:
        return round(float(self.factors.sum()))

    def __iter__(self):
        whole = self.factors.floor()
        extra = torch.rand(len(self.factors), generator=self.generator, dtype=torch.float64)
        repeats = (whole + (extra < self.factors - whole)).long()
        indices = torch.arange(len(self.factors)).repeat_interleave(repeats)
        yield from indices[torch.randperm(len(indices), generator=self.generator)].tolist()


def plate_ocr_collate(
    samples: list[tuple[Tensor, Tensor, str]],
) -> tuple[Tensor, Tensor, Tensor, list[str]]:
    """Stack images and right-pad encoded BOS/text/EOS targets with PAD_INDEX."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    images, targets, texts = zip(*samples, strict=True)
    max_length = max(target.shape[0] for target in targets)
    target_ids = torch.full((len(samples), max_length), PAD_INDEX, dtype=torch.long)
    for row, target in enumerate(targets):
        target_ids[row, : target.shape[0]] = target
    decoder_input_ids = target_ids[:, :-1]
    return torch.stack(images), decoder_input_ids, target_ids, list(texts)
