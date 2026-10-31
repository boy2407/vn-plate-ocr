"""Upscalers keep crop shape, odd sides included; fixes/breaks count against the as-is read."""

import numpy as np
import torch

from custom_alpr.sr_sweep import RRDBNet, bicubic_upscale, bucket, compare, esrgan_upscale


def test_esrgan_upscale_shapes_for_odd_crops():
    for scale in (2, 4):
        model = RRDBNet(scale, features=8, blocks=1, growth=4).eval()
        out = esrgan_upscale(model, np.zeros((15, 41, 3), dtype=np.uint8), torch.device("cpu"))
        assert out.shape == (15 * scale, 41 * scale, 3)
        assert out.dtype == np.uint8


def test_bicubic_upscale_shape():
    assert bicubic_upscale(np.zeros((10, 30, 3), dtype=np.uint8)).shape == (40, 120, 3)


def test_bucket_edges():
    assert bucket(15) == "0-16px"
    assert bucket(16) == "16-24px"
    assert bucket(32) == ">=32px"


def test_compare_counts_fixes_and_breaks():
    texts = {"none": ["A", "X", "C"], "esrgan": ["A", "B", "Y"]}
    report = compare(texts, ["A", "B", "C"], ["g", "g", "g"])
    assert report["g"]["n"] == 3
    assert report["g"]["esrgan"] == dict(correct=2, exact=round(2 / 3, 4), fixes=1, breaks=1)
    assert report["g"]["none"]["fixes"] == report["g"]["none"]["breaks"] == 0
