"""Degradation must hit the requested height and keep the plate's aspect."""

import numpy as np

from custom_alpr.resolution_sweep import accuracy, degrade


def test_degrade_keeps_aspect_and_rgb():
    image = np.full((80, 360, 3), 200, dtype=np.uint8)
    small = degrade(image, 20)
    assert small.shape == (20, 90, 3)
    assert small.dtype == np.uint8


def test_degrade_never_returns_zero_width():
    assert degrade(np.zeros((400, 10, 3), dtype=np.uint8), 12).shape[1] >= 1


def test_accuracy_splits_layouts():
    report = accuracy([(True, False), (False, False), (True, True)])
    assert report["all"] == dict(n=3, exact=2 / 3)
    assert report["one_line"] == dict(n=2, exact=0.5)
    assert report["two_line"] == dict(n=1, exact=1.0)
