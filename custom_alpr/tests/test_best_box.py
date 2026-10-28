"""Ensure extra YOLO detections never reach OCR in the demo."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from custom_alpr.video_demo import select_best_box


def box(coords, confidence):
    return SimpleNamespace(xyxy=torch.tensor([coords]), conf=torch.tensor(confidence))


def test_best_valid_box_is_selected_even_when_results_are_unsorted():
    boxes = [
        box([0, 0, 2, 2], 0.4),
        box([2, 1, 7, 5], 0.9),
        box([0, 0, 3, 3], 0.7),
        box([20, 20, 30, 30], 0.99),
    ]
    assert select_best_box(boxes, 10, 8) == pytest.approx((2, 1, 7, 5, 0.9))


@pytest.mark.parametrize("boxes", [None, [], [box([20, 20, 30, 30], 0.9)]])
def test_no_valid_box_returns_none(boxes):
    assert select_best_box(boxes, 10, 8) is None


def test_web_passes_only_best_crop_to_ocr_and_skips_empty(monkeypatch):
    web_app = pytest.importorskip("custom_alpr.web_app")
    engine = web_app.InferenceEngine.__new__(web_app.InferenceEngine)
    engine.device = engine.yolo_device = "cpu"
    engine.recognizer = object()
    boxes = [box([0, 0, 2, 2], 0.4), box([2, 1, 7, 5], 0.9)]
    engine.detector = SimpleNamespace(
        predict=lambda *_args, **_kwargs: [SimpleNamespace(boxes=boxes)]
    )
    received = []

    def recognize(crops, *_args, **_kwargs):
        received.extend(crops)
        return [("30A12345", 0.95)]

    monkeypatch.setattr(web_app, "ocr_crops", recognize)
    frame = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    results = engine.recognize(frame, 0.25)
    assert len(results) == len(received) == 1
    assert results[0]["box"] == [2, 1, 7, 5]
    np.testing.assert_array_equal(received[0], frame[1:5, 2:7, ::-1])
    boxes.clear()
    assert engine.recognize(frame, 0.25) == []
    assert len(received) == 1
