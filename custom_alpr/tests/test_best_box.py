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


def fake_engine(web_app, boxes, received):
    engine = web_app.InferenceEngine.__new__(web_app.InferenceEngine)
    engine.device = engine.yolo_device = "cpu"
    engine.recognizer = object()
    engine.detector = SimpleNamespace(
        predict=lambda *_args, **_kwargs: [SimpleNamespace(boxes=boxes)]
    )

    def recognize(arrays, *_args, **_kwargs):
        received.extend(arrays)
        return [("30A12345", 0.95)] * len(arrays)

    return engine, recognize


def test_web_reads_every_valid_box_in_one_ocr_call_and_skips_empty(monkeypatch):
    web_app = pytest.importorskip("custom_alpr.web_app")
    boxes = [box([0, 0, 2, 2], 0.4), box([2, 1, 7, 5], 0.9), box([20, 20, 30, 30], 0.99)]
    received = []
    engine, recognize = fake_engine(web_app, boxes, received)
    monkeypatch.setattr(web_app, "ocr_arrays", recognize)
    frame = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    results, too_small = engine.recognize(frame, 0.25)
    assert too_small == 0
    assert [item["box"] for item in results] == [[2, 1, 7, 5], [0, 0, 2, 2]]
    assert len(received) == 2
    boxes.clear()
    assert engine.recognize(frame, 0.25) == ([], 0)


def test_crops_below_min_height_are_counted_but_never_read(monkeypatch):
    web_app = pytest.importorskip("custom_alpr.web_app")
    boxes = [box([0, 0, 8, 3], 0.9), box([0, 2, 9, 8], 0.5)]
    received = []
    engine, recognize = fake_engine(web_app, boxes, received)
    monkeypatch.setattr(web_app, "ocr_arrays", recognize)
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    results, too_small = engine.recognize(frame, 0.25, min_height=4)
    assert too_small == 1
    assert [item["box"] for item in results] == [[0, 2, 9, 8]]
    assert len(received) == 1


def test_tensorrt_engine_uses_its_fixed_input_whatever_size_is_requested(monkeypatch):
    web_app = pytest.importorskip("custom_alpr.web_app")
    seen = []
    engine = web_app.InferenceEngine.__new__(web_app.InferenceEngine)
    engine.device = engine.yolo_device = "cpu"
    engine.fixed_size = (384, 640)
    engine.detector = SimpleNamespace(
        predict=lambda *_a, **kw: seen.append(kw["imgsz"]) or [SimpleNamespace(boxes=[])]
    )
    engine.recognize(np.zeros((8, 10, 3), dtype=np.uint8), 0.25, image_size=320)
    assert seen == [(384, 640)]
