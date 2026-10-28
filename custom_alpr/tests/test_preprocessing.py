"""Check input integrity and preview/inference agreement for optional filtering."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from custom_alpr.preprocessing import crop_quality, padded_box, prepare_ocr_input, rectify_plate


@pytest.mark.parametrize("shape", [(1, 1, 3), (19, 47, 3), (64, 128, 3), (180, 400, 3)])
def test_disabled_preprocessing_matches_training_resize_and_preserves_source(shape):
    crop = np.random.default_rng(12).integers(0, 256, shape, dtype=np.uint8)
    original = crop.copy()
    expected = np.asarray(Image.fromarray(crop).resize((128, 64), Image.Resampling.BILINEAR))
    actual = prepare_ocr_input(crop)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(crop, original)


def test_denoise_reduces_noise_on_a_flat_known_signal():
    noise = np.random.default_rng(4).normal(0, 6, (64, 128, 3))
    crop = np.clip(128 + noise, 0, 255).astype(np.uint8)
    filtered = prepare_ocr_input(crop, denoise=True)
    assert np.mean((filtered.astype(float) - 128) ** 2) < np.mean((crop.astype(float) - 128) ** 2)


def test_sharpen_strengthens_a_blurred_edge_without_overflow():
    crop = np.full((64, 128, 3), 90, np.uint8)
    crop[:, 64:] = 160
    crop = cv2.GaussianBlur(crop, (0, 0), sigmaX=1)
    enhanced = prepare_ocr_input(crop, sharpen=True)
    assert np.diff(enhanced.astype(float), axis=1).max() > np.diff(crop.astype(float), axis=1).max()
    assert enhanced.dtype == np.uint8


@pytest.mark.parametrize("denoise,sharpen", [(True, False), (False, True), (True, True)])
def test_filters_preserve_a_constant_image_without_color_shift(denoise, sharpen):
    crop = np.full((12, 30, 3), (15, 128, 240), dtype=np.uint8)
    filtered = prepare_ocr_input(crop, denoise=denoise, sharpen=sharpen)
    assert filtered.shape == (64, 128, 3)
    assert filtered.dtype == np.uint8
    np.testing.assert_array_equal(filtered, np.full((64, 128, 3), (15, 128, 240), dtype=np.uint8))


@pytest.mark.parametrize(
    "crop",
    [np.zeros((0, 10, 3), np.uint8), np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), float)],
)
def test_invalid_inputs_fail_before_inference(crop):
    with pytest.raises(ValueError, match="RGB uint8"):
        prepare_ocr_input(crop, denoise=True, sharpen=True)


@pytest.mark.parametrize(
    "denoise,sharpen", [(False, False), (True, False), (False, True), (True, True)]
)
def test_saved_ocr_input_matches_pixels_passed_to_model(tmp_path, denoise, sharpen):
    web_app = pytest.importorskip("custom_alpr.web_app")
    received = []

    def recognize(images):
        received.append(images.cpu())
        return ["30A12345"], [0.9]

    engine = web_app.InferenceEngine.__new__(web_app.InferenceEngine)
    engine.device = engine.yolo_device = "cpu"
    engine.recognizer = SimpleNamespace(generate_with_confidence=recognize)
    box = SimpleNamespace(xyxy=torch.tensor([[2, 3, 49, 29]]), conf=torch.tensor(0.9))
    engine.detector = SimpleNamespace(
        predict=lambda *_args, **_kwargs: [SimpleNamespace(boxes=[box])]
    )
    frame = np.random.default_rng(4).integers(0, 256, (35, 55, 3), dtype=np.uint8)
    results = engine.recognize(frame, 0.25, denoise=denoise, sharpen=sharpen)
    web_app.store_crops(frame, results, tmp_path, 0, 25)
    item = results[0]
    saved = cv2.cvtColor(cv2.imread(str(tmp_path / item["ocr_input"])), cv2.COLOR_BGR2RGB)
    pixels = received[0][0].permute(1, 2, 0).mul(255).round().numpy().astype(np.uint8)
    np.testing.assert_array_equal(saved, pixels)
    assert item["preprocessing"] == {"denoise": denoise, "sharpen": sharpen, "rectify": False}
    original = cv2.imread(str(tmp_path / item["crop"]))
    np.testing.assert_array_equal(original, frame[3:29, 2:49])


@pytest.mark.parametrize("form", [{"denoise": "false"}, {"sharpen": "yes"}, {"confidence": "NaN"}])
def test_invalid_preprocessing_settings_are_not_silently_enabled(form):
    web_app = pytest.importorskip("custom_alpr.web_app")
    with pytest.raises(ValueError):
        web_app.inference_settings(form)


def skewed_plate():
    image = np.full((110, 240, 3), 55, np.uint8)
    corners = np.int32([[12, 23], [219, 7], [229, 89], [20, 102]])
    cv2.fillConvexPoly(image, corners, (220, 220, 220))
    cv2.polylines(image, [corners], True, (5, 5, 5), 2)
    cv2.putText(image, "30A12345", (32, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)
    return image


def test_padding_clamps_to_original_frame_without_changing_detector_box():
    box = [1, 2, 101, 52]
    assert padded_box(box, 105, 55) == [0, 0, 105, 54]
    assert box == [1, 2, 101, 52]


def test_visible_skewed_boundary_is_rectified_and_source_preserved():
    source = skewed_plate()
    original = source.copy()
    rectified, corners = rectify_plate(source)
    assert corners is not None
    assert len(corners) == 4
    assert rectified.dtype == np.uint8 and min(rectified.shape[:2]) >= 16
    np.testing.assert_array_equal(source, original)
    assert rectified.shape != source.shape


@pytest.mark.parametrize("shape", [(8, 30, 3), (64, 128, 3), (16, 20000, 3)])
def test_no_visible_boundary_preserves_input(shape):
    source = np.full(shape, 100, np.uint8)
    actual, corners = rectify_plate(source)
    assert corners is None
    np.testing.assert_array_equal(actual, source)


def test_padding_and_rectification_preview_matches_actual_ocr_tensor(tmp_path):
    web = pytest.importorskip("custom_alpr.web_app")
    received = []

    def recognize(images):
        received.append(images.cpu())
        return ["30A12345"], [0.9]

    engine = web.InferenceEngine.__new__(web.InferenceEngine)
    engine.device = engine.yolo_device = "cpu"
    engine.recognizer = SimpleNamespace(generate_with_confidence=recognize)
    box = SimpleNamespace(xyxy=torch.tensor([[8, 4, 232, 106]]), conf=torch.tensor(0.9))
    engine.detector = SimpleNamespace(predict=lambda *_a, **_kw: [SimpleNamespace(boxes=[box])])
    frame = cv2.cvtColor(skewed_plate(), cv2.COLOR_RGB2BGR)
    results = engine.recognize(frame, 0.4, padding=True, rectify=True)
    assert results[0]["box"] == [8, 4, 232, 106]
    assert results[0]["crop_box"] == [0, 0, 240, 110]
    assert results[0]["rectified"]
    web.store_crops(frame, results, tmp_path, 0, 25)
    item = results[0]
    saved = cv2.cvtColor(cv2.imread(str(tmp_path / item["ocr_input"])), cv2.COLOR_BGR2RGB)
    actual = received[0][0].permute(1, 2, 0).mul(255).round().numpy().astype(np.uint8)
    np.testing.assert_array_equal(saved, actual)
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / item["crop"])), frame)


def test_quality_prefers_sharp_text_over_blurred_same_crop():
    crop = skewed_plate()
    assert crop_quality(crop)["score"] > crop_quality(cv2.GaussianBlur(crop, (15, 15), 4))["score"]
