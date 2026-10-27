"""Data integrity and inference boundary tests without model downloads."""

import csv
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from custom_alpr.data import parse_box, prepare, read_annotations
from custom_alpr.pipeline import PlateReader, edit_distance


@pytest.mark.parametrize(
    "label",
    [
        "0 0.5 0.5 0.4 0.2",
        "1 0.3 0.4 0.7 0.4 0.7 0.6 0.3 0.6",
    ],
)
def test_box_and_polygon_conversion(label):
    assert parse_box(label) == pytest.approx((0.3, 0.4, 0.7, 0.6))


@pytest.mark.parametrize(
    "label",
    [
        "0 0.5 0.5 -0.1 0.2",
        "0 nan 0.5 0.1 0.2",
        "0 0.1 0.2 0.3",
        "0 -0.1 0.2 0.5 0.2 0.5 0.8",
        "0 0.5 0.5 0 0.2",
    ],
)
def test_invalid_coordinates_rejected(label):
    with pytest.raises(ValueError):
        parse_box(label)


def test_prepare_deduplicates_and_preserves_blank_transcriptions(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
        for name, value in (("duplicate", 100), (split, 40 if split == "train" else 50)):
            image = np.full((50, 100, 3), value, np.uint8)
            cv2.imwrite(str(source / "images" / split / f"{name}.png"), image)
            (source / "labels" / split / f"{name}.txt").write_text("1 0.5 0.5 0.4 0.4\n")
    report = prepare(source, output)
    assert report["splits"]["train"] == {"images": 2, "crops": 2}
    assert report["splits"]["val"] == {"images": 1, "crops": 1}
    assert len(report["duplicates_removed"]) == 1
    crop = cv2.imread(str(output / "ocr" / "train" / "images" / "train_000.png"))
    assert crop.shape == (20, 40, 3)
    csv_path = output / "ocr" / "train" / "annotations.csv"
    with csv_path.open() as file:
        assert all(row["plate_text"] == "" for row in csv.DictReader(file))
    with pytest.raises(ValueError, match="label must"):
        read_annotations(csv_path)
    with pytest.raises(ValueError, match="already exists"):
        prepare(source, output)


def test_annotations_resolve_relative_to_csv(tmp_path):
    image = tmp_path / "plate.png"
    cv2.imwrite(str(image), np.zeros((30, 80, 3), np.uint8))
    path = tmp_path / "annotations.csv"
    path.write_text("image_path,plate_text\nplate.png,30A12345\n")
    assert read_annotations(path) == [(image, "30A12345")]


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def test_pipeline_clips_boxes_skips_empty_and_converts_bgr():
    reader = PlateReader.__new__(PlateReader)
    reader.device = "cpu"
    boxes = [
        SimpleNamespace(xyxy=[FakeTensor(coords)], conf=np.array(0.9))
        for coords in ((-5, -4, 20, 12), (50, 50, 60, 60))
    ]
    reader.detector = SimpleNamespace(predict=lambda **_kwargs: [SimpleNamespace(boxes=boxes)])
    crops = []

    def recognize(crop, **_kwargs):
        crops.append(crop)
        return SimpleNamespace(plate="30A12345", char_probs=np.array([0.95]))

    reader.ocr = SimpleNamespace(config=SimpleNamespace(image_color_mode="rgb"), run_one=recognize)
    image = np.full((10, 15, 3), (10, 20, 30), np.uint8)
    results = reader.predict(image)
    assert len(results) == 1
    assert results[0]["box"] == [0, 0, 15, 10]
    assert results[0]["text"] == "30A12345"
    assert crops[0][0, 0].tolist() == [30, 20, 10]
    with pytest.raises(ValueError, match="nonempty"):
        reader.predict(None)


@pytest.mark.parametrize(
    ("expected", "actual", "distance"),
    [
        ("30A12345", "30A12345", 0),
        ("30A12345", "30A1234", 1),
        ("30A12345", "30A123456", 1),
        ("30A12345", "30B12345", 1),
    ],
)
def test_character_error_counts_insertions_and_deletions(expected, actual, distance):
    assert edit_distance(expected, actual) == distance
