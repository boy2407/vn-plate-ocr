"""Validate native YOLOv9 label/color integrity and persistent-worker controls."""

from types import SimpleNamespace

import numpy as np
import torch

from custom_alpr import yolo9_vast


def test_native_corruptions_use_correct_pixel_boxes_without_changing_labels(tmp_path, monkeypatch):
    seen = []

    def corrupt(image, _rng, boxes):
        seen.append((image.copy(), boxes.copy()))
        return np.full_like(image, (10, 20, 30)), ["gaussian"]

    monkeypatch.setattr(yolo9_vast, "apply_budget", corrupt)
    image = (
        torch.arange(3 * 20 * 40, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(3, 20, 40)
    )
    original = image.clone()
    labels = torch.tensor([[0, 0, 0.5, 0.5, 0.5, 0.4]])
    labels_before = labels.clone()
    result = yolo9_vast.NativeQuality(tmp_path)(image, labels)
    np.testing.assert_array_equal(seen[0][0], image.numpy().transpose(1, 2, 0)[..., ::-1])
    np.testing.assert_allclose(seen[0][1], [[10, 6, 30, 14]], atol=1e-5)
    assert result[:, 0, 0].tolist() == [30, 20, 10]
    assert torch.equal(original, image) and torch.equal(labels_before, labels)


def test_native_quality_is_deterministic_and_uses_no_global_numpy_rng(tmp_path):
    image = torch.full((3, 32, 64), 128, dtype=torch.uint8)
    labels = torch.zeros((0, 6))
    torch.manual_seed(42)
    a, b = yolo9_vast.NativeQuality(tmp_path / "a"), yolo9_vast.NativeQuality(tmp_path / "b")
    np.random.seed(55)
    expected = np.random.random()
    np.random.seed(55)
    for _ in range(12):
        assert torch.equal(a(image, labels), b(image, labels))
    assert np.random.random() == expected


def test_close_mosaic_reads_shared_epoch_and_validation_is_untouched(tmp_path):
    class Dataset:
        def __init__(self, augment):
            self.augment, self.mosaic = augment, augment
            self.image = torch.full((3, 32, 64), 128, dtype=torch.uint8)
            self.labels = torch.tensor([[0, 0, 0.5, 0.5, 0.4, 0.4]])
            self.mosaic_observed = []

        def __getitem__(self, _index):
            self.mosaic_observed.append(self.mosaic)
            return self.image, self.labels, "sample.jpg", None

    epoch = SimpleNamespace(value=19)
    yolo9_vast.install_quality(Dataset, epoch, epochs=30, close_mosaic=10, folder=tmp_path)
    train, val = Dataset(True), Dataset(False)
    train[0]
    epoch.value = 20
    train[0]
    validation = val[0]
    assert train.mosaic_observed == [True, False]
    assert train._plate_quality.counts["mosaic_closed_images"] == 1
    assert validation[0] is val.image and validation[1] is val.labels
    assert not hasattr(val, "_plate_quality")


def test_export_contract_keeps_xyxy_confidence_and_suppresses_duplicates():
    from custom_alpr.yolo9_export import FastPlateExport  # noqa: PLC0415

    class Detector(torch.nn.Module):
        def forward(self, _images):
            return torch.tensor(
                [[[5, 5, 25], [5, 5, 25], [10, 10, 10], [10, 10, 10], [0.9, 0.8, 0.7]]]
            )

    result = FastPlateExport(Detector())(torch.zeros((1, 3, 32, 32))).detach().numpy()
    np.testing.assert_allclose(
        result, [[0, 0, 0, 10, 10, 0, 0.9], [0, 20, 20, 30, 30, 0, 0.7]], atol=1e-6
    )
