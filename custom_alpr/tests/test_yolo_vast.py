"""Integrity and augmentation boundaries for the rented-GPU bundle."""

import json
from collections import Counter
import numpy as np
import pytest
from PIL import Image
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Albumentations
from ultralytics.utils.instance import Instances

from custom_alpr.yolo_vast.corruptions import (
    EFFECTS,
    GROUPS,
    QualityTransform,
    apply_budget,
    apply_effect,
)
from custom_alpr.yolo_vast.data import (
    audit_dataset,
    runtime_yaml,
    sha256,
    training_policy,
    verify_inventory,
)
from custom_alpr.yolo_vast.trainer import QualityDataset
from custom_alpr.yolo_vast.wait_for import REQUIRED_MARKERS, gate_state


@pytest.mark.parametrize("effect", EFFECTS)
def test_corruption_keeps_geometry_source_and_reproducible_pixels(effect):
    image = np.random.default_rng(31).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    original = image.copy()
    boxes = np.array([[10.0, 12.0, 60.0, 45.0]])
    expected_boxes = boxes.copy()
    a = apply_effect(image, effect, np.random.default_rng(42), boxes=boxes)
    b = apply_effect(image, effect, np.random.default_rng(42), boxes=boxes)
    assert a.shape == image.shape and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(boxes, expected_boxes)


def test_budget_caps_combination_and_matches_target_frequency(monkeypatch):
    import custom_alpr.yolo_vast.corruptions as module  # noqa: PLC0415

    monkeypatch.setattr(module, "apply_effect", lambda image, *_args: image)
    image = np.zeros((8, 8, 3), np.uint8)
    rng = np.random.default_rng(42)
    counts = Counter()
    for _ in range(5000):
        _, effects = apply_budget(image, rng)
        counts[len(effects)] += 1
        groups = [
            next(key for key, values in GROUPS.items() if value in values) for value in effects
        ]
        assert len(effects) <= 2 and len(groups) == len(set(groups))
    assert 1850 < counts[0] < 2150
    assert 2100 < counts[1] < 2400
    assert 650 < counts[2] < 850


def test_transform_never_changes_instances_or_adds_collation_keys():
    image = np.full((64, 64, 3), 120, np.uint8)
    instances = Instances(
        np.array([[0.5, 0.5, 0.4, 0.2]], dtype=np.float32),
        segments=np.zeros((0, 1000, 2), np.float32),
        normalized=True,
    )
    before = instances.bboxes.copy()
    transform = QualityTransform("moderate")
    for _ in range(30):
        result = transform({"img": image, "instances": instances})
        assert set(result) == {"img", "instances"}
        assert result["instances"] is instances and instances.normalized
        np.testing.assert_array_equal(instances.bboxes, before)
    control = QualityTransform("control")
    assert control({"img": image, "instances": instances})["img"] is image


def make_dataset(root):
    for i, split in enumerate(("train", "val", "test")):
        images, labels = root / "images" / split, root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for j in range(4 if split == "train" else 1):
            pixels = np.random.default_rng(i * 10 + j).integers(0, 256, (64, 64, 3), np.uint8)
            Image.fromarray(pixels).save(images / f"{j}.png")
            (labels / f"{j}.txt").write_text("0 0.5 0.5 0.4 0.2\n")
    return root


def test_audit_rejects_invalid_labels_and_pixel_leakage(tmp_path):
    root = make_dataset(tmp_path)
    valid = audit_dataset(root, workers=1)
    assert valid["counts"]["train"]["images"] == 4
    label = root / "labels/train/0.txt"
    label.write_text("0 0.5 0.5 2.0 0.4\n")
    with pytest.raises(ValueError, match="Out-of-bounds"):
        audit_dataset(root, workers=1)
    label.write_text("0 0.5 0.5 0.4 0.2\n")
    (root / "images/val/0.png").write_bytes((root / "images/train/0.png").read_bytes())
    with pytest.raises(ValueError, match="Duplicate image/leakage"):
        audit_dataset(root, workers=1)


def test_transfer_hash_detects_tampering_and_runtime_paths_are_portable(tmp_path):
    root = make_dataset(tmp_path / "data")
    inventory = audit_dataset(root, workers=1)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights/initial.pt").write_bytes(b"test-weights")
    inventory["initial_sha256"] = sha256(tmp_path / "weights/initial.pt")
    (tmp_path / "inventory.json").write_text(json.dumps(inventory))
    verify_inventory(tmp_path)
    path = runtime_yaml(tmp_path, smoke=True)
    assert str(root.resolve()) in path.read_text()
    assert (tmp_path / "smoke-train.txt").read_text().count("\n") == 4
    (root / "labels/train/0.txt").write_text("")
    with pytest.raises(ValueError, match="integrity"):
        verify_inventory(tmp_path)


def test_close_mosaic_keeps_quality_hook_and_validation_stays_clean(tmp_path):
    root = make_dataset(tmp_path)
    hyp = get_cfg(
        overrides={"imgsz": 64, "mosaic": 1.0, "degrees": 0.0, "translate": 0.0, "scale": 0.0}
    )
    data = {"names": {0: "license_plate"}, "nc": 1}
    train = QualityDataset(
        img_path=str(root / "images/train"),
        imgsz=64,
        augment=True,
        hyp=hyp,
        data=data,
        quality_arm="moderate",
        batch_size=2,
    )
    assert any(isinstance(t, QualityTransform) for t in train.transforms.transforms)
    assert not any(isinstance(t, Albumentations) for t in train.transforms.transforms)
    sample = train[0]
    assert tuple(sample["img"].shape) == (3, 64, 64)
    train.close_mosaic(hyp)
    assert sum(isinstance(t, QualityTransform) for t in train.transforms.transforms) == 1
    val = QualityDataset(
        img_path=str(root / "images/val"), imgsz=64, augment=False, hyp=hyp, data=data, batch_size=2
    )
    assert not any(isinstance(t, QualityTransform) for t in val.transforms.transforms)


def test_unreviewed_empty_labels_stay_out_of_train_but_files_are_preserved(tmp_path):
    root = make_dataset(tmp_path / "data")
    label = root / "labels/train/0.txt"
    label.write_text("")
    inventory = training_policy(audit_dataset(root, workers=1))
    (tmp_path / "inventory.json").write_text(json.dumps(inventory))
    assert inventory["counts"]["train"]["images"] == 4
    assert inventory["effective_counts"]["train"]["images"] == 3
    runtime_yaml(tmp_path)
    assert str(root / "images/train/0.png") not in (tmp_path / "train.runtime.txt").read_text()
    assert label.is_file() and (root / "images/train/0.png").is_file()
    assert inventory["effective_counts"]["val"] == inventory["counts"]["val"]


def test_queue_requires_prior_exit_test_markers_and_a_quiet_gpu():
    done = "\n".join(REQUIRED_MARKERS)
    assert gate_state(True, done, [], 100) == "waiting_prior_run"
    assert gate_state(False, "epoch 60", [], 100) == "prior_run_incomplete"
    assert gate_state(False, done, [1234], 100) == "waiting_other_gpu_processes"
    assert gate_state(False, done, [], 2) == "waiting_idle_window"
    assert gate_state(False, done, [], 61) == "ready"
