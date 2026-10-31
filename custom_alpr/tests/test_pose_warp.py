"""Perspective crop tests use synthetic pixels, no model or GPU."""

import numpy as np
import pytest
import torch

from custom_alpr.pose_eval import PoseBoxValidator, warp_plate


def test_warp_corner_mapping_and_real_context_margin():
    y, x = np.indices((100, 150))
    frame = np.stack([x, y, np.full_like(x, 200)], axis=-1).astype(np.uint8)
    corners = [[30, 20], [120, 20], [120, 60], [30, 60]]
    tight = warp_plate(frame, corners, 0)
    assert tight.shape == (40, 90, 3)
    np.testing.assert_array_equal(tight[0, 0], frame[20, 30])
    np.testing.assert_array_equal(tight[-1, -1], frame[60, 120])
    padded = warp_plate(frame, corners, 0.1)
    assert padded.shape == (48, 108, 3)
    assert padded[0, 0, 0] < 30 and padded[0, 0, 1] < 20
    assert padded[0, 0, 2] == 200  # Margin samples real context, not a painted border.


def test_warp_rejects_reversed_winding():
    with pytest.raises(ValueError, match="winding"):
        warp_plate(np.zeros((100, 100, 3), np.uint8), [[10, 10], [10, 90], [90, 90], [90, 10]], 0)


def test_warp_accepts_strided_xy_view_from_pose_keypoints():
    keypoints = np.array(
        [[30, 20, 0.9], [120, 20, 0.9], [120, 60, 0.9], [30, 60, 0.9]], dtype=np.float32
    )
    corners = keypoints[:, :2]
    assert not corners.flags.c_contiguous
    actual = warp_plate(np.full((100, 150, 3), 127, np.uint8), corners, 0.05)
    assert actual.shape == (44, 98, 3)
    assert actual.min() == 127


def test_pose_box_validator_does_not_treat_coordinates_as_classes():
    validator = PoseBoxValidator(args=dict(conf=0.001, iou=0.7, plots=False))
    validator.nc = 1
    validator.end2end = False
    prediction = torch.tensor(
        [20, 20, 20, 20, 0.9, 10, 10, 0.9, 30, 10, 0.9, 30, 30, 0.9, 10, 30, 0.9]
    ).reshape(1, 17, 1)
    result = validator.postprocess(prediction)[0]
    assert result["cls"].tolist() == [0.0]
    assert result["conf"].tolist() == pytest.approx([0.9])
    assert result["extra"].shape == (1, 12)
    assert validator.args.task == "detect"
