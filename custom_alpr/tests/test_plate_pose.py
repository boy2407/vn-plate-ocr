"""Detector boxes take the corners of the overlapping pose box, or keep their own crop."""

import numpy as np

from custom_alpr.plate_pose import corner_box, match_corners

CORNERS = np.array(
    [[[12, 10], [48, 12], [47, 30], [11, 28]], [[200, 50], [260, 50], [260, 80], [200, 80]]]
)
CONFIDENT = np.array([[0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9]])
POSE_BOXES = [[11, 10, 48, 30], [200, 50, 260, 80]]


def test_takes_corners_of_most_overlapping_pose_box():
    corners = match_corners([195, 45, 265, 85], POSE_BOXES, CORNERS, CONFIDENT)
    assert corners == [[200, 50], [260, 50], [260, 80], [200, 80]]


def test_no_overlap_or_weak_corner_keeps_detector_crop():
    assert match_corners([500, 500, 540, 520], POSE_BOXES, CORNERS, CONFIDENT) is None
    assert match_corners([10, 8, 50, 32], [], None, None) is None
    weak = CONFIDENT.copy()
    weak[0, 2] = 0.1
    assert match_corners([10, 8, 50, 32], POSE_BOXES, CORNERS, weak) is None
    assert match_corners([10, 8, 50, 32], POSE_BOXES, CORNERS, None) is None


def test_corner_box_is_clipped_to_frame():
    assert corner_box([[-3, 5], [40.2, 4], [41, 30.5], [2, 31]], 40, 30) == [0, 4, 40, 30]
