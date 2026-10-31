"""Corner identity and horizontal augmentation must preserve pose semantics."""

import itertools

import pytest

from custom_alpr.pose_data import FLIP_IDX, order_corners, pose_line


def test_corner_order_is_invariant_to_source_winding_and_permutation():
    expected = [(0.1, 0.2), (0.9, 0.1), (0.8, 0.8), (0.2, 0.9)]
    for permutation in itertools.permutations(expected):
        assert order_corners(permutation) == expected


def test_flip_idx_preserves_tl_tr_br_bl_and_is_an_involution():
    points = [(0.1, 0.2), (0.9, 0.1), (0.8, 0.8), (0.2, 0.9)]
    reflected = [(1 - x, y) for x, y in points]
    flipped = [reflected[i] for i in FLIP_IDX]
    assert flipped == order_corners(reflected)
    twice = [(1 - flipped[i][0], flipped[i][1]) for i in FLIP_IDX]
    for actual, expected in zip(twice, points, strict=True):
        assert actual == pytest.approx(expected)
    assert FLIP_IDX == [1, 0, 3, 2]


def test_pose_format_and_visibility():
    values = list(map(float, pose_line("3 .1 .2 .9 .2 .9 .8 .1 .8").split()))
    assert values[:5] == pytest.approx([0, 0.5, 0.5, 0.8, 0.6])
    assert values[7::3] == [2, 2, 2, 2]
    assert len(values) == 17
    assert pose_line("0 .1 .1 .2 .1 .3 .2 .2 .3 .1 .2") is None


@pytest.mark.parametrize(
    "points", [[(0, 0)] * 4, [(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (1, 0), (0, 1), (0.1, 0.1)]]
)
def test_invalid_quadrilaterals_fail(points):
    with pytest.raises(ValueError):
        order_corners(points)


def test_skewed_quad_keeps_identities_after_horizontal_flip():
    points = [(0.1, 0.1), (0.3, 0.25), (0.25, 0.6), (0.15, 0.55)]
    ordered = order_corners(points)
    mirrored = [(1 - x, y) for x, y in ordered]
    assert [mirrored[i] for i in FLIP_IDX] == order_corners(mirrored)
