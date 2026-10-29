"""A refined box replaces the first-pass box only when it is plausibly tighter."""

import pytest

from custom_alpr.refine_box import SIZE, accept_refined, refine_box

FIRST = [100, 100, 240, 180]  # 140x80 first-pass box, area 11200


def test_tighter_plate_box_is_accepted():
    assert accept_refined(FIRST, [130, 115, 220, 165], 0.8)  # 90x50, 40% of the area


@pytest.mark.parametrize(
    ("refined", "confidence", "reason"),
    [
        ([130, 115, 220, 165], 0.2, "weak detection"),
        ([150, 130, 170, 140], 0.8, "tiny: a character or logo"),
        ([101, 101, 239, 179], 0.8, "not tighter"),
        ([150, 100, 160, 180], 0.8, "tall sliver, not a plate"),
        ([300, 115, 390, 165], 0.8, "outside the first box"),
    ],
)
def test_implausible_boxes_fall_back(refined, confidence, reason):
    assert not accept_refined(FIRST, refined, confidence), reason


def test_refine_maps_back_to_image_and_falls_back():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    image = np.zeros((400, 400, 3), np.uint8)
    seen = {}

    def predict(region):
        seen["shape"] = region.shape
        height, width = region.shape[:2]
        # The region is the first box plus CONTEXT on each side (about twice its
        # size); a plate in its middle 30% x 30% is ~0.6 x 0.6 of the first box.
        return [(0.35 * width, 0.35 * height, 0.65 * width, 0.65 * height, 0.9)]

    box, refined = refine_box(predict, image, FIRST)
    assert refined
    assert max(seen["shape"][:2]) == SIZE
    assert FIRST[0] < box[0] < box[2] < FIRST[2]
    assert FIRST[1] < box[1] < box[3] < FIRST[3]
    assert refine_box(lambda region: [], image, FIRST) == (FIRST, False)
