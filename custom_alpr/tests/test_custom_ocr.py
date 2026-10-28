import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from custom_alpr.augment import (
    looks_straight,
    random_blue_plate,
    random_downscale,
    random_gaussian_noise,
    random_glare,
    random_photometric,
    random_rotate_perspective,
    random_salt_pepper,
    random_view_tilt,
)
from custom_alpr.custom_ocr import (
    BOS_INDEX,
    EOS_INDEX,
    PAD_INDEX,
    PlateOCR2D,
    PlateOCRParseq,
    build_model,
    cloze_attention_mask,
    context_from_prediction,
    encode_target,
    permutation_attention_mask,
)
from custom_alpr.custom_ocr_data import (
    PlateOCRDataset,
    RepeatFactorSampler,
    plate_ocr_collate,
    rare_letter_repeat_factors,
)


@pytest.mark.parametrize(
    "augment",
    [
        random_downscale,
        random_glare,
        random_photometric,
        random_gaussian_noise,
        random_salt_pepper,
        random_blue_plate,
        random_rotate_perspective,
        random_view_tilt,
    ],
)
def test_augmentations_keep_shape_and_respect_probability(augment):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(64, 128, 3), dtype=np.uint8)
    assert augment(image, rng, probability=0.0) is image
    changed = augment(image, rng, probability=1.0)
    assert changed.shape == image.shape
    assert changed.dtype == np.uint8
    assert not np.array_equal(changed, image)


def test_left_to_right_order_gives_causal_mask():
    blocked = permutation_attention_mask(torch.arange(4))
    # Query i sees BOS and t0..t(i-1), i.e. slots 0..i.
    expected_visible = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(~blocked, expected_visible)


def test_permuted_order_shows_only_earlier_tokens():
    # Slots are [BOS, t0, t1]; reading order t1 -> t0 -> t2.
    visible = ~permutation_attention_mask(torch.tensor([1, 0, 2]))
    assert visible[:, 0].all()
    assert not visible[1, 1:].any()  # t1 is read first: only BOS
    assert visible[0, 2] and not visible[0, 1]  # t0 sees t1 (slot 2), not itself
    assert visible[2, 1] and visible[2, 2]  # t2 sees t0 and t1


def test_cloze_mask_hides_only_own_token():
    visible = ~cloze_attention_mask(4, torch.device("cpu"))
    for query in range(3):
        assert not visible[query, query + 1]
        assert visible[query].sum() == 3
    assert visible[3].all()


def test_context_from_prediction_pads_after_eos():
    a, b = encode_target("AB")[1:3]
    predicted = torch.tensor([[a, b, EOS_INDEX, a, b]])
    context = context_from_prediction(predicted)
    assert context.tolist() == [[BOS_INDEX, a, b, EOS_INDEX, PAD_INDEX]]


def test_parseq_trains_and_generates():
    torch.manual_seed(0)
    model = PlateOCRParseq(model_dim=32, attention_heads=2, encoder_layers=1, decoder_layers=1)
    targets = [torch.tensor(encode_target(text)) for text in ("51D61603", "QM5588")]
    images, decoder_inputs, target_ids, _ = plate_ocr_collate(
        [(torch.rand(3, 64, 128), target, "") for target in targets]
    )
    loss = model.permutation_loss(images, decoder_inputs, target_ids, label_smoothing=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.position_queries.grad is not None
    assert model(images, decoder_inputs).shape[:2] == decoder_inputs.shape

    model.eval()
    texts, confidences = model.generate_with_confidence(images)
    assert len(texts) == len(confidences) == 2
    assert all(0.0 <= c <= 1.0 for c in confidences)


def test_build_model_uses_checkpoint_decoder():
    config = {"max_output_length": 14}
    assert type(build_model({"model_config": config})) is PlateOCR2D
    assert type(build_model({"model_config": config, "decoder": "parseq"})) is PlateOCRParseq


def test_blue_plate_makes_text_light_on_blue():
    image = np.full((64, 128, 3), 230, dtype=np.uint8)
    image[20:44, 40:88:6] = 20  # thin dark strokes ("text") on a light plate
    recolored = random_blue_plate(image, np.random.default_rng(0), probability=1.0)
    background, text = recolored[5, 5].astype(int), recolored[30, 40].astype(int)
    assert background[2] > background[0] + 80  # blue dominates the background
    assert text.min() > 200  # text becomes near white


def test_rare_letter_repeat_factors_follow_rfs():
    texts = ["51A12345"] * 300 + ["29Z12345"] * 50 + ["30N12345"] * 100
    factors, rare = rare_letter_repeat_factors(texts, rare_threshold=200)
    assert rare == {"Z", "N"}
    assert factors[0] == 1.0  # A is common
    assert factors[300] == pytest.approx(2.0)  # Z: sqrt(200 / 50)
    assert factors[350] == pytest.approx(2**0.5)  # N: sqrt(200 / 100)


def test_blue_plate_leaves_light_on_dark_plates_unchanged():
    image = np.full((64, 128, 3), 40, dtype=np.uint8)
    image[20:44, 40:88:6] = 230  # thin light strokes on a dark (already blue-like) plate
    assert random_blue_plate(image, np.random.default_rng(0), probability=1.0) is image


def test_rotation_keeps_whole_plate_inside_frame():
    image = np.zeros((64, 128, 3), dtype=np.uint8)
    image[:, :4] = 255  # bright left and right edges stand in for edge characters
    image[:, -4:] = 255
    for seed in range(20):
        warped = random_rotate_perspective(image, np.random.default_rng(seed), probability=1.0)
        assert warped[:, :64].max() == 255 and warped[:, 64:].max() == 255


def test_repeat_factor_sampler_keeps_every_plate_and_repeats_rare_ones():
    sampler = RepeatFactorSampler([1.0, 1.0, 4.0, 2.5], generator=torch.Generator().manual_seed(0))
    counts = {i: 0 for i in range(4)}
    for _ in range(200):
        epoch = list(sampler)
        assert set(epoch) == {0, 1, 2, 3}  # no plate is skipped
        for i in epoch:
            counts[i] += 1
    assert counts[0] == counts[1] == 200 and counts[2] == 800
    assert 450 < counts[3] < 550  # 2 or 3 times, 2.5 on average


def test_blue_plate_ignores_dark_surroundings():
    image = np.full((64, 128, 3), 20, dtype=np.uint8)  # dark car body around the plate
    image[16:48, 32:96] = 230  # white plate in the middle
    image[26:38, 50:78:6] = 20  # thin dark strokes
    recolored = random_blue_plate(image, np.random.default_rng(0), probability=1.0)
    assert recolored is not image


def test_strong_downscale_shrinks_more_crops(tmp_path):
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, size=(64, 128, 3), dtype=np.uint8)).save(
        tmp_path / "a.png"
    )
    (tmp_path / "m.csv").write_text("image_path,plate_text\na.png,51A12345\n", encoding="utf-8")

    def blurred_share(strong: bool) -> float:
        dataset = PlateOCRDataset(tmp_path / "m.csv", augment=True, strong_downscale=strong, seed=1)
        sharp = torch.stack([dataset[0][0] for _ in range(200)])
        # High-frequency energy drops when a crop was shrunk and re-enlarged.
        return float(
            (sharp[..., 1:] - sharp[..., :-1]).abs().mean(dim=(1, 2, 3)).lt(0.15).float().mean()
        )

    assert blurred_share(True) > blurred_share(False)


def test_view_tilt_keeps_whole_plate_inside_frame():
    image = np.zeros((64, 128, 3), dtype=np.uint8)
    image[:, :4] = 255  # bright left and right edges stand in for edge characters
    image[:, -4:] = 255
    for seed in range(20):
        warped = random_view_tilt(image, np.random.default_rng(seed), probability=1.0)
        assert warped[:, :64].max() == 255 and warped[:, 64:].max() == 255


def test_looks_straight_rejects_tilted_and_featureless_crops():
    image = np.full((64, 128, 3), 230, dtype=np.uint8)
    image[10:14, 8:120] = 20  # level top and bottom edges
    image[50:54, 8:120] = 20
    assert looks_straight(image)
    tilted = random_view_tilt(image, np.random.default_rng(0), probability=1.0)
    rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D((64, 32), 15, 1.0), (128, 64))
    assert not looks_straight(rotated)
    assert not looks_straight(np.full((64, 128, 3), 128, dtype=np.uint8))
    assert tilted.shape == image.shape
