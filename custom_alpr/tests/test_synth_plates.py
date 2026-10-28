import csv

import cv2
import numpy as np

from custom_alpr.synth_plates import (
    Render,
    add_finish,
    background_patch,
    combine_with_train,
    compose,
    glyph_library,
    new_text,
    segment,
)


def drawn_plate(text: str) -> np.ndarray:
    """A clean 1-row plate: black border, dark characters drawn apart on a light background."""
    image = np.full((50, 235, 3), 230, dtype=np.uint8)
    cv2.rectangle(image, (1, 1), (233, 48), (20, 20, 20), 2)
    for index, character in enumerate(text):
        cv2.putText(
            image, character, (12 + index * 26, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3
        )
    return image


def test_segment_finds_one_box_per_character_and_skips_border():
    background, ink, texts, boxes = segment(
        drawn_plate("51A12345"), ["51A", "12345"], two_rows=False
    )
    assert texts == ["51A12345"]
    assert len(boxes[0]) == 8
    assert background.mean() > 200 and ink.mean() < 60
    assert segment(drawn_plate("51A12345"), ["51A", "1234"], two_rows=False) is None


def test_compose_draws_new_characters_in_template_boxes():
    image = drawn_plate("51A12345")
    background, ink, texts, boxes = segment(image, ["51A", "12345"], two_rows=False)
    render = Render("type1", False, image, background, ink, texts, boxes)
    library = glyph_library([render])
    target = {c: float(np.median([g.mean() for g in glyphs])) for c, glyphs in library.items()}
    out = compose(render, ["15A54321"], library, np.random.default_rng(0), target)
    assert out.shape == image.shape and out.dtype == np.uint8
    first = boxes[0][0]
    assert not np.array_equal(
        out[first[1] : first[3], first[0] : first[2]],
        image[first[1] : first[3], first[0] : first[2]],
    )


def test_new_text_keeps_pattern_and_fills_the_largest_letter_gap():
    library = {c: [np.ones((4, 2))] for c in "0123456789ABZ"}
    plates_with = {"A": 1000, "B": 1000, "Z": 10}
    text = new_text(
        "29AĐ12345",
        library | {"Đ": [np.ones((4, 2))]},
        plates_with,
        ["51"],
        np.random.default_rng(0),
    )
    assert text[:2] == "51"  # province from the given list
    assert text[2] == "Z"  # only Z is below its target
    assert text[3] == "Đ" and text[4:].isdigit()
    assert plates_with["Z"] == 11


def test_background_patch_follows_vertical_gradient():
    column = np.linspace(100, 200, 20, dtype=np.float32)
    image = np.repeat(column[:, None, None], 10, axis=1).repeat(3, axis=2)
    patch = background_patch(image, np.zeros((20, 10)), np.zeros(3), (2, 5, 8, 15))
    assert patch.shape == (10, 6, 3)
    assert np.allclose(patch[0], image[4, 2:8]) and np.allclose(patch[-1], image[15, 2:8])


def test_combined_manifest_paths_resolve_from_output_folder(tmp_path):
    train_dir, output = tmp_path / "prepared", tmp_path / "synthetic"
    train_dir.mkdir()
    output.mkdir()
    with (train_dir / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_path", "plate_text", "source", "type", "sha256"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "../raw/a.jpg",
                "plate_text": "51A12345",
                "source": "cropped",
                "type": "0",
                "sha256": "",
            }
        )
    synthetic = [
        {
            "image_path": "images/b.png",
            "plate_text": "29Z12345",
            "source": "synthetic",
            "type": "type1",
            "sha256": "",
        }
    ]
    path = combine_with_train(train_dir / "train.csv", synthetic, output)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [(output / r["image_path"]).resolve() for r in rows] == [
        (tmp_path / "raw" / "a.jpg").resolve(),
        (output / "images" / "b.png").resolve(),
    ]


def test_add_finish_keeps_shape_and_shades_the_plate():
    image = drawn_plate("51A12345")
    background, ink, texts, boxes = segment(image, ["51A", "12345"], two_rows=False)
    render = Render("type1", False, image, background, ink, texts, boxes)
    out = add_finish(image, render, np.random.default_rng(0))
    assert out.shape == image.shape and out.dtype == np.uint8
    assert not np.array_equal(out, image)
