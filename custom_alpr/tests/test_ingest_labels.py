"""Labels must be validated, conflicts held back and splits kept per video."""

import csv
import json
import sys

from custom_alpr import ingest_labels
from custom_alpr.ingest_labels import (
    inconsistent_tracks,
    ocr_report,
    review_sample,
    split_by_camera,
    validate_row,
)

QUEUE_FIELDS = ["crop_id", "image_path", "video_id", "track_id", "verdict", "plate_text", "lines", "note"]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row(verdict="label", text="92A39826", lines="2", video="v1", track="t1", crop="c1"):
    return dict(
        crop_id=crop, image_path=f"{video}/crops/{crop}.png", video_id=video, track_id=track,
        verdict=verdict, plate_text=text, lines=lines, note="",
    )  # fmt: skip


def test_valid_label_and_non_label_verdicts_pass():
    assert validate_row(row()) == []
    assert validate_row(row(text="92a-398.26")) == []  # normalized before the format check
    assert validate_row(row("unreadable", "", "2")) == []
    assert validate_row(row("not_plate", "", "")) == []


def test_invalid_rows_are_explained():
    assert validate_row(row("maybe")) == ["unknown verdict 'maybe'"]
    assert "does not match PLATE_FORMAT" in validate_row(row(text="92A3"))[0]
    assert validate_row(row(text="")) == ["label without plate_text"]
    assert validate_row(row(lines="3")) == ["lines must be 1 or 2, got '3'"]
    assert validate_row(row("unreadable", "92A39826")) == ["plate_text given with verdict unreadable"]


def test_tracks_with_two_texts_are_inconsistent():
    rows = [
        row(crop="a"),
        row(crop="b", text="95A39826"),
        row(crop="c", track="t2"),
        row("unreadable", "", crop="d", track="t2"),
    ]
    assert inconsistent_tracks(rows) == {("v1", "t1")}


def test_split_never_puts_a_camera_in_two_splits():
    cameras = [f"c{index}" for index in range(10)] * 3
    splits = split_by_camera(cameras, ["c9"], val_fraction=0.2, seed=1)
    assert set(splits) == {f"c{index}" for index in range(10)}
    assert splits["c9"] == "test"
    assert sum(split == "val" for split in splits.values()) == 2
    assert split_by_camera(cameras, ["c9"], 0.2, seed=1) == splits


def test_single_camera_goes_to_train():
    assert split_by_camera(["c1", "c1"], [], 0.15, seed=1) == {"c1": "train"}


def test_review_sample_is_fixed_by_seed():
    rows = [row(crop=f"c{index}") for index in range(100)]
    first = review_sample(rows, 0.05, seed=7)
    assert len(first) == 5
    assert review_sample(list(reversed(rows)), 0.05, seed=7) == first


def test_ocr_report_counts_crops_tracks_and_swaps():
    rows = [
        dict(row(crop="a"), label="92A39826", ocr_text="95A39826", track_consensus="92A39826"),
        dict(row(crop="b"), label="92A39826", ocr_text="92A39826", track_consensus="92A39826"),
    ]
    report = ocr_report(rows)
    assert report["all"] == dict(crops=2, crop_exact=0.5, tracks=1, track_exact=1.0)
    assert report["one_line"]["crops"] == 0
    assert report["confusions"] == [["2->5", 1]]


def test_end_to_end_conflicts_are_held_back(tmp_path, monkeypatch):
    collected = tmp_path / "collected"
    crops = collected / "v1/crops"
    crops.mkdir(parents=True)
    fields = ["crop_id", "image_path", "video_id", "track_id", "verdict", "plate_text", "lines", "note"]
    queue_rows = [
        row(crop="a"), row(crop="b"),  # consistent track t1
        row(crop="c", track="t2"), row(crop="d", track="t2", text="95A39826"),  # conflict
        row("not_plate", "", "", crop="e", track="t3"),
        row("label", "BAD", "1", crop="f", track="t4"),
    ]  # fmt: skip
    meta_fields = ["crop_id", "video_id", "frame", "box", "ocr_text", "track_consensus"]
    with (collected / "queue.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(queue_rows)
    with (collected / "v1/metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=meta_fields)
        writer.writeheader()
        for item in queue_rows:
            writer.writerow(dict(crop_id=item["crop_id"], video_id="v1", frame=1, box="[0, 0, 1, 1]",
                                 ocr_text="95A39826", track_consensus="92A39826"))  # fmt: skip
            (crops / f"{item['crop_id']}.png").write_bytes(item["crop_id"].encode())
            (crops / f"{item['crop_id']}_pad.png").write_bytes(b"pad" + item["crop_id"].encode())
        (crops / "a_ref.png").write_bytes(b"refined a")
    output = tmp_path / "manifest"
    monkeypatch.setattr(
        sys, "argv", ["ingest", "--collected", str(collected), "--output", str(output), "--review-fraction", "0"]
    )
    ingest_labels.main()

    with (output / "train.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert list(manifest[0]) == ["image_path", "plate_text", "source", "type", "sha256"]
    # Track t1 only: two crops, each with its padded view; crop a also refined.
    assert len(manifest) == 5
    assert {item["plate_text"] for item in manifest} == {"92A39826"}
    assert {item["source"] for item in manifest} == {"youtube"}
    assert {item["type"] for item in manifest} == {"1"}
    assert all((output / item["image_path"]).exists() for item in manifest)
    with (collected / "review.csv").open(newline="", encoding="utf-8") as handle:
        reasons = sorted(item["reason"].split(":")[0] for item in csv.DictReader(handle))
    assert reasons == ["invalid", "track_conflict", "track_conflict"]
    report = json.loads((collected / "ingest-report.json").read_text(encoding="utf-8"))
    assert report["negatives"] == 1
    assert report["ocr"]["all"]["crop_exact"] == 0.0
    assert report["ocr"]["all"]["track_exact"] == 1.0


def test_two_recordings_of_one_camera_share_a_split(tmp_path, monkeypatch):
    collected = tmp_path / "collected"
    rows = [
        row(video="dayA", crop="a"), row(video="dayB", crop="b"),  # same camera, two days
        row(video="drive", crop="c", text="30A12345", lines="1"),
    ]  # fmt: skip
    cameras = {"dayA": "danang-ptz", "dayB": "danang-ptz", "drive": "hanoi-drive"}
    write_csv(collected / "queue.csv", QUEUE_FIELDS, rows)
    for item in rows:
        write_csv(
            collected / item["video_id"] / "metadata.csv",
            ["crop_id", "camera", "video_id", "frame", "box", "ocr_text", "track_consensus"],
            [dict(crop_id=item["crop_id"], camera=cameras[item["video_id"]], video_id=item["video_id"],
                  frame=1, box="[0, 0, 1, 1]", ocr_text="", track_consensus="")],
        )  # fmt: skip
        image = collected / item["image_path"]
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(item["crop_id"].encode())
    output = tmp_path / "manifest"
    monkeypatch.setattr(sys, "argv", [
        "ingest", "--collected", str(collected), "--output", str(output),
        "--test-cameras", "danang-ptz", "--review-fraction", "0",
    ])  # fmt: skip
    ingest_labels.main()

    def texts(split):
        with (output / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            return sorted(item["plate_text"] for item in csv.DictReader(handle))

    assert texts("test") == ["92A39826", "92A39826"]
    assert texts("train") == ["30A12345"]
