"""Temporal grouping must retain evidence and keep crop/text alignment."""

import csv
import json

import pytest

from custom_alpr.plate_tracks import PlateTracker, group_observations, per_character_vote


def observation(text="51H21510", time=0, box=None, confidence=0.9, crop="crop.png"):
    return dict(
        text=text,
        time_seconds=time,
        frame=round(time * 30),
        box=box or [10, 20, 110, 70],
        width=100,
        height=50,
        ocr_confidence=confidence,
        detection_confidence=0.8,
        crop=crop,
        preprocessing={"denoise": False, "sharpen": False},
    )


def test_repeated_frames_and_ocr_variation_choose_observed_majority_and_matching_crop():
    rows = [
        observation(time=0, confidence=0.9, crop="first.png"),
        observation("51H21580", time=0.1, confidence=0.99, crop="wrong.png"),
        observation(time=0.2, confidence=0.95, crop="best.png"),
        observation(time=0.3, confidence=0.8),
    ]
    groups = group_observations(rows)
    assert len(groups) == 1
    result = groups[0]
    assert result["text"] == "51H21510"
    assert result["crop"] == "best.png"
    assert result["ocr_confidence"] == 0.95
    assert result["observations"] == 4
    assert result["text_agreement"] == 0.75
    assert result["first_time"] == 0 and result["last_time"] == 0.3
    assert rows[1]["text"] == "51H21580"  # raw evidence is not rewritten


def test_spatially_separate_observations_are_different_tracks_even_with_same_text():
    tracker = PlateTracker()
    a = observation(time=0)
    b = observation(time=0.1, box=[500, 300, 600, 350])
    tracker.observe(a)
    tracker.observe(b)
    assert a["track_id"] != b["track_id"]
    # The gallery is unique by observed consensus text, not a vehicle counter.
    summary = tracker.summaries()
    assert len(summary) == 1
    assert summary[0]["track_count"] == 2
    assert summary[0]["observations"] == 2


def test_different_plates_far_apart_are_not_merged():
    groups = group_observations(
        [observation(), observation("30A12345", time=0.1, box=[500, 300, 600, 350])]
    )
    assert {item["text"] for item in groups} == {"51H21510", "30A12345"}


def test_time_gap_starts_new_track_and_empty_texts_are_not_merged():
    rows = [observation("", time=0), observation("", time=2)]
    groups = group_observations(rows)
    assert len(groups) == 2
    assert rows[0]["track_id"] != rows[1]["track_id"]


def test_empty_video_has_no_groups():
    assert group_observations([]) == []


def test_existing_job_gets_grouped_view_without_rerunning_model(tmp_path):
    web = pytest.importorskip("custom_alpr.web_app")
    job_id = "a" * 32
    folder = tmp_path / job_id
    folder.mkdir()
    rows = [observation(time=index / 30) for index in range(8)]
    (folder / "results.json").write_text(json.dumps(rows))
    (folder / "job.json").write_text(json.dumps(dict(id=job_id, status="done", kind="video")))
    client = web.create_app(output=tmp_path).test_client()
    grouped = client.get(f"/api/jobs/{job_id}/results").get_json()
    raw = client.get(f"/api/jobs/{job_id}/results?view=frames").get_json()
    assert grouped["total"] == 1 and grouped["items"][0]["observations"] == 8
    assert raw["total"] == 8
    assert client.get(f"/api/jobs/{job_id}/results?view=invalid").status_code == 400
    with (folder / "results.csv").open(encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        header = next(reader)
        assert len(header) == len(set(header))
        assert all(len(row) == len(header) for row in reader)
    assert (folder / "plates.csv").is_file()


@pytest.mark.parametrize("value", ["640.5", "10000", "bad"])
def test_invalid_detector_resolution_is_rejected(value):
    web = pytest.importorskip("custom_alpr.web_app")
    with pytest.raises(ValueError, match="320 hoặc 640"):
        web.inference_settings({"image_size": value})


def test_quality_selects_clearer_crop_of_consensus_without_rewriting_ocr():
    clear = observation(time=0, confidence=0.9, crop="clear.png")
    clear["crop_quality"] = {"score": 0.85}
    blurred = observation(time=0.1, confidence=0.99, crop="blurred.png")
    blurred["crop_quality"] = {"score": 0.1}
    # One sharp misread does not outvote two reads of the same string.
    wrong = observation("51H21580", time=0.2, confidence=0.999, crop="wrong.png")
    wrong["crop_quality"] = {"score": 0.5}
    result = group_observations([clear, blurred, wrong])[0]
    assert result["text"] == "51H21510"
    assert result["crop"] == "clear.png"
    assert result["text_agreement"] == pytest.approx(2 / 3)
    assert wrong["text"] == "51H21580"


def test_review_page_and_csv_regroup_raw_observations(tmp_path):
    web = pytest.importorskip("custom_alpr.web_app")
    job_id = "b" * 32
    folder = tmp_path / job_id
    folder.mkdir()
    rows = [
        observation(time=0),
        observation(time=0.1),
        observation("30A12345", time=0.2, confidence=0.99),
    ]
    (folder / "results.json").write_text(json.dumps(rows))
    (folder / "job.json").write_text(json.dumps(dict(id=job_id, status="done", kind="video")))
    client = web.create_app(output=tmp_path).test_client()
    route = f"/api/jobs/{job_id}/results"
    assert client.get(route).get_json()["items"][0]["text"] == "51H21510"
    downloaded = client.get(route + "?format=csv")
    assert "51H21510" in downloaded.text and "30A12345" not in downloaded.text
    raw = client.get(route + "?view=frames").get_json()["items"]
    assert raw[2]["text"] == "30A12345" and raw[2]["ocr_confidence"] == 0.99
    assert json.loads((folder / "results.json").read_text())[0]["text"] == "51H21510"


def test_yolo_track_id_keeps_one_track_across_a_jump_iou_would_split():
    tracker = PlateTracker()
    first = observation(time=0, box=[10, 20, 110, 70])
    first["yolo_track_id"] = 7
    jumped = observation(time=0.2, box=[400, 300, 500, 350])
    jumped["yolo_track_id"] = 7
    tracker.observe_frame([first])
    tracker.observe_frame([jumped])
    assert first["track_id"] == jumped["track_id"]


def test_different_yolo_ids_in_one_frame_never_share_a_track():
    tracker = PlateTracker()
    a, b = observation(time=0), observation("29A11111", time=0, box=[12, 21, 112, 71])
    a["yolo_track_id"], b["yolo_track_id"] = 1, 2
    tracker.observe_frame([a, b])
    c, d = observation(time=0.1), observation("29A11111", time=0.1, box=[12, 21, 112, 71])
    c["yolo_track_id"], d["yolo_track_id"] = 2, 1
    tracker.observe_frame([c, d])
    assert c["track_id"] == b["track_id"] and d["track_id"] == a["track_id"]


def test_saved_plates_match_the_review_grouping(tmp_path):
    web = pytest.importorskip("custom_alpr.web_app")
    rows = [observation(time=t / 10, crop=f"{t}.png") for t in range(3)]
    for row in rows:
        row.update(track_id=None)
    plates = web.write_results(tmp_path, rows)
    saved = json.loads((tmp_path / "plates.json").read_text(encoding="utf-8"))
    assert len(plates) == len(saved) == 1
    assert saved[0]["text_votes"] == 3


def _char_distribution(char, wrong=None):
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZĐ"
    probs = [0.001] * len(alphabet)
    probs[alphabet.index(char)] = 0.9 if wrong is None else 0.2
    if wrong is not None:
        probs[alphabet.index(wrong)] = 0.75
    return probs


def _char_observation(text, evidence_text=None, wrong_position=None, wrong_char=None, quality=1.0):
    evidence_text = evidence_text or text
    return {
        "text": text,
        "quality": quality,
        "char_probabilities": [
            _char_distribution(char, wrong_char if index == wrong_position else None)
            for index, char in enumerate(evidence_text)
        ],
    }


def test_per_character_vote_recovers_plate_with_distinct_single_digit_errors():
    truth = "30A12345"
    rows = [
        _char_observation("80A12345", "30A12345", 0, "8"),
        _char_observation("30A92345", "30A12345", 3, "9"),
        _char_observation("30A12305", "30A12345", 6, "0"),
    ]
    result = per_character_vote(rows)
    assert result["text"] == truth
    assert result["accepted"]

    tracker = PlateTracker()
    for time, row in enumerate(rows):
        item = observation(row["text"], time=time / 10)
        item.update(quality=row["quality"], char_probabilities=row["char_probabilities"])
        tracker.observe(item)
    assert tracker.summaries()[0]["text"] == "80A12345"  # string vote remains the default
    assert tracker.summaries(vote_strategy="per_character")[0]["text"] == truth


def test_per_character_vote_uses_dominant_length_and_layout():
    rows = [
        _char_observation("30A12345"),
        _char_observation("30A12345"),
        _char_observation("30A1234"),
    ]
    result = per_character_vote(rows)
    assert result["text"] == "30A12345"
    assert result["observations"] == 2


def test_per_character_vote_rejects_low_posterior_margin():
    probabilities = [[0.05] * 37 for _ in "30A12345"]
    row = {"text": "30A12345", "quality": 1.0, "char_probabilities": probabilities}
    result = per_character_vote([row], rejection_threshold=0.1)
    assert result["text"] == "unreadable"
    assert not result["accepted"]


def test_per_character_rejection_without_probabilities_is_unreadable():
    row = {"text": "30A12345", "ocr_confidence": 0.9}
    result = per_character_vote([row], rejection_threshold=0.1)
    assert result["text"] == "unreadable"
    assert result["candidate_text"] == "30A12345"
    assert not result["accepted"]
