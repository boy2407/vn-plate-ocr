"""Temporal grouping must retain evidence and keep crop/text alignment."""

import csv
import json

import pytest

from custom_alpr.plate_tracks import PlateTracker, group_observations


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
    with pytest.raises(ValueError, match="320, 640"):
        web.inference_settings({"image_size": value})


def test_quality_selects_clearer_crop_of_consensus_without_rewriting_ocr():
    clear = observation(time=0, confidence=0.9, crop="clear.png")
    clear["crop_quality"] = {"score": 0.85}
    blurred = observation(time=0.1, confidence=0.99, crop="blurred.png")
    blurred["crop_quality"] = {"score": 0.1}
    wrong = observation("51H21580", time=0.2, confidence=0.999, crop="wrong.png")
    wrong["crop_quality"] = {"score": 0.99}
    result = group_observations([clear, blurred, wrong])[0]
    assert result["text"] == "51H21510"
    assert result["crop"] == "clear.png"
    assert result["text_agreement"] == pytest.approx(2 / 3)
    assert wrong["text"] == "51H21580"


def test_ocr_selection_can_choose_high_confidence_minority_and_preserves_evidence():
    rows = [
        observation(time=0),
        observation(time=0.1),
        observation("51H21580", time=0.2, confidence=0.99, crop="highest.png"),
    ]
    assert group_observations(rows)[0]["text"] == "51H21510"
    result = group_observations(rows, selection="ocr")[0]
    assert result["text"] == "51H21580"
    assert result["crop"] == "highest.png"
    assert result["text_agreement"] == pytest.approx(1 / 3)
    assert rows[0]["text"] == "51H21510"


def test_ocr_review_and_csv_follow_selection_without_rerunning_models(tmp_path):
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
    consensus = client.get(route).get_json()
    selected = client.get(route + "?selection=ocr").get_json()
    assert consensus["items"][0]["text"] == "51H21510"
    assert selected["items"][0]["text"] == "30A12345"
    downloaded = client.get(route + "?selection=ocr&format=csv")
    assert "30A12345" in downloaded.text and "51H21510" not in downloaded.text
    assert client.get(route + "?selection=unknown").status_code == 400
    assert json.loads((folder / "results.json").read_text())[0]["text"] == "51H21510"


def test_average_votes_over_five_frames_and_averages_only_winning_string():
    rows = [
        observation(time=0, confidence=0.6, crop="a.png"),
        observation(time=0.1, confidence=0.8, crop="b.png"),
        observation("51H21580", time=0.2, confidence=0.99, crop="wrong.png"),
        observation(time=0.3, confidence=0.7, crop="c.png"),
    ]
    result = group_observations(rows, selection="average")[0]
    assert result["text"] == "51H21510"
    assert result["ocr_confidence"] == pytest.approx(0.7)
    assert result["averaged_frames"] == 3 and result["window_frames"] == 4
    assert result["representative_ocr_confidence"] == 0.8
    assert result["crop"] == "b.png"
    assert result["text_agreement"] == 0.75
    assert [r["ocr_confidence"] for r in rows] == [0.6, 0.8, 0.99, 0.7]


def test_average_uses_latest_five_and_excludes_earlier_predictions():
    rows = [observation(time=i / 10, confidence=0.9) for i in range(5)]
    rows += [observation("30A12345", time=(i + 5) / 10, confidence=0.6 + i / 10) for i in range(3)]
    rows += [observation(time=i / 10, confidence=0.99) for i in (8, 9)]
    result = group_observations(rows, selection="average")[0]
    assert result["text"] == "30A12345"
    assert result["window_frames"] == 5 and result["averaged_frames"] == 3
    assert result["ocr_confidence"] == pytest.approx(0.7)
    assert result["observations"] == 10
    assert result["text_agreement"] == 0.6


def test_new_track_starts_a_fresh_window_and_merged_gallery_does_not_average_tracks():
    tracker = PlateTracker(selection="average")
    tracker.observe(observation(time=0, confidence=0.5))
    tracker.observe(observation(time=0.1, confidence=0.7))
    latest = tracker.observe(observation(time=2, confidence=0.9))
    assert latest["window_frames"] == latest["averaged_frames"] == 1
    assert latest["ocr_confidence"] == 0.9
    merged = tracker.summaries()[0]
    assert merged["track_count"] == 2 and merged["observations"] == 3
    assert merged["window_frames"] == 1 and merged["text_votes"] == 1
    assert merged["ocr_confidence"] == 0.9


def test_average_review_csv_keeps_original_frame_predictions(tmp_path):
    web = pytest.importorskip("custom_alpr.web_app")
    job_id = "c" * 32
    folder = tmp_path / job_id
    folder.mkdir()
    rows = [
        observation(time=0, confidence=0.6),
        observation(time=0.1, confidence=0.8),
        observation("30A12345", time=0.2, confidence=0.99),
    ]
    (folder / "results.json").write_text(json.dumps(rows))
    (folder / "job.json").write_text(json.dumps(dict(id=job_id, status="done", kind="video")))
    client = web.create_app(output=tmp_path).test_client()
    route = f"/api/jobs/{job_id}/results"
    result = client.get(route + "?selection=average").get_json()["items"][0]
    assert result["text"] == "51H21510" and result["ocr_confidence"] == pytest.approx(0.7)
    csv_text = client.get(route + "?selection=average&format=csv").text
    assert "mean_ocr_confidence" in csv_text and "averaged_frames" in csv_text
    raw = client.get(route + "?view=frames&selection=average").get_json()["items"]
    assert raw[2]["text"] == "30A12345" and raw[2]["ocr_confidence"] == 0.99
    assert web.temporal_setting({}) == "average"
    with pytest.raises(ValueError):
        web.temporal_setting({"selection": "bad"})
