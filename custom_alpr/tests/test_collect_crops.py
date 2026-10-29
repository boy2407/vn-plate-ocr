"""Crop selection must stay small, spread in time and free of duplicates."""

import pytest

from custom_alpr.collect_crops import (
    CandidatePool,
    queue_rows,
    select_track_crops,
    stretch_contrast,
    track_priority,
    video_id_from_url,
)


def candidate(time, quality=0.5, width=80, height=40, sha=None):
    return dict(time=time, quality=quality, width=width, height=height, sha256=sha or f"s{time}")


def test_parked_car_keeps_k_time_spread_crops():
    # 200 near-identical reads of one parked car over 100 seconds.
    rows = [candidate(t * 0.5, quality=0.5 + (t % 7) / 100) for t in range(200)]
    chosen = select_track_crops(rows, k=4)
    assert len(chosen) == 4
    times = [item["time"] for item in chosen]
    assert times == sorted(times)
    # One crop from each quarter of the visit.
    assert [int(time // 25) for time in times] == [0, 1, 2, 3]


def test_duplicate_pixels_are_selected_once():
    rows = [candidate(t, sha="same") for t in range(10)] + [candidate(20, sha="other")]
    chosen = select_track_crops(rows, k=4)
    assert sorted(item["sha256"] for item in chosen) == ["other", "same"]


def test_bins_prefer_sharp_and_large_crops():
    rows = [candidate(0, quality=0.9, width=40, height=20), candidate(1, quality=0.8, width=160, height=80)]
    assert select_track_crops(rows, k=1)[0]["time"] == 1


def test_short_track_tops_up_and_never_exceeds_available():
    assert len(select_track_crops([candidate(0, sha="a"), candidate(0, sha="b")], k=4)) == 2
    assert select_track_crops([], k=4) == []


def test_pool_memory_is_bounded_and_spans_the_whole_track():
    pool = CandidatePool(size=10)
    built = []
    for index in range(1000):
        pool.offer(lambda index=index: built.append(index) or {"time": index})
    assert len(pool.items) < 10
    assert pool.items[0]["time"] == 0
    assert pool.items[-1]["time"] > 700
    # Skipped observations are never built (no PNG encoding for them).
    assert len(built) < 100


def test_priority_ranks_hard_tracks_first():
    hard = [dict(ocr_conf=0.5, width=30, height=22, quality=0.7)] * 5  # small two-line, unsure
    easy = [dict(ocr_conf=0.95, width=200, height=45, quality=0.7)] * 5  # large one-line, confident
    assert track_priority(hard, agreement=0.3) > 0.7
    assert track_priority(easy, agreement=1.0) < 0.05


def test_unreadable_tracks_rank_below_readable_hard_tracks():
    # Same disagreement, but every crop of one track is blurred or blown out.
    sharp = [dict(ocr_conf=0.6, width=60, height=40, quality=0.7)] * 4
    blurred = [dict(ocr_conf=0.4, width=40, height=30, quality=0.15)] * 4
    assert track_priority(sharp, agreement=0.2) > 0.5
    assert track_priority(blurred, agreement=0.1) == 0
    # One sharp crop is enough for a person to read the plate.
    mixed = blurred[:3] + sharp[:1]
    assert track_priority(mixed, agreement=0.1) > track_priority(blurred, agreement=0.1)


def test_queue_hides_ocr_output_and_skips_queued_crops(tmp_path):
    queue = tmp_path / "queue.csv"
    queue.write_text(
        "crop_id,image_path,video_id,track_id,verdict,plate_text,lines,note\nold,a.png,v,1,,,,\n",
        encoding="utf-8",
    )
    selected = [
        dict(crop_id=crop_id, image_path=f"{crop_id}.png", video_id="v", track_id="1",
             ocr_text="51G12345", ocr_conf=0.9)
        for crop_id in ("old", "new")
    ]  # fmt: skip
    rows = queue_rows(selected, queue)
    assert [row["crop_id"] for row in rows] == ["new"]
    assert "ocr_text" not in rows[0]
    assert "ocr_conf" not in rows[0]
    assert rows[0]["verdict"] == rows[0]["plate_text"] == ""


def test_stretch_contrast_lifts_dark_crops():
    np = pytest.importorskip("numpy")
    dark = np.tile(np.linspace(10, 40, 64, dtype=np.float32), (32, 1))[..., None].repeat(3, 2).astype(np.uint8)
    stretched = stretch_contrast(dark)
    assert stretched.min() == 0
    assert stretched.max() == 255
    assert stretched.shape == dark.shape
    assert stretch_contrast(np.full((8, 8, 3), 128, np.uint8)).dtype == np.uint8  # flat crop, no divide by zero


def test_video_id_from_watch_and_short_urls():
    assert video_id_from_url("https://www.youtube.com/watch?v=wzTiGOlidk4&t=30s") == "wzTiGOlidk4"
    assert video_id_from_url("https://youtu.be/wzTiGOlidk4") == "wzTiGOlidk4"
