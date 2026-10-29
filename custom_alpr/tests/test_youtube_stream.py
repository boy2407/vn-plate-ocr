"""YouTube streams: only YouTube links, newest frame first, and a working stop."""

import threading

import numpy as np
import pytest

web = pytest.importorskip("custom_alpr.web_app")


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=G_G8A6JU_LI",
        "https://youtu.be/G_G8A6JU_LI",
        "http://m.youtube.com/watch?v=abc",
    ],
)
def test_youtube_links_are_accepted(url):
    assert web.youtube_url(f"  {url} ") == url


@pytest.mark.parametrize(
    "url",
    [
        "",
        "file:///etc/passwd",
        "https://evil.example/watch?v=x",
        "https://youtube.com.evil.example/x",
        "rtsp://www.youtube.com/live",
    ],
)
def test_other_urls_are_rejected(url):
    with pytest.raises(ValueError):
        web.youtube_url(url)


@pytest.mark.parametrize("value", ["4", "601", "bad"])
def test_live_duration_is_bounded(value):
    with pytest.raises(ValueError):
        web.live_seconds_setting({"max_seconds": value})


class FakeCapture:
    """Delivers numbered frames, then ends; ``release`` is recorded."""

    def __init__(self, count, gate=None):
        self.count, self.sent, self.gate, self.released = count, 0, gate, False

    def read(self):
        if self.gate is not None:
            self.gate.wait()
        if self.sent >= self.count:
            return False, None
        self.sent += 1
        return True, np.full((2, 2, 3), self.sent, dtype=np.uint8)

    def release(self):
        self.released = True


def test_buffer_keeps_order_and_drops_only_the_oldest_when_full():
    capture = FakeCapture(5)
    reader = web.FrameBuffer(capture, size=3)
    reader.thread.join(timeout=5)  # a burst arrived faster than we read
    numbers = [reader.next()[1] for _ in range(3)]
    assert numbers == [3, 4, 5]
    assert reader.next(timeout=0.1) == (None, None)
    reader.close()
    assert capture.released


def test_buffer_waits_for_a_stalled_stream_then_resumes():
    gate = threading.Event()
    reader = web.FrameBuffer(FakeCapture(1, gate))
    assert reader.next(timeout=0.05) == (None, None)
    gate.set()
    frame, number = reader.next()
    assert number == 1 and frame[0, 0, 0] == 1
    reader.close()


def test_youtube_job_validates_and_stop_is_accepted(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(
        web.ThreadPoolExecutor, "submit", lambda _self, fn, job_id: started.append(job_id)
    )
    client = web.create_app(output=tmp_path).test_client()
    bad = client.post("/api/youtube", data={"url": "https://example.com/v"})
    assert bad.status_code == 400
    response = client.post(
        "/api/youtube",
        data={"url": "https://youtu.be/G_G8A6JU_LI", "max_seconds": "30"},
    )
    assert response.status_code == 202
    job_id = response.get_json()["id"]
    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["source_url"] == "https://youtu.be/G_G8A6JU_LI"
    assert job["max_seconds"] == 30 and job["kind"] == "video"
    assert started == [job_id]
    assert client.post(f"/api/jobs/{job_id}/stop").get_json() == {"stopping": True}
    assert client.post("/api/jobs/missing/stop").status_code == 404
