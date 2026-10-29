"""Local website that reads plates from YouTube links with YOLO and PARSeq model F."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import subprocess
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from urllib.parse import urlparse

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from ultralytics import YOLO

from custom_alpr.preprocessing import crop_quality, padded_box, prepare_ocr_input, rectify_plate
from custom_alpr.plate_tracks import PlateTracker, group_observations
from custom_alpr.video_demo import (
    DEFAULT_DETECTOR,
    DEFAULT_OCR,
    ROOT,
    draw_detections,
    font_for_video,
    load_ocr,
    ocr_arrays,
    ocr_crops,
    resolve_device,
    select_boxes,
)

OUTPUT = ROOT / "custom_alpr/outputs/web-demo"
DEFAULT_MIN_HEIGHT = 16
# Input of the TensorRT export (FP32): a 16:9 frame at imgsz 640. On a 1080p clip it
# gave the same 1371 boxes as best.pt at 2.6 ms vs 7.0 ms per frame.
ENGINE_SHAPE = (384, 640)


class InferenceEngine:
    """One worker owns the models so concurrent uploads do not overlap GPU calls."""

    fixed_size: tuple[int, int] | None = None

    def __init__(self, detector: Path, ocr: Path, device: str):
        self.device, self.yolo_device = resolve_device(device)
        self.detector = YOLO(str(detector), task="detect")
        # A TensorRT engine has one fixed input; the 320/640 setting cannot apply.
        self.fixed_size = ENGINE_SHAPE if Path(detector).suffix == ".engine" else None
        self.recognizer = load_ocr(ocr, self.device)
        # Compile/warm accelerator paths before starting the clip's playback clock.
        for shape in ((640, 640, 3), (360, 640, 3)):
            self.detector.predict(
                np.zeros(shape, dtype=np.uint8),
                device=self.yolo_device,
                verbose=False,
                imgsz=self.fixed_size or 640,
            )
        ocr_crops([np.zeros((64, 128, 3), dtype=np.uint8)], self.recognizer, self.device)
        self.warmed_sizes = {640}

    def warm_detector(self, image_size: int):
        if self.fixed_size is None and image_size not in self.warmed_sizes:
            self.detector.predict(
                np.zeros((360, 640, 3), dtype=np.uint8),
                imgsz=image_size,
                device=self.yolo_device,
                verbose=False,
            )
            self.warmed_sizes.add(image_size)

    def recognize(
        self,
        frame: np.ndarray,
        confidence: float,
        *,
        denoise: bool = False,
        sharpen: bool = False,
        padding: bool = False,
        rectify: bool = False,
        image_size: int = 640,
        min_height: int = 0,
        track: bool = False,
    ) -> tuple[list[dict], int]:
        """OCR every detected plate; returns (results, boxes too small to read).

        With ``track`` YOLO's ByteTrack assigns IDs that persist across calls;
        call ``reset_tracker`` before a new clip.
        """
        options = dict(
            conf=confidence,
            device=self.yolo_device,
            verbose=False,
            imgsz=self.fixed_size or image_size,
        )
        if track:
            prediction = self.detector.track(
                frame, persist=True, tracker="bytetrack.yaml", **options
            )[0]
        else:
            prediction = self.detector.predict(frame, **options)[0]
        height, width = frame.shape[:2]
        boxes = select_boxes(prediction.boxes, width, height, with_ids=True)
        # Below this height digits are a few pixels tall; OCR output is a guess.
        readable = [box for box in boxes if box[3] - box[1] >= min_height]
        too_small = len(boxes) - len(readable)
        if not readable:
            return [], too_small
        results, inputs = [], []
        for x1, y1, x2, y2, detection_confidence, yolo_track_id in readable:
            crop_box = padded_box([x1, y1, x2, y2], width, height) if padding else [x1, y1, x2, y2]
            cx1, cy1, cx2, cy2 = crop_box
            crop = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
            rectified, corners = rectify_plate(crop) if rectify else (crop, None)
            # Prepared once: the same pixels go to the model and to the saved preview.
            inputs.append(prepare_ocr_input(rectified, denoise=denoise, sharpen=sharpen))
            results.append(
                {
                    "box": [x1, y1, x2, y2],
                    "crop_box": crop_box,
                    "padding": padding,
                    "rectified": corners is not None,
                    "plate_corners": corners,
                    "crop_quality": crop_quality(crop),
                    "detection_confidence": detection_confidence,
                    "yolo_track_id": yolo_track_id,
                    "preprocessing": {"denoise": denoise, "sharpen": sharpen, "rectify": rectify},
                }
            )
        # One OCR call per frame, however many plates YOLO found.
        recognized = ocr_arrays(inputs, self.recognizer, self.device)
        for item, model_input, (text, score) in zip(results, inputs, recognized, strict=True):
            item.update(text=text, ocr_confidence=float(score), _ocr_input=model_input)
        return results, too_small

    def reset_tracker(self):
        """Forget ByteTrack state so IDs from a previous clip are not reused."""
        predictor = self.detector.predictor
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()


def write_image(path: Path, image: np.ndarray, message: str):
    if not cv2.imwrite(str(path), image):
        raise OSError(message)


def store_crops(  # noqa: PLR0913
    frame: np.ndarray,
    results: list[dict],
    folder: Path,
    index: int,
    fps: float,
    pool: ThreadPoolExecutor | None = None,
) -> list:
    """Name and save crops; with ``pool`` the PNG writes run in the background.

    Returns the pending write futures so the caller can surface write errors.
    """
    pending = []

    def save(path, image, message):
        if pool is None:
            write_image(path, image, message)
        else:
            pending.append(pool.submit(write_image, path, image, message))

    for number, item in enumerate(results):
        x1, y1, x2, y2 = item.get("crop_box", item["box"])
        name = f"crop-{index:08d}-{number}.png"
        save(folder / name, frame[y1:y2, x1:x2].copy(), "Không thể lưu crop biển số.")
        model_input = item.pop("_ocr_input", None)
        if model_input is None:
            model_input = prepare_ocr_input(
                cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), **item["preprocessing"]
            )
        input_name = f"ocr-input-{index:08d}-{number}.png"
        save(
            folder / input_name,
            cv2.cvtColor(model_input, cv2.COLOR_RGB2BGR),
            "Không thể lưu ảnh đầu vào OCR.",
        )
        item.update(
            frame=index,
            time_seconds=index / fps,
            crop=name,
            ocr_input=input_name,
            width=x2 - x1,
            height=y2 - y1,
        )
    return pending


def export_mp4(raw: Path, output: Path, input_fps: float | None = None):
    """Re-encode for browsers; ``input_fps`` retimes a clip written at a guessed rate."""
    retime = ["-r", f"{input_fps:.3f}"] if input_fps else []
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            *retime,
            "-i",
            str(raw),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    raw.unlink()


YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
MAX_LIVE_SECONDS = 600


def youtube_url(value: str) -> str:
    """Accept only http(s) YouTube links; the server never opens arbitrary URLs."""
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in YOUTUBE_HOSTS:
        raise ValueError("Chỉ hỗ trợ link youtube.com hoặc youtu.be.")
    return parsed.geturl()


def live_seconds_setting(form) -> int:
    try:
        value = int(form.get("max_seconds", "60"))
        if not 5 <= value <= MAX_LIVE_SECONDS:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"Thời lượng phải từ 5 đến {MAX_LIVE_SECONDS} giây.") from exc
    return value


def resolve_stream(url: str) -> tuple[str, str]:
    """Direct media URL (≤1080p, video only) and title for a YouTube video or livestream."""
    import yt_dlp  # noqa: PLC0415 -- optional dependency, only needed for YouTube jobs

    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080]/best[height<=1080]",
    }
    try:
        info = yt_dlp.YoutubeDL(options).extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise ValueError("Không lấy được luồng video từ YouTube.") from exc
    return info["url"], info.get("title") or url


class FrameBuffer:
    """Read a stream on its own thread into a short buffer, dropping the oldest.

    HLS delivers frames in bursts of a few seconds. A buffer of about two
    seconds lets OCR work through a burst instead of keeping only its last frame,
    while a stream faster than OCR still loses its oldest frames instead of
    drifting further behind the live picture.
    """

    def __init__(self, capture: cv2.VideoCapture, size: int = 60):
        self.capture = capture
        self.frames: deque = deque(maxlen=size)
        self.received, self.ended = 0, False
        self.changed = Condition()
        self.thread = Thread(target=self.read, daemon=True, name="stream-reader")
        self.thread.start()

    def read(self):
        while not self.ended:
            ok, frame = self.capture.read()
            with self.changed:
                if not ok:
                    self.ended = True
                else:
                    self.received += 1
                    self.frames.append((frame, self.received))
                self.changed.notify_all()

    def next(self, timeout: float = 10.0):
        """Oldest buffered frame and its stream number; (None, None) when the stream ends."""
        with self.changed:
            self.changed.wait_for(lambda: self.ended or self.frames, timeout=timeout)
            if not self.frames:
                return None, None
            return self.frames.popleft()

    def close(self):
        self.ended = True
        self.thread.join(timeout=5)
        self.capture.release()


def process_stream(engine, job, folder, update, emit_frame, stop):  # noqa: PLR0915
    """Recognize a YouTube video or livestream as it plays, newest frame first.

    About two seconds of frames are buffered; when OCR is slower than the stream
    the oldest are dropped (``skipped_frames``), so the preview stays near live. Ends at ``max_seconds``, at the end of the stream, or
    when ``stop`` is set.
    """
    if not shutil.which("ffmpeg"):
        raise ValueError("Cần cài FFmpeg để xuất MP4 phát được trên trình duyệt.")
    update(phase="Đang lấy luồng YouTube")
    stream_url, title = resolve_stream(job["source_url"])
    capture = cv2.VideoCapture(stream_url)
    if not capture.isOpened():
        raise ValueError("Không mở được luồng video YouTube.")
    stream_fps = capture.get(cv2.CAP_PROP_FPS)
    stream_fps = stream_fps if math.isfinite(stream_fps) and 0 < stream_fps < 240 else 30.0
    reader = FrameBuffer(capture, size=max(1, round(2 * stream_fps)))
    raw = folder / "raw.mp4"
    writer = None
    results, processed, last_number, too_small = [], 0, 0, 0
    tracker = PlateTracker()
    engine.reset_tracker()
    inference_seconds = 0.0
    pending = []
    writes = ThreadPoolExecutor(max_workers=2, thread_name_prefix="crop-writer")
    update(name=title, source_fps=stream_fps, phase="Đang nhận diện trực tiếp")
    started = time.perf_counter()
    try:
        while not stop.is_set() and time.perf_counter() - started < job["max_seconds"]:
            frame, number = reader.next()
            if frame is None:
                break
            last_number = number
            height, width = frame.shape[:2]
            scale = min(1.0, 1280 / max(width, height, 1))
            preview_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            if writer is None:
                font = font_for_video(max(16, round(min(preview_size) / 35)))
                # Real rate is unknown until the end; export_mp4 retimes it.
                writer = cv2.VideoWriter(
                    str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, preview_size
                )
                if not writer.isOpened():
                    raise ValueError("Không thể tạo video kết quả.")
            preview = cv2.resize(frame, preview_size) if scale < 1 else frame
            frame_started = time.perf_counter()
            detected, small = engine.recognize(
                frame,
                job["confidence"],
                **job["preprocessing"],
                image_size=job["image_size"],
                min_height=job.get("min_height", 0),
                track=True,
            )
            too_small += small
            elapsed = frame_started - started
            pending += store_crops(frame, detected, folder, processed, 1.0, pool=writes)
            for item in detected:
                item.update(time_seconds=elapsed, stream_frame=number)
            smoothed = tracker.observe_frame(detected)
            results.extend(detected)
            display_boxes = [
                dict(summary, box=[round(v * scale) for v in item["box"]])
                for item, summary in zip(detected, smoothed, strict=True)
            ]
            annotated = draw_detections(preview, display_boxes, font)
            writer.write(annotated)
            frame_seconds = time.perf_counter() - frame_started
            inference_seconds += frame_seconds
            emit_frame(annotated, processed)
            processed += 1
            update(
                frames=processed,
                total_frames=0,
                detections=len(results),
                plate_count=len(tracker.summaries()),
                latest=smoothed[0] if smoothed else None,
                latest_raw=detected[0] if detected else None,
                latest_frame=processed - 1,
                time_seconds=elapsed,
                processed_frames=processed,
                skipped_frames=max(0, number - processed),
                too_small=too_small,
                frame_ms=round(frame_seconds * 1000, 1),
                inference_fps=round(processed / max(inference_seconds, 0.001), 1),
                live_fps=round(processed / max(time.perf_counter() - started, 0.001), 1),
            )
    finally:
        reader.close()
        if writer is not None:
            writer.release()
        writes.shutdown(wait=True)
    for future in pending:
        future.result()  # re-raise a failed crop write
    if not processed:
        raise ValueError("Không đọc được frame nào từ luồng YouTube.")
    duration = time.perf_counter() - started
    update(phase="Đang xuất video")
    export_mp4(raw, folder / "result.mp4", input_fps=processed / max(duration, 0.001))
    update(
        frames=processed,
        total_frames=processed,
        fps=processed / max(duration, 0.001),
        media="result.mp4",
        processed_frames=processed,
        skipped_frames=max(0, last_number - processed),
        stream_seconds=round(duration, 1),
    )
    return results


def write_results(folder, results):
    plates = group_observations(results)
    (folder / "results.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    with (folder / "results.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "frame",
                "time_seconds",
                "text",
                "detection_confidence",
                "ocr_confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "crop",
                "ocr_input",
                "denoise",
                "sharpen",
                "padding",
                "rectify_requested",
                "rectified",
                "track_id",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item["frame"],
                    item["time_seconds"],
                    item["text"],
                    item["detection_confidence"],
                    item["ocr_confidence"],
                    *item["box"],
                    item["crop"],
                    item.get("ocr_input", ""),
                    item.get("preprocessing", {}).get("denoise", False),
                    item.get("preprocessing", {}).get("sharpen", False),
                    item.get("padding", False),
                    item.get("preprocessing", {}).get("rectify", False),
                    item.get("rectified", False),
                    item["track_id"],
                ]
            )
    (folder / "plates.json").write_text(json.dumps(plates, ensure_ascii=False), encoding="utf-8")
    fields = [
        "text",
        "observations",
        "first_time",
        "last_time",
        "text_agreement",
        "ocr_confidence",
        "detection_confidence",
        "frame",
        "crop",
        "track_count",
    ]
    with (folder / "plates.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(plates)
    return plates


def inference_settings(form) -> tuple[float, dict, int]:
    try:
        confidence = float(form.get("confidence", "0.4"))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError
    except ValueError as exc:
        raise ValueError("Confidence phải từ 0 đến 1.") from exc
    options = {name: form.get(name, "0") for name in ("denoise", "sharpen", "padding", "rectify")}
    if any(value not in {"0", "1"} for value in options.values()):
        raise ValueError("Tùy chọn tiền xử lý phải là 0 hoặc 1.")
    try:
        image_size = int(form.get("image_size", "640"))
        # Trained at 320; 960/1280 lost recall (0.97 -> 0.83) on validation.
        if image_size not in {320, 640}:
            raise ValueError
    except ValueError as exc:
        raise ValueError("Độ phân giải YOLO phải là 320 hoặc 640.") from exc
    return confidence, {name: value == "1" for name, value in options.items()}, image_size


def min_height_setting(form) -> int:
    """Crops shorter than this many source pixels are counted, not read."""
    try:
        value = int(form.get("min_height", str(DEFAULT_MIN_HEIGHT)))
        if not 0 <= value <= 200:
            raise ValueError
    except ValueError as exc:
        raise ValueError("Chiều cao crop tối thiểu phải từ 0 đến 200 px.") from exc
    return value


def create_app(  # noqa: PLR0915 -- route registration for a small local application
    detector=DEFAULT_DETECTOR, ocr=DEFAULT_OCR, device="auto", output=OUTPUT
):
    app = Flask(__name__, static_folder="web", static_url_path="/static")
    output.mkdir(parents=True, exist_ok=True)
    jobs, engines, live_frames, stops = {}, [], {}, {}
    for saved in output.glob("*/job.json"):
        try:
            recovered = json.loads(saved.read_text(encoding="utf-8"))
            if recovered["id"] != saved.parent.name:
                continue
            if recovered["status"] not in {"done", "error"}:
                recovered.update(status="error", error="Tác vụ dừng khi server khởi động lại.")
            jobs[recovered["id"]] = recovered
        except (OSError, ValueError, KeyError):
            app.logger.warning("Could not recover job metadata: %s", saved)
    lock = Lock()
    changes = Condition(lock)
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plate-inference")

    def run(job_id):
        started = time.perf_counter()
        with lock:
            job = jobs[job_id].copy()
        folder = output / job_id

        def update(**values):
            with changes:
                jobs[job_id].update(values)
                changes.notify_all()

        def emit_frame(frame, index):
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise OSError("Không thể tạo frame trực tiếp.")
            with changes:
                live_frames[job_id] = (index, encoded.tobytes())
                changes.notify_all()

        try:
            update(status="running", phase="Đang nạp và làm nóng mô hình")
            if not engines:
                engines.append(InferenceEngine(detector, ocr, device))
            engines[0].warm_detector(job["image_size"])
            results = process_stream(engines[0], job, folder, update, emit_frame, stops[job_id])
            plates = write_results(folder, results)
            update(
                status="done",
                phase="Hoàn tất",
                seconds=round(time.perf_counter() - started, 2),
                detections=len(results),
                plate_count=len(plates),
                device=str(engines[0].device),
            )
        except Exception as exc:  # Surface background worker errors to the UI.
            app.logger.exception("Inference job %s failed", job_id)
            update(status="error", error=str(exc), phase="Xử lý thất bại")
        finally:
            with lock:
                snapshot = jobs[job_id].copy()
            (folder / "job.json").write_text(
                json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
            )

    def enqueue(confidence, preprocessing, image_size, min_height, **extra):
        job_id = uuid.uuid4().hex
        (output / job_id).mkdir()
        stops[job_id] = Event()
        with lock:
            jobs[job_id] = dict(
                id=job_id,
                name="YouTube",
                kind="video",
                confidence=confidence,
                preprocessing=preprocessing,
                image_size=image_size,
                min_height=min_height,
                models={"ocr_checkpoint": str(ocr), "detector_checkpoint": str(detector)},
                status="queued",
                phase="Đang chờ",
                frames=0,
                total_frames=0,
                detections=0,
                plate_count=0,
                **extra,
            )
        worker.submit(run, job_id)
        return jsonify(id=job_id), 202

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.post("/api/youtube")
    def youtube():
        try:
            url = youtube_url(request.form.get("url", ""))
            max_seconds = live_seconds_setting(request.form)
            confidence, preprocessing, image_size = inference_settings(request.form)
            min_height = min_height_setting(request.form)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return enqueue(
            confidence,
            preprocessing,
            image_size,
            min_height,
            source_url=url,
            max_seconds=max_seconds,
        )

    @app.post("/api/jobs/<job_id>/stop")
    def stop(job_id):
        with lock:
            if job_id not in jobs:
                return jsonify(error="Không có tác vụ này."), 404
        if job_id in stops:
            stops[job_id].set()
        return jsonify(stopping=True)

    @app.get("/api/jobs/<job_id>")
    def status(job_id):
        with lock:
            job = jobs.get(job_id)
            return (
                (jsonify(job.copy()), 200) if job else (jsonify(error="Không có tác vụ này."), 404)
            )

    @app.get("/api/jobs/<job_id>/results")
    def results_page(job_id):
        with lock:
            if job_id not in jobs or jobs[job_id]["status"] != "done":
                return jsonify(error="Kết quả chưa sẵn sàng."), 404
        page = max(0, request.args.get("page", 0, type=int))
        view = request.args.get("view", "plates")
        if view not in {"plates", "frames"}:
            return jsonify(error="Chế độ xem không hợp lệ."), 400
        folder = output / job_id
        results = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        if not (folder / "plates.json").exists():
            write_results(folder, results)  # jobs saved before plates.csv existed
        if view == "plates":
            # Regrouped from raw observations, so jobs saved before a voting change agree.
            results = group_observations(results)
        if request.args.get("format") == "csv":
            stream = io.StringIO()
            fields = [
                "text",
                "ocr_confidence",
                "detection_confidence",
                "frame",
                "crop",
                "observations",
                "text_agreement",
                "track_count",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
            return Response(
                "\ufeff" + stream.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment; filename=plates.csv"},
            )
        return jsonify(
            items=results[page * 24 : (page + 1) * 24],
            total=len(results),
            view=view,
        )

    @app.get("/api/jobs/<job_id>/live")
    def live(job_id):
        with lock:
            if job_id not in jobs or jobs[job_id]["kind"] != "video":
                return jsonify(error="Không có luồng video này."), 404

        def frames():
            sent = -1
            while True:
                with changes:
                    changes.wait_for(
                        lambda sent=sent: (
                            live_frames.get(job_id, (-1,))[0] > sent
                            or jobs[job_id]["status"] in {"done", "error"}
                        ),
                        timeout=15,
                    )
                    index, data = live_frames.get(job_id, (-1, b""))
                    finished = jobs[job_id]["status"] in {"done", "error"}
                if index > sent:
                    sent = index
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(data)).encode()
                        + b"\r\n\r\n"
                        + data
                        + b"\r\n"
                    )
                if finished:
                    break

        return Response(
            frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/media/<job_id>/<path:filename>")
    def media(job_id, filename):
        with lock:
            if job_id not in jobs:
                return jsonify(error="Không có tác vụ này."), 404
        return send_from_directory(output / job_id, filename)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--detector",
        type=Path,
        default=DEFAULT_DETECTOR,
        help="YOLO .pt, or a TensorRT .engine exported at 384x640",
    )
    parser.add_argument("--ocr-checkpoint", type=Path, default=DEFAULT_OCR)
    args = parser.parse_args()
    for path in (args.detector, args.ocr_checkpoint):
        if not path.is_file():
            parser.error(f"Checkpoint not found: {path}")
    create_app(args.detector, args.ocr_checkpoint, args.device).run(
        host=args.host, port=args.port, threaded=True, use_reloader=False
    )


if __name__ == "__main__":
    main()
