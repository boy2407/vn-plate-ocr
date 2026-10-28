"""Local upload website for trained YOLO and PARSeq model E."""

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Condition, Lock

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
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
    ocr_crops,
    resolve_device,
    select_best_box,
)

OUTPUT = ROOT / "custom_alpr/outputs/web-demo"
SAMPLE = ROOT / "custom_alpr/outputs/video-demo/highway-test-sample.mp4"
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".mp4"}


class InferenceEngine:
    """One worker owns the models so concurrent uploads do not overlap GPU calls."""

    def __init__(self, detector: Path, ocr: Path, device: str):
        self.device, self.yolo_device = resolve_device(device)
        self.detector = YOLO(str(detector), task="detect")
        self.recognizer = load_ocr(ocr, self.device)
        # Compile/warm accelerator paths before starting the clip's playback clock.
        for shape in ((640, 640, 3), (360, 640, 3)):
            self.detector.predict(
                np.zeros(shape, dtype=np.uint8), device=self.yolo_device, verbose=False
            )
        ocr_crops([np.zeros((64, 128, 3), dtype=np.uint8)], self.recognizer, self.device)
        self.warmed_sizes = {640}

    def warm_detector(self, image_size: int):
        if image_size not in self.warmed_sizes:
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
    ) -> list[dict]:
        prediction = self.detector.predict(
            frame, conf=confidence, device=self.yolo_device, verbose=False, imgsz=image_size
        )[0]
        height, width = frame.shape[:2]
        best = select_best_box(prediction.boxes, width, height)
        if best is None:
            return []
        x1, y1, x2, y2, detection_confidence = best
        crop_box = padded_box([x1, y1, x2, y2], width, height) if padding else [x1, y1, x2, y2]
        cx1, cy1, cx2, cy2 = crop_box
        crop = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
        _, corners = rectify_plate(crop) if rectify else (crop, None)
        text, score = ocr_crops(
            [crop], self.recognizer, self.device, denoise=denoise, sharpen=sharpen, rectify=rectify
        )[0]
        return [
            {
                "box": [x1, y1, x2, y2],
                "crop_box": crop_box,
                "padding": padding,
                "rectified": corners is not None,
                "plate_corners": corners,
                "crop_quality": crop_quality(crop),
                "detection_confidence": detection_confidence,
                "text": text,
                "ocr_confidence": float(score),
                "preprocessing": {"denoise": denoise, "sharpen": sharpen, "rectify": rectify},
            }
        ]


def store_crops(frame: np.ndarray, results: list[dict], folder: Path, index: int, fps: float):
    for number, item in enumerate(results):
        x1, y1, x2, y2 = item.get("crop_box", item["box"])
        name = f"crop-{index:08d}-{number}.png"
        if not cv2.imwrite(str(folder / name), frame[y1:y2, x1:x2]):
            raise OSError("Không thể lưu crop biển số.")
        model_input = prepare_ocr_input(
            cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), **item["preprocessing"]
        )
        input_name = f"ocr-input-{index:08d}-{number}.png"
        if not cv2.imwrite(str(folder / input_name), cv2.cvtColor(model_input, cv2.COLOR_RGB2BGR)):
            raise OSError("Không thể lưu ảnh đầu vào OCR.")
        item.update(
            frame=index,
            time_seconds=index / fps,
            crop=name,
            ocr_input=input_name,
            width=x2 - x1,
            height=y2 - y1,
        )


def process_video(engine, job, folder, update, emit_frame):  # noqa: PLR0915
    if not shutil.which("ffmpeg"):
        raise ValueError("Cần cài FFmpeg để xuất MP4 phát được trên trình duyệt.")
    capture = cv2.VideoCapture(str(folder / job["input"]))
    if not capture.isOpened():
        raise ValueError("Không đọc được MP4. Hãy chọn một video hợp lệ.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if math.isfinite(fps) and fps > 0 else 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, 1280 / max(width, height, 1))
    preview_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    raw = folder / "raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, preview_size)
    results, index, processed, skipped = [], 0, 0, 0
    tracker = PlateTracker(selection=job.get("temporal_selection", "average"))
    inference_seconds = 0.0
    started = time.perf_counter()
    update(source_fps=fps)
    font = font_for_video(max(16, round(min(preview_size) / 35)))
    try:
        if not writer.isOpened():
            raise ValueError("Không thể tạo video kết quả.")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            preview = cv2.resize(frame, preview_size) if scale < 1 else frame
            # Keep pace with the source clock; stale frames never queue for OCR.
            deadline = started + index / fps
            now = time.perf_counter()
            if index and now > deadline + 1 / fps:
                writer.write(preview)
                index += 1
                skipped += 1
                continue
            if now < deadline:
                time.sleep(deadline - now)
            frame_started = time.perf_counter()
            detected = engine.recognize(
                frame, job["confidence"], **job["preprocessing"], image_size=job["image_size"]
            )
            store_crops(frame, detected, folder, index, fps)
            smoothed = [tracker.observe(item) for item in detected]
            results.extend(detected)
            display_boxes = [
                dict(summary, box=[round(v * scale) for v in item["box"]])
                for item, summary in zip(detected, smoothed, strict=True)
            ]
            annotated = draw_detections(preview, display_boxes, font)
            writer.write(annotated)
            frame_seconds = time.perf_counter() - frame_started
            inference_seconds += frame_seconds
            processed += 1
            emit_frame(annotated, index)
            index += 1
            update(
                frames=index,
                total_frames=total,
                detections=len(results),
                plate_count=len(tracker.summaries()),
                phase="Đang nhận diện trực tiếp",
                latest=smoothed[0] if smoothed else None,
                latest_raw=detected[0] if detected else None,
                latest_frame=index - 1,
                time_seconds=(index - 1) / fps,
                processed_frames=processed,
                skipped_frames=skipped,
                frame_ms=round(frame_seconds * 1000, 1),
                inference_fps=round(processed / max(inference_seconds, 0.001), 1),
                live_fps=round(processed / max(time.perf_counter() - started, 0.001), 1),
            )
    finally:
        capture.release()
        writer.release()
    if not index:
        raise ValueError("Video không có frame đọc được.")
    update(phase="Đang xuất video")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
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
            str(folder / "result.mp4"),
        ],
        check=True,
        capture_output=True,
    )
    raw.unlink()
    update(
        frames=index,
        total_frames=index,
        fps=fps,
        media="result.mp4",
        processed_frames=processed,
        skipped_frames=skipped,
    )
    return results


def process_image(engine, job, folder, update):
    try:
        with Image.open(folder / job["input"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Không đọc được ảnh. Hãy chọn JPG, PNG hoặc WebP hợp lệ.") from exc
    results = engine.recognize(
        frame, job["confidence"], **job["preprocessing"], image_size=job["image_size"]
    )
    store_crops(frame, results, folder, 0, 1)
    annotated = draw_detections(frame, results, font_for_video(22))
    if not cv2.imwrite(str(folder / "result.jpg"), annotated):
        raise OSError("Không thể lưu ảnh kết quả.")
    update(frames=1, total_frames=1, media="result.jpg", detections=len(results))
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
        if image_size not in {320, 640, 960, 1280}:
            raise ValueError
    except ValueError as exc:
        raise ValueError("Độ phân giải YOLO phải là 320, 640, 960 hoặc 1280.") from exc
    return confidence, {name: value == "1" for name, value in options.items()}, image_size


def temporal_setting(form):
    selection = form.get("selection", "average")
    if selection not in {"average", "ocr", "consensus"}:
        raise ValueError("Cách tổng hợp OCR không hợp lệ.")
    return selection


def create_app(  # noqa: PLR0915 -- route registration for a small local application
    detector=DEFAULT_DETECTOR, ocr=DEFAULT_OCR, device="auto", output=OUTPUT
):
    app = Flask(__name__, static_folder="web", static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024
    output.mkdir(parents=True, exist_ok=True)
    jobs, engines, live_frames = {}, [], {}
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
            if job["kind"] == "video":
                results = process_video(engines[0], job, folder, update, emit_frame)
            else:
                results = process_image(engines[0], job, folder, update)
            write_results(folder, results)
            plates = group_observations(results, selection=job.get("temporal_selection", "average"))
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

    def enqueue(source, suffix, confidence, name, preprocessing, image_size, temporal_selection):
        job_id = uuid.uuid4().hex
        folder = output / job_id
        folder.mkdir()
        filename = "input" + suffix
        source(folder / filename)
        with lock:
            jobs[job_id] = dict(
                id=job_id,
                name=name,
                input=filename,
                kind="video" if suffix == ".mp4" else "image",
                confidence=confidence,
                preprocessing=preprocessing,
                image_size=image_size,
                temporal_selection=temporal_selection,
                models={"ocr_checkpoint": str(ocr), "detector_checkpoint": str(detector)},
                status="queued",
                phase="Đang chờ",
                frames=0,
                total_frames=0,
                detections=0,
                plate_count=0,
            )
        worker.submit(run, job_id)
        return jsonify(id=job_id), 202

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.post("/api/jobs")
    def upload():
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify(error="Hãy chọn ảnh hoặc MP4."), 400
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED:
            return jsonify(error="Chỉ hỗ trợ JPG, PNG, WebP và MP4."), 400
        try:
            confidence, preprocessing, image_size = inference_settings(request.form)
            temporal_selection = temporal_setting(request.form)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return enqueue(
            file.save,
            suffix,
            confidence,
            Path(file.filename).name,
            preprocessing,
            image_size,
            temporal_selection,
        )

    @app.post("/api/sample")
    def sample():
        if not SAMPLE.is_file():
            return jsonify(error="Không có clip mẫu trên máy này."), 404
        try:
            confidence, preprocessing, image_size = inference_settings(request.form)
            temporal_selection = temporal_setting(request.form)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return enqueue(
            lambda path: shutil.copyfile(SAMPLE, path),
            ".mp4",
            confidence,
            "Cao tốc · clip mẫu",
            preprocessing,
            image_size,
            temporal_selection,
        )

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
        selection = request.args.get("selection", "consensus")
        if selection not in {"consensus", "ocr", "average"}:
            return jsonify(error="Cách chọn crop không hợp lệ."), 400
        if view not in {"plates", "frames"}:
            return jsonify(error="Chế độ xem không hợp lệ."), 400
        folder = output / job_id
        if not (folder / "plates.json").exists():
            write_results(folder, json.loads((folder / "results.json").read_text(encoding="utf-8")))
        filename = "plates.json" if view == "plates" else "results.json"
        results = json.loads((folder / filename).read_text(encoding="utf-8"))
        if view == "plates" and selection in {"ocr", "average"}:
            raw = json.loads((folder / "results.json").read_text(encoding="utf-8"))
            results = group_observations(raw, selection=selection)
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
                "averaged_frames",
                "window_frames",
                "mean_ocr_confidence",
                "representative_ocr_confidence",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
            return Response(
                "\ufeff" + stream.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition": f"attachment; filename=plates-{selection}.csv"},
            )
        return jsonify(
            items=results[page * 24 : (page + 1) * 24],
            total=len(results),
            view=view,
            selection=selection,
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

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error="Tệp vượt quá giới hạn 300 MB."), 413

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
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
