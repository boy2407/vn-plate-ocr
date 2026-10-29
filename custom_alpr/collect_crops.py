"""Collect a few diverse plate crops per track from a YouTube video for labelling.

Every sampled frame goes through the demo's detector, OCR and tracker. Only a
bounded, time-spread pool of crops is kept per track; at the end each track
keeps a handful of sharp, large, time-separated crops and gets a labelling
priority (low OCR agreement or confidence, small or two-line plates first).

Crops land in ``data/collected/<video_id>/`` with an internal ``metadata.csv``
(includes OCR output); new rows are appended to ``data/collected/queue.csv``,
which carries no OCR output so the labeller is not primed by the model.

    python -m custom_alpr.collect_crops https://www.youtube.com/watch?v=...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import time
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "custom_alpr/data/collected"
GUIDE = Path(__file__).resolve().parent / "labeling/GUIDE.md"
TWO_LINE_ASPECT = 2.0  # Width/height below this is treated as a two-line plate.
POOL_SIZE = 60  # Candidate crops kept in memory per track.
# Readability ramp on brightness-normalized crop_quality: of 55 hand-checked day
# and night crops none below 0.47 was readable (AUC 0.93; raw scores, 0.71, put
# readable night plates below unreadable day ones).
UNREADABLE_QUALITY, READABLE_QUALITY = 0.4, 0.6
QUEUE_FIELDS = ["crop_id", "image_path", "video_id", "track_id", "verdict", "plate_text", "lines", "note"]
METADATA_FIELDS = [
    "crop_id", "camera", "video_id", "frame", "time", "track_id", "box", "width", "height",
    "ocr_text", "ocr_conf", "quality", "track_consensus", "priority", "sha256", "padded_path",
    "refined_path", "refined_box", "ocr_text_refined",
]  # fmt: skip


def video_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    return parse_qs(parsed.query).get("v", [None])[0] or parsed.path.rstrip("/").rsplit("/", 1)[-1]


def crop_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_two_line(width: float, height: float) -> bool:
    return height > 0 and width / height < TWO_LINE_ASPECT


def stretch_contrast(crop):
    """Map the 2nd-98th intensity percentiles to 0-255 so dark scenes score like day ones."""
    import numpy as np  # noqa: PLC0415 -- keep the module importable without numpy

    low, high = np.percentile(crop, 2), np.percentile(crop, 98)
    scaled = (crop.astype(np.float32) - low) * 255 / max(high - low, 1)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def crop_score(candidate: dict) -> float:
    """Sharp and large beats blurred and small; the quality score is bounded in [0, 1]."""
    return candidate["quality"] * sqrt(candidate["width"] * candidate["height"])


@dataclass
class CandidatePool:
    """Time-spread sample of a track's crops with bounded memory.

    When full, every other candidate is dropped and only every ``stride``-th
    later observation is offered, so a parked car read hundreds of times still
    keeps crops from its whole visit.
    """

    size: int = POOL_SIZE
    stride: int = 1
    seen: int = 0
    items: list = field(default_factory=list)

    def offer(self, make) -> bool:
        """``make()`` builds the candidate only when it is kept, so skipped crops cost nothing."""
        self.seen += 1
        if (self.seen - 1) % self.stride:
            return False
        self.items.append(make())
        if len(self.items) >= self.size:
            self.items = self.items[::2]
            self.stride *= 2
        return True


def select_track_crops(candidates: list[dict], k: int = 4) -> list[dict]:
    """Best crop per equal time bin, unique by sha256, topped up by score."""
    if not candidates:
        return []
    start = min(item["time"] for item in candidates)
    span = max(item["time"] for item in candidates) - start
    bins: dict[int, list[dict]] = {}
    for item in candidates:
        index = min(k - 1, int((item["time"] - start) / span * k)) if span else 0
        bins.setdefault(index, []).append(item)
    chosen, hashes = [], set()
    for index in sorted(bins):
        for item in sorted(bins[index], key=crop_score, reverse=True):
            if item["sha256"] not in hashes:
                chosen.append(item)
                hashes.add(item["sha256"])
                break
    for item in sorted(candidates, key=crop_score, reverse=True):
        if len(chosen) >= k:
            break
        if item["sha256"] not in hashes:
            chosen.append(item)
            hashes.add(item["sha256"])
    return sorted(chosen, key=lambda item: item["time"])


def track_priority(observations: list[dict], agreement: float) -> float:
    """Labelling priority in [0, 1]: where the model is most likely wrong yet a person can read it."""
    confidence = statistics.median(item["ocr_conf"] for item in observations)
    height = statistics.median(item["height"] for item in observations)
    aspect = statistics.median(item["width"] / item["height"] for item in observations)
    small = min(1.0, max(0.0, (40 - height) / 24))  # 1 at 16 px, 0 from 40 px
    hard = (
        0.35 * (1 - agreement)
        + 0.25 * (1 - confidence)
        + 0.2 * small
        + 0.2 * (aspect < TWO_LINE_ASPECT)
    )
    # Blur and glare also look "hard" but only yield `unreadable` labels, so a
    # track whose sharpest crop is unreadable sinks to the bottom of the queue.
    best = max(item["quality"] for item in observations)
    readable = (best - UNREADABLE_QUALITY) / (READABLE_QUALITY - UNREADABLE_QUALITY)
    return round(hard * min(1.0, max(0.0, readable)), 4)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_rows(path: Path, fields: list[str], rows: list[dict]):
    new = not path.exists()
    if not new:
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        if header != fields:
            raise SystemExit(f"{path} has columns {header}, expected {fields}; migrate it first.")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def known_hashes(output: Path) -> set[str]:
    """sha256 of every crop already collected, from any video."""
    return {row["sha256"] for path in output.glob("*/metadata.csv") for row in read_rows(path)}


def queue_rows(selected: list[dict], queue: Path) -> list[dict]:
    """Labelling rows without OCR output; crops already queued are skipped."""
    queued = {row["crop_id"] for row in read_rows(queue)}
    return [
        dict(
            crop_id=item["crop_id"],
            image_path=item["image_path"],
            video_id=item["video_id"],
            track_id=item["track_id"],
            verdict="",
            plate_text="",
            lines="",
            note="",
        )
        for item in selected
        if item["crop_id"] not in queued
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--camera", default=None, help="Scene/camera group; splits never share one")
    parser.add_argument("--start", type=float, default=0, help="Seconds into the video")
    parser.add_argument("--seconds", type=float, default=120, help="Footage to read")
    parser.add_argument("--every", type=int, default=3, help="Use every Nth frame")
    parser.add_argument("--imgsz", type=int, default=640, help="320 gives loose boxes")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--min-height", type=int, default=None)
    parser.add_argument("--crops-per-track", type=int, default=4)
    parser.add_argument("--min-observations", type=int, default=2)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detector", type=Path, default=None)
    parser.add_argument("--ocr-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:  # noqa: PLR0915
    # Torch and Ultralytics are imported here so the pure helpers load without them.
    import cv2  # noqa: PLC0415

    from custom_alpr.plate_tracks import PlateTracker, normalized_text  # noqa: PLC0415
    from custom_alpr.preprocessing import crop_quality, padded_box, prepare_ocr_input  # noqa: PLC0415
    from custom_alpr.refine_box import refine_box, yolo_predictor  # noqa: PLC0415
    from custom_alpr.video_demo import DEFAULT_DETECTOR, DEFAULT_OCR, ocr_arrays  # noqa: PLC0415
    from custom_alpr.web_app import (  # noqa: PLC0415
        DEFAULT_MIN_HEIGHT,
        InferenceEngine,
        resolve_stream,
    )

    args = parse_args()
    video_id = video_id_from_url(args.url)
    # Recordings of one camera on different days still share parked cars.
    camera = args.camera or video_id
    folder = args.output / video_id
    (folder / "crops").mkdir(parents=True, exist_ok=True)
    if GUIDE.exists() and not (args.output / "GUIDE.md").exists():
        shutil.copy(GUIDE, args.output / "GUIDE.md")
    # Track IDs restart every run; the run tag keeps labels of two runs apart.
    run = time.strftime("%Y%m%d%H%M%S")

    stream_url, title = resolve_stream(args.url)
    capture = cv2.VideoCapture(stream_url)
    if not capture.isOpened():
        raise SystemExit("Could not open the YouTube stream.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    # Absolute frame numbers keep crop IDs unique across segments of one video.
    first = round(args.start * fps)
    if first:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        landed = round(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if abs(landed - first) > fps:
            raise SystemExit(f"Could not seek to {args.start}s (landed on frame {landed}).")
        first = landed
    engine = InferenceEngine(
        args.detector or DEFAULT_DETECTOR, args.ocr_checkpoint or DEFAULT_OCR, args.device
    )
    engine.warm_detector(args.imgsz)
    engine.reset_tracker()
    # A separate model instance: predict() on the tracking model shares its
    # predictor, whose ByteTrack callbacks would see the second-pass boxes.
    from ultralytics import YOLO  # noqa: PLC0415

    predict = yolo_predictor(
        YOLO(str(args.detector or DEFAULT_DETECTOR), task="detect"), engine.yolo_device
    )
    tracker = PlateTracker()
    min_height = DEFAULT_MIN_HEIGHT if args.min_height is None else args.min_height
    pools: dict[int, CandidatePool] = {}
    stats: dict[int, list[dict]] = {}
    index, frames, reads = first, 0, 0
    while index < first + args.seconds * fps:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        if (index - 1) % args.every:
            continue
        frames += 1
        height, width = frame.shape[:2]
        results, _ = engine.recognize(
            frame, args.confidence, image_size=args.imgsz, min_height=min_height, track=True
        )
        for item in results:
            x1, y1, x2, y2 = item["box"]
            # Stream time, not processing time: the tracker's gap is in video seconds.
            item.update(time_seconds=(index - 1) / fps, frame=index - 1, width=x2 - x1, height=y2 - y1)
        tracker.observe_frame(results)
        reads += len(results)
        for item in results:
            x1, y1, x2, y2 = item["box"]
            observation = dict(
                frame=item["frame"],
                time=round(item["time_seconds"], 3),
                box=item["box"],
                width=item["width"],
                height=item["height"],
                ocr_text=normalized_text(item["text"]),
                ocr_conf=round(item["ocr_confidence"], 4),
                quality=crop_quality(stretch_contrast(frame[y1:y2, x1:x2]))["score"],
            )
            stats.setdefault(item["track_id"], []).append(observation)

            def candidate(observation=observation, box=(x1, y1, x2, y2)):
                # Cut from the original frame, never from the 128x64 model input.
                cx1, cy1, cx2, cy2 = box
                px1, py1, px2, py2 = padded_box(list(box), width, height)
                tight = cv2.imencode(".png", frame[cy1:cy2, cx1:cx2])[1].tobytes()
                padded = cv2.imencode(".png", frame[py1:py2, px1:px2])[1].tobytes()
                # Second detector pass with full-frame context; None when it falls back.
                refined_box, refined = refine_box(predict, frame, list(box))
                refined_png, refined_text = None, ""
                if refined:
                    rx1, ry1, rx2, ry2 = refined_box
                    crop = frame[ry1:ry2, rx1:rx2]
                    refined_png = cv2.imencode(".png", crop)[1].tobytes()
                    model_input = prepare_ocr_input(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    refined_text = normalized_text(ocr_arrays([model_input], engine.recognizer, engine.device)[0][0])
                return dict(
                    observation, sha256=crop_sha256(tight), png=tight, padded_png=padded,
                    refined_png=refined_png, refined_box=refined_box if refined else None,
                    refined_text=refined_text,
                )  # fmt: skip

            pools.setdefault(item["track_id"], CandidatePool()).offer(candidate)
    capture.release()
    if not frames:
        raise SystemExit("No frames were read from the stream.")

    tracks = []
    for track in tracker.tracks:
        observations = stats.get(track.track_id, [])
        if len(observations) < args.min_observations:
            continue
        summary = track.summary("weighted")
        aspect = statistics.median(item["width"] / item["height"] for item in observations)
        tracks.append(
            dict(
                track_id=f"{run}-{track.track_id}",
                pool=pools[track.track_id].items,
                consensus=normalized_text(summary["text"]),
                agreement=round(summary["text_agreement"], 3),
                observations=len(observations),
                two_line=aspect < TWO_LINE_ASPECT,
                priority=track_priority(observations, summary["text_agreement"]),
            )
        )
    tracks.sort(key=lambda row: row["priority"], reverse=True)
    if args.max_tracks is not None:
        tracks = tracks[: args.max_tracks]

    seen_hashes = known_hashes(args.output)
    selected = []
    for track in tracks:
        for item in select_track_crops(track["pool"], args.crops_per_track):
            if item["sha256"] in seen_hashes:
                continue
            seen_hashes.add(item["sha256"])
            x1, y1 = item["box"][:2]
            crop_id = f"{video_id}-f{item['frame']}-x{x1}y{y1}"
            (folder / "crops" / f"{crop_id}.png").write_bytes(item["png"])
            (folder / "crops" / f"{crop_id}_pad.png").write_bytes(item["padded_png"])
            if item["refined_png"]:
                (folder / "crops" / f"{crop_id}_ref.png").write_bytes(item["refined_png"])
            selected.append(
                dict(
                    crop_id=crop_id,
                    camera=camera,
                    video_id=video_id,
                    frame=item["frame"],
                    time=item["time"],
                    track_id=track["track_id"],
                    box=json.dumps(item["box"]),
                    width=item["width"],
                    height=item["height"],
                    ocr_text=item["ocr_text"],
                    ocr_conf=item["ocr_conf"],
                    quality=round(item["quality"], 4),
                    track_consensus=track["consensus"],
                    priority=track["priority"],
                    sha256=item["sha256"],
                    padded_path=f"{video_id}/crops/{crop_id}_pad.png",
                    refined_path=f"{video_id}/crops/{crop_id}_ref.png" if item["refined_png"] else "",
                    refined_box=json.dumps(item["refined_box"]) if item["refined_box"] else "",
                    ocr_text_refined=item["refined_text"],
                    image_path=f"{video_id}/crops/{crop_id}.png",
                )
            )

    metadata = [{key: row[key] for key in METADATA_FIELDS} for row in selected]
    append_rows(folder / "metadata.csv", METADATA_FIELDS, metadata)
    queue = args.output / "queue.csv"
    new_rows = queue_rows(selected, queue)
    append_rows(queue, QUEUE_FIELDS, new_rows)

    bands = {"high (>=0.5)": 0, "medium (0.3-0.5)": 0, "low (<0.3)": 0}
    for track in tracks:
        if track["priority"] >= 0.5:
            bands["high (>=0.5)"] += 1
        elif track["priority"] >= 0.3:
            bands["medium (0.3-0.5)"] += 1
        else:
            bands["low (<0.3)"] += 1
    report = dict(
        url=args.url,
        camera=camera,
        start_seconds=args.start,
        title=title,
        run=run,
        source_fps=fps,
        frames_read=frames,
        every=args.every,
        imgsz=args.imgsz,
        confidence=args.confidence,
        min_height=min_height,
        plate_reads=reads,
        tracks_seen=len(tracker.tracks),
        tracks_kept=len(tracks),
        crops_saved=len(selected),
        queue_rows_added=len(new_rows),
        priority_bands=bands,
        two_line_tracks=sum(track["two_line"] for track in tracks),
        top_tracks=[
            {key: track[key] for key in ("track_id", "priority", "consensus", "agreement", "observations", "two_line")}
            for track in tracks[:15]
        ],
    )  # fmt: skip
    text = json.dumps(report, indent=2, ensure_ascii=False)
    (folder / f"summary-{run}.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
