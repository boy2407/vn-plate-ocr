"""Associate plate observations in time, then summarize observed OCR strings."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import hypot


def normalized_text(text: str) -> str:
    return "".join(char for char in text.upper() if char.isalnum())


def representative_key(item: dict) -> tuple:
    confidence = item["ocr_confidence"]
    quality = item.get("crop_quality", {}).get("score")
    score = confidence if quality is None else 0.65 * quality + 0.35 * confidence
    return (score, confidence, item["width"] * item["height"], item["detection_confidence"])


def association_score(a: list, b: list) -> float:
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    if min(aw, ah, bw, bh) <= 0:
        return 0.0
    size_ratio = max(aw / bw, bw / aw, ah / bh, bh / ah)
    if size_ratio > 2.5:
        return 0.0
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    iou = intersection / (aw * ah + bw * bh - intersection)
    distance = hypot(
        (a[0] + a[2] - b[0] - b[2]) / (aw + bw), (a[1] + a[3] - b[1] - b[3]) / (ah + bh)
    )
    if iou >= 0.2 or distance <= 0.5:
        return iou + 0.25 / (1 + distance)
    return 0.0


@dataclass
class PlateTrack:
    track_id: int
    box: list
    first_time: float
    last_time: float
    observations: int = 0
    counts: dict = field(default_factory=lambda: defaultdict(int))
    scores: dict = field(default_factory=lambda: defaultdict(float))
    best: dict = field(default_factory=dict)
    best_ocr: dict = field(default_factory=dict)
    recent: deque = field(default_factory=lambda: deque(maxlen=5))

    def add(self, item: dict):
        text = normalized_text(item["text"])
        item["track_id"] = self.track_id
        self.recent.append(item.copy())
        self.observations += 1
        self.counts[text] += 1
        self.scores[text] += item["ocr_confidence"]
        if text not in self.best or representative_key(item) > representative_key(self.best[text]):
            self.best[text] = item
        if (
            text not in self.best_ocr
            or item["ocr_confidence"] > self.best_ocr[text]["ocr_confidence"]
        ):
            self.best_ocr[text] = item
        self.box = item["box"]
        self.last_time = item["time_seconds"]

    def summary(self, selection: str = "consensus") -> dict:
        if selection == "average":
            return self.average_summary()
        if selection == "ocr":
            text = max(self.best_ocr, key=lambda value: self.best_ocr[value]["ocr_confidence"])
            item = self.best_ocr[text].copy()
        else:
            text = max(self.counts, key=lambda value: (self.counts[value], self.scores[value]))
            item = self.best[text].copy()
        item.update(
            observations=self.observations,
            first_time=self.first_time,
            last_time=self.last_time,
            text_votes=self.counts[text],
            text_agreement=self.counts[text] / self.observations,
            track_ids=[self.track_id],
            track_count=1,
            selection=selection,
        )
        return item

    def average_summary(self) -> dict:
        """Vote over at most five actual observations, never average different strings."""
        grouped = defaultdict(list)
        for row in self.recent:
            grouped[normalized_text(row["text"])].append(row)
        text = max(
            grouped,
            key=lambda value: (
                len(grouped[value]),
                sum(r["ocr_confidence"] for r in grouped[value]) / len(grouped[value]),
            ),
        )
        matching = grouped[text]
        item = max(matching, key=representative_key).copy()
        mean = sum(r["ocr_confidence"] for r in matching) / len(matching)
        item.update(
            representative_ocr_confidence=item["ocr_confidence"],
            ocr_confidence=mean,
            mean_ocr_confidence=mean,
            averaged_frames=len(matching),
            window_frames=len(self.recent),
            window_size=5,
            window_first_time=self.recent[0]["time_seconds"],
            window_last_time=self.recent[-1]["time_seconds"],
            observations=self.observations,
            first_time=self.first_time,
            last_time=self.last_time,
            text_votes=len(matching),
            text_agreement=len(matching) / len(self.recent),
            track_ids=[self.track_id],
            track_count=1,
            selection="average",
        )
        return item


class PlateTracker:
    """Light spatial association for the already-selected one box per frame.

    A gap over one second starts another track. This is an OCR review aid, not
    vehicle identity tracking. Exact consensus texts are merged for the gallery;
    individual track IDs and all frame observations remain available separately.
    """

    def __init__(self, max_gap: float = 1.0, selection: str = "consensus"):
        if selection not in {"consensus", "ocr", "average"}:
            raise ValueError("selection must be consensus, ocr or average")
        self.max_gap = max_gap
        self.selection = selection
        self.tracks: list[PlateTrack] = []
        self.active: list[PlateTrack] = []

    def observe(self, item: dict):
        now = item["time_seconds"]
        self.active = [track for track in self.active if 0 <= now - track.last_time <= self.max_gap]
        candidates = [(association_score(track.box, item["box"]), track) for track in self.active]
        score, track = max(candidates, key=lambda pair: pair[0]) if candidates else (0, None)
        if score <= 0:
            track = PlateTrack(len(self.tracks) + 1, item["box"], now, now)
            self.tracks.append(track)
            self.active.append(track)
        track.add(item)
        return track.summary(self.selection)

    def summaries(self) -> list[dict]:
        grouped = {}
        for track in self.tracks:
            item = track.summary(self.selection)
            text = normalized_text(item["text"])
            key = text or f"empty-track-{track.track_id}"
            if key not in grouped:
                grouped[key] = item
                continue
            previous = grouped[key]
            rank = (
                (lambda row: row["ocr_confidence"])
                if self.selection in {"ocr", "average"}
                else representative_key
            )
            best = (item if rank(item) > rank(previous) else previous).copy()
            best.update(
                observations=previous["observations"] + item["observations"],
                first_time=min(previous["first_time"], item["first_time"]),
                last_time=max(previous["last_time"], item["last_time"]),
                track_ids=previous["track_ids"] + item["track_ids"],
                track_count=previous["track_count"] + item["track_count"],
            )
            if self.selection != "average":
                best["text_votes"] = previous["text_votes"] + item["text_votes"]
                best["text_agreement"] = best["text_votes"] / best["observations"]
            grouped[key] = best
        return sorted(grouped.values(), key=lambda item: item["first_time"])


def group_observations(observations: list[dict], selection: str = "consensus") -> list[dict]:
    tracker = PlateTracker(selection=selection)
    for item in sorted(observations, key=lambda entry: (entry["time_seconds"], entry["frame"])):
        tracker.observe(item)
    return tracker.summaries()
